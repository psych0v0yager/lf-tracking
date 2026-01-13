"""
Main Tracking Pipeline

Integrates all components:
- Primary tracker (ViTTrack/CSRT)
- Extended Kalman Filter with acceleration
- Motion compensation
- Phase detection
- Adaptive search regions
- Async detection recovery
- Multi-signal fusion
"""

import cv2
import numpy as np
import time
from typing import Tuple, Optional, Dict
from pathlib import Path

from trackers.ekf_tracker import ExtendedKalmanTracker
from trackers.phase_detector import PhaseDetector, TrackingPhase
from core.motion_compensation import MotionCompensator
from core.search_region import AdaptiveSearchRegion
from fusion.fusion_module import FusionModule


class DroneTrackingPipeline:
    """
    Complete drone tracking pipeline.

    Handles:
    1. Stationary drone on ground
    2. Camera following moving drone
    3. Drone accelerating faster than camera
    """

    def __init__(self,
                 tracker_type: str = "csrt",
                 model_dir: str = "./models",
                 motion_threshold: float = 15.0,
                 use_detection: bool = True,
                 detection_model: Optional[str] = None):
        """
        Args:
            tracker_type: Primary tracker ("csrt", "kcf", "vit")
            model_dir: Directory containing tracker models
            motion_threshold: Camera motion threshold for reinitialization
            use_detection: Whether to use async YOLO detection
            detection_model: Path to YOLO model
        """
        self.tracker_type = tracker_type.lower()
        self.model_dir = Path(model_dir)
        self.motion_threshold = motion_threshold
        self.use_detection = use_detection

        # Core components
        self.tracker = None
        self.tracker_name = None
        self.ekf = None
        self.motion_comp = MotionCompensator()
        self.phase_detector = PhaseDetector()
        self.search_region_gen = AdaptiveSearchRegion()
        self.fusion = FusionModule()

        # Optional detection
        self.detector = None
        if use_detection:
            try:
                from recovery.async_detector import AsyncDetector
                self.detector = AsyncDetector(model_path=detection_model)
            except ImportError:
                print("Warning: Detection recovery not available")
                self.use_detection = False

        # State
        self.bbox = None
        self.prev_bbox = None
        self.lost_count = 0
        self.frame_count = 0
        self.last_detection_bbox = None
        self.frames_since_detection = 0
        self.template = None

        # Timing
        self.last_frame_time = 0
        self.fps_history = []

    def _create_tracker(self):
        """Create OpenCV tracker based on type."""
        if self.tracker_type == "vit":
            vit_path = self.model_dir / "vittrack.onnx"
            if vit_path.exists():
                try:
                    params = cv2.TrackerVit_Params()
                    params.net = str(vit_path)
                    return cv2.TrackerVit_create(params), "ViTTrack"
                except Exception as e:
                    print(f"ViTTrack failed: {e}")

        if self.tracker_type == "kcf":
            return cv2.TrackerKCF_create(), "KCF"

        # Default to CSRT
        return cv2.TrackerCSRT_create(), "CSRT"

    def initialize(self, frame: np.ndarray,
                   bbox: Tuple[int, int, int, int]) -> bool:
        """
        Initialize pipeline with first frame and bounding box.

        Args:
            frame: First BGR frame
            bbox: Initial bounding box (x, y, w, h)

        Returns:
            True if initialization successful
        """
        self.bbox = tuple(int(v) for v in bbox)
        self.prev_bbox = self.bbox
        self.original_size = (self.bbox[2], self.bbox[3])  # Store original w, h

        # Initialize components
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.motion_comp.update(gray, self.bbox)

        # Set reference size in fusion module
        self.fusion.reference_size = self.original_size

        # EKF
        self.ekf = ExtendedKalmanTracker(
            self.bbox,
            dt=1.0,
            acceleration_decay=0.85,
            process_noise_pos=0.5,
            process_noise_vel=1.0,
            process_noise_acc=3.0
        )

        # Primary tracker
        self.tracker, self.tracker_name = self._create_tracker()
        try:
            success = self.tracker.init(frame, self.bbox)
            if not success:
                print("Warning: Tracker init returned False")
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

        # Template
        x, y, w, h = self.bbox
        self.template = gray[y:y+h, x:x+w].copy()

        # Start detection if enabled
        if self.detector:
            self.detector.start()

        self.lost_count = 0
        self.frame_count = 0
        self.last_frame_time = time.time()

        print(f"Pipeline initialized: Tracker={self.tracker_name}, Detection={self.use_detection}")
        return True

    def update(self, frame: np.ndarray) -> Dict:
        """
        Process one frame.

        Args:
            frame: BGR frame

        Returns:
            Dict with tracking results and debug info
        """
        t0 = time.perf_counter()
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]

        # 1. Motion compensation
        H, camera_motion = self.motion_comp.update(gray, self.bbox)

        # 2. EKF predict
        ekf_pred = self.ekf.predict()
        velocity = self.ekf.get_velocity()
        acceleration = self.ekf.get_acceleration()
        accel_mag = self.ekf.get_acceleration_magnitude()

        # 3. Phase detection
        phase = self.phase_detector.update(
            camera_motion=camera_motion,
            object_velocity=velocity,
            prev_bbox=self.prev_bbox,
            curr_bbox=self.bbox,
            camera_homography=H
        )

        # Adapt EKF noise based on phase
        self.ekf.adapt_noise(phase.value)

        # 4. Compute motion-compensated prediction
        if H is not None:
            motion_pred = self.motion_comp.transform_bbox(self.bbox, H, (fh, fw))
        else:
            motion_pred = self.bbox

        # 5. Check if we should reinitialize
        should_reinit = self.fusion.should_reinitialize(
            tracker_success=self.lost_count == 0,
            tracker_confidence=1.0 - self.lost_count * 0.1,
            camera_motion=camera_motion,
            phase=phase,
            lost_frames=self.lost_count
        )

        if should_reinit:
            # Use motion-compensated prediction for reinitialization
            # Keep the original bbox SIZE, just update position
            orig_w, orig_h = self.bbox[2], self.bbox[3]

            # Blend EKF and motion prediction for position
            ekf_cx = ekf_pred[0] + ekf_pred[2] / 2
            ekf_cy = ekf_pred[1] + ekf_pred[3] / 2
            motion_cx = motion_pred[0] + motion_pred[2] / 2
            motion_cy = motion_pred[1] + motion_pred[3] / 2

            # Weight towards motion prediction during camera pans
            if camera_motion > 10:
                alpha = 0.7  # Trust motion compensation more
            else:
                alpha = 0.3

            pred_cx = alpha * motion_cx + (1 - alpha) * ekf_cx
            pred_cy = alpha * motion_cy + (1 - alpha) * ekf_cy

            # Create reinit bbox with ORIGINAL size, predicted position
            reinit_bbox = (
                int(pred_cx - orig_w / 2),
                int(pred_cy - orig_h / 2),
                orig_w,
                orig_h
            )

            # Clamp to frame bounds
            reinit_bbox = (
                max(0, min(reinit_bbox[0], fw - orig_w)),
                max(0, min(reinit_bbox[1], fh - orig_h)),
                orig_w,
                orig_h
            )

            # Reinitialize tracker with predicted bbox (NOT expanded search region)
            self.tracker, _ = self._create_tracker()
            self.tracker.init(frame, reinit_bbox)

        # 6. Run primary tracker
        try:
            tracker_success, tracker_box = self.tracker.update(frame)
        except Exception as e:
            print(f"Tracker error: {e}")
            tracker_success = False
            tracker_box = self.bbox

        # 7. Request detection if confidence is low
        detection_bbox = None
        detection_conf = 0.0

        if self.detector:
            # Request detection
            tracker_conf = 1.0 if tracker_success else max(0, 0.5 - self.lost_count * 0.1)
            self.detector.request_detection(
                frame=frame,
                search_region=self.search_region_gen.compute_for_recovery(
                    self.bbox, self.lost_count, velocity, (fh, fw)
                ) if self.lost_count > 0 else None,
                confidence=tracker_conf
            )

            # Check for detection results
            det_result = self.detector.get_result()
            if det_result and det_result['best_match']:
                detection_bbox = det_result['best_match']['bbox']
                detection_conf = det_result['best_match']['confidence']
                self.last_detection_bbox = detection_bbox
                self.frames_since_detection = 0

        # 8. EKF update with tracker measurement
        if tracker_success:
            tracker_bbox = tuple(int(v) for v in tracker_box)
            self.ekf.update(tracker_bbox)
            self.lost_count = 0
        else:
            self.lost_count += 1
            tracker_bbox = None

        self.frames_since_detection += 1

        # 9. Fusion
        tracker_conf = 1.0 if tracker_success else 0.0
        fused_bbox = self.fusion.fuse(
            tracker_bbox=tracker_bbox,
            tracker_confidence=tracker_conf,
            ekf_bbox=ekf_pred,
            detection_bbox=detection_bbox,
            detection_confidence=detection_conf,
            phase=phase,
            tracker_success=tracker_success
        )

        # 10. Update state
        self.prev_bbox = self.bbox
        self.bbox = fused_bbox

        # Update template periodically
        if tracker_success and self.frame_count % 30 == 0:
            x, y, w, h = self.bbox
            if 0 <= x < fw and 0 <= y < fh and x+w <= fw and y+h <= fh:
                self.template = gray[y:y+h, x:x+w].copy()

        # Compute FPS
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        self.fps_history.append(fps)
        if len(self.fps_history) > 30:
            self.fps_history.pop(0)

        # Overall confidence
        overall_conf = self.fusion.compute_confidence(
            tracker_confidence=tracker_conf,
            tracker_success=tracker_success,
            detection_available=detection_bbox is not None,
            detection_confidence=detection_conf,
            phase=phase,
            frames_since_detection=self.frames_since_detection
        )

        return {
            'bbox': self.bbox,
            'success': tracker_success or detection_bbox is not None,
            'confidence': overall_conf,
            'phase': phase.value,
            'camera_motion': camera_motion,
            'velocity': velocity,
            'acceleration': acceleration,
            'accel_magnitude': accel_mag,
            'lost_count': self.lost_count,
            'fps': fps,
            'avg_fps': np.mean(self.fps_history),
            'tracker_success': tracker_success,
            'detection_available': detection_bbox is not None,
            'ekf_prediction': ekf_pred,
            'motion_prediction': motion_pred,
        }

    def shutdown(self):
        """Clean up resources."""
        if self.detector:
            self.detector.stop()
