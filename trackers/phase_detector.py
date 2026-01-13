"""
Phase Detector for Drone Tracking

Detects which phase of tracking we're in:
1. STATIONARY: Drone on ground, minimal motion
2. CAMERA_FOLLOWING: Camera pans to follow drone, drone relatively centered
3. DRONE_ACCELERATING: Drone moves faster than camera can follow

Each phase requires different tracking strategies.
"""

import numpy as np
from enum import Enum
from typing import Tuple, Optional
from collections import deque


class TrackingPhase(Enum):
    STATIONARY = "stationary"
    CAMERA_FOLLOWING = "camera_following"
    DRONE_ACCELERATING = "drone_accelerating"
    UNKNOWN = "unknown"


class PhaseDetector:
    """
    Detects current tracking phase based on motion patterns.

    Uses:
    - Camera motion magnitude
    - Object's independent velocity (relative to camera)
    - Motion variance over time
    """

    def __init__(self,
                 stationary_camera_threshold: float = 5.0,
                 stationary_object_threshold: float = 3.0,
                 following_ratio_threshold: float = 0.3,
                 history_length: int = 10):
        """
        Args:
            stationary_camera_threshold: Camera motion below this = stationary
            stationary_object_threshold: Object velocity below this = stationary
            following_ratio_threshold: If relative_motion < camera_motion * ratio, camera is following
            history_length: Frames to consider for smoothing
        """
        self.stationary_camera_threshold = stationary_camera_threshold
        self.stationary_object_threshold = stationary_object_threshold
        self.following_ratio_threshold = following_ratio_threshold

        self.camera_motion_history = deque(maxlen=history_length)
        self.object_velocity_history = deque(maxlen=history_length)
        self.relative_motion_history = deque(maxlen=history_length)
        self.phase_history = deque(maxlen=history_length)

        self.current_phase = TrackingPhase.UNKNOWN
        self.frames_in_phase = 0

    def update(self,
               camera_motion: float,
               object_velocity: Tuple[float, float],
               prev_bbox: Optional[Tuple[int, int, int, int]] = None,
               curr_bbox: Optional[Tuple[int, int, int, int]] = None,
               camera_homography: Optional[np.ndarray] = None) -> TrackingPhase:
        """
        Update phase detection with new frame data.

        Args:
            camera_motion: Magnitude of camera motion (pixels)
            object_velocity: Object velocity from EKF (vx, vy)
            prev_bbox: Previous bounding box (for relative motion calc)
            curr_bbox: Current bounding box
            camera_homography: Homography matrix for motion compensation

        Returns:
            Detected phase
        """
        vx, vy = object_velocity
        object_speed = np.sqrt(vx**2 + vy**2)

        # Calculate relative motion (object motion in frame coordinates)
        relative_motion = self._compute_relative_motion(
            prev_bbox, curr_bbox, camera_homography
        )

        # Store history
        self.camera_motion_history.append(camera_motion)
        self.object_velocity_history.append(object_speed)
        self.relative_motion_history.append(relative_motion)

        # Detect phase
        new_phase = self._detect_phase(camera_motion, object_speed, relative_motion)

        # Update phase with hysteresis
        if new_phase != self.current_phase:
            # Require consistent detection before switching phases
            recent_phases = list(self.phase_history)[-5:] if len(self.phase_history) >= 5 else []
            if len(recent_phases) >= 3 and recent_phases.count(new_phase) >= 3:
                self.current_phase = new_phase
                self.frames_in_phase = 0
        else:
            self.frames_in_phase += 1

        self.phase_history.append(new_phase)
        return self.current_phase

    def _compute_relative_motion(self,
                                  prev_bbox: Optional[Tuple[int, int, int, int]],
                                  curr_bbox: Optional[Tuple[int, int, int, int]],
                                  H: Optional[np.ndarray]) -> float:
        """
        Compute object's motion relative to background.

        This is the key metric: if the drone is moving independently of camera,
        this value will be high even when camera is panning.
        """
        if prev_bbox is None or curr_bbox is None:
            return 0.0

        # Current object center
        curr_cx = curr_bbox[0] + curr_bbox[2] / 2
        curr_cy = curr_bbox[1] + curr_bbox[3] / 2

        # Previous object center
        prev_cx = prev_bbox[0] + prev_bbox[2] / 2
        prev_cy = prev_bbox[1] + prev_bbox[3] / 2

        if H is not None:
            # Transform previous center using camera motion
            try:
                prev_pt = np.array([[prev_cx, prev_cy]], dtype=np.float32).reshape(-1, 1, 2)
                compensated = cv2.perspectiveTransform(prev_pt, H).reshape(2)
                compensated_cx, compensated_cy = compensated
            except:
                compensated_cx, compensated_cy = prev_cx, prev_cy
        else:
            compensated_cx, compensated_cy = prev_cx, prev_cy

        # Relative motion = actual motion - expected motion from camera
        relative_dx = curr_cx - compensated_cx
        relative_dy = curr_cy - compensated_cy

        return np.sqrt(relative_dx**2 + relative_dy**2)

    def _detect_phase(self, camera_motion: float, object_speed: float,
                      relative_motion: float) -> TrackingPhase:
        """
        Core phase detection logic.
        """
        # Phase 1: STATIONARY
        # Camera barely moving, object barely moving
        if (camera_motion < self.stationary_camera_threshold and
            object_speed < self.stationary_object_threshold):
            return TrackingPhase.STATIONARY

        # Phase 2: CAMERA_FOLLOWING
        # Camera moving, but object stays relatively centered (low relative motion)
        if camera_motion > self.stationary_camera_threshold:
            if relative_motion < camera_motion * self.following_ratio_threshold:
                return TrackingPhase.CAMERA_FOLLOWING

        # Phase 3: DRONE_ACCELERATING
        # Object moving faster than camera can follow
        # High relative motion indicates drone is outpacing camera
        if relative_motion > camera_motion * self.following_ratio_threshold:
            return TrackingPhase.DRONE_ACCELERATING

        # Default to camera following if camera is moving
        if camera_motion > self.stationary_camera_threshold:
            return TrackingPhase.CAMERA_FOLLOWING

        return TrackingPhase.STATIONARY

    def get_smoothed_metrics(self) -> dict:
        """
        Get smoothed motion metrics for debugging/visualization.
        """
        return {
            "avg_camera_motion": np.mean(self.camera_motion_history) if self.camera_motion_history else 0,
            "avg_object_speed": np.mean(self.object_velocity_history) if self.object_velocity_history else 0,
            "avg_relative_motion": np.mean(self.relative_motion_history) if self.relative_motion_history else 0,
            "current_phase": self.current_phase.value,
            "frames_in_phase": self.frames_in_phase,
        }

    def is_accelerating(self) -> bool:
        """Quick check if in acceleration phase."""
        return self.current_phase == TrackingPhase.DRONE_ACCELERATING

    def is_stationary(self) -> bool:
        """Quick check if in stationary phase."""
        return self.current_phase == TrackingPhase.STATIONARY

    def get_confidence_modifier(self) -> float:
        """
        Get a confidence modifier based on phase stability.

        Returns value between 0.5 and 1.0:
        - Higher when phase is stable (many frames in same phase)
        - Lower when phase is uncertain or transitioning
        """
        if self.frames_in_phase > 10:
            return 1.0
        elif self.frames_in_phase > 5:
            return 0.8
        else:
            return 0.6


# Import cv2 at module level for _compute_relative_motion
try:
    import cv2
except ImportError:
    cv2 = None
