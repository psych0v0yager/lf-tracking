"""
Multi-Signal Fusion Module

Combines multiple sources of bbox estimates:
- Primary tracker output
- EKF prediction
- Detection recovery results
- Template matching

Uses phase-dependent weighting for optimal fusion.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from trackers.phase_detector import TrackingPhase


class FusionModule:
    """
    Fuses multiple bbox estimates into a single output.

    Weighting strategy:
    - Stationary: Trust tracker heavily
    - Camera following: Balance tracker and EKF
    - Accelerating: Trust EKF prediction and detection more
    """

    def __init__(self,
                 tracker_weight_base: float = 0.6,
                 ekf_weight_base: float = 0.3,
                 detection_weight_base: float = 0.8,
                 reference_size: Optional[Tuple[int, int]] = None):
        # Reference size to prevent drift
        self.reference_size = reference_size
        """
        Args:
            tracker_weight_base: Base weight for tracker output
            ekf_weight_base: Base weight for EKF prediction
            detection_weight_base: Weight for detection when available
        """
        self.tracker_weight_base = tracker_weight_base
        self.ekf_weight_base = ekf_weight_base
        self.detection_weight_base = detection_weight_base

        # Phase-specific weight modifiers
        self.phase_weights = {
            TrackingPhase.STATIONARY: {
                'tracker': 0.8,
                'ekf': 0.15,
                'detection': 0.5
            },
            TrackingPhase.CAMERA_FOLLOWING: {
                'tracker': 0.6,
                'ekf': 0.3,
                'detection': 0.6
            },
            TrackingPhase.DRONE_ACCELERATING: {
                'tracker': 0.3,
                'ekf': 0.5,
                'detection': 0.9
            },
            TrackingPhase.UNKNOWN: {
                'tracker': 0.5,
                'ekf': 0.3,
                'detection': 0.7
            }
        }

    def fuse(self,
             tracker_bbox: Optional[Tuple[int, int, int, int]],
             tracker_confidence: float,
             ekf_bbox: Tuple[int, int, int, int],
             detection_bbox: Optional[Tuple[int, int, int, int]],
             detection_confidence: float,
             phase: TrackingPhase,
             tracker_success: bool = True) -> Tuple[int, int, int, int]:
        """
        Fuse multiple bbox estimates.

        Args:
            tracker_bbox: Bbox from primary tracker (or None if failed)
            tracker_confidence: Tracker confidence [0, 1]
            ekf_bbox: Bbox from EKF prediction
            detection_bbox: Bbox from detection recovery (or None)
            detection_confidence: Detection confidence [0, 1]
            phase: Current tracking phase
            tracker_success: Whether tracker reported success

        Returns:
            Fused bbox (x, y, w, h)
        """
        weights = self.phase_weights.get(phase, self.phase_weights[TrackingPhase.UNKNOWN])

        # Build list of sources with weights
        sources = []

        # Tracker
        if tracker_bbox is not None and tracker_success:
            w = weights['tracker'] * tracker_confidence
            sources.append(('tracker', tracker_bbox, w))

        # EKF prediction (always available)
        ekf_w = weights['ekf']
        # Boost EKF weight when tracker fails or confidence is low
        if not tracker_success or tracker_confidence < 0.3:
            ekf_w *= 1.5
        sources.append(('ekf', ekf_bbox, ekf_w))

        # Detection
        if detection_bbox is not None:
            det_w = weights['detection'] * detection_confidence
            sources.append(('detection', detection_bbox, det_w))

        if not sources:
            return ekf_bbox

        # Normalize weights
        total_weight = sum(s[2] for s in sources)
        if total_weight == 0:
            return ekf_bbox

        # Weighted average of bbox centers only (NOT sizes)
        fused_cx = 0.0
        fused_cy = 0.0

        for name, bbox, weight in sources:
            x, y, w, h = bbox
            cx = x + w / 2
            cy = y + h / 2
            norm_w = weight / total_weight

            fused_cx += cx * norm_w
            fused_cy += cy * norm_w

        # Use reference size if available, otherwise use EKF size
        if self.reference_size is not None:
            fused_w, fused_h = self.reference_size
        else:
            fused_w, fused_h = ekf_bbox[2], ekf_bbox[3]

        # Convert back to (x, y, w, h)
        fused_x = fused_cx - fused_w / 2
        fused_y = fused_cy - fused_h / 2

        return (int(fused_x), int(fused_y), int(fused_w), int(fused_h))

    def compute_confidence(self,
                           tracker_confidence: float,
                           tracker_success: bool,
                           detection_available: bool,
                           detection_confidence: float,
                           phase: TrackingPhase,
                           frames_since_detection: int) -> float:
        """
        Compute overall tracking confidence.

        Args:
            tracker_confidence: Primary tracker confidence
            tracker_success: Whether tracker reported success
            detection_available: Whether detection found the object
            detection_confidence: Detection confidence
            phase: Current tracking phase
            frames_since_detection: Frames since last successful detection

        Returns:
            Overall confidence [0, 1]
        """
        if not tracker_success and not detection_available:
            # Both failed - confidence decreases with frames lost
            return max(0.1, 0.5 - frames_since_detection * 0.05)

        if detection_available:
            # Detection confirms tracking
            return min(1.0, 0.6 + detection_confidence * 0.4)

        # Only tracker available
        base_conf = tracker_confidence

        # Reduce confidence in acceleration phase if no detection
        if phase == TrackingPhase.DRONE_ACCELERATING:
            base_conf *= 0.7

        return base_conf

    def should_reinitialize(self,
                            tracker_success: bool,
                            tracker_confidence: float,
                            camera_motion: float,
                            phase: TrackingPhase,
                            lost_frames: int) -> bool:
        """
        Determine if tracker should be reinitialized.

        Args:
            tracker_success: Whether tracker reported success
            tracker_confidence: Tracker confidence
            camera_motion: Camera motion magnitude
            phase: Current tracking phase
            lost_frames: Consecutive frames without successful tracking

        Returns:
            True if reinitialization recommended
        """
        # Always reinit if lost too many frames
        if lost_frames > 10:
            return True

        # Reinit on significant camera motion
        if camera_motion > 15:
            return True

        # More aggressive reinit in acceleration phase
        if phase == TrackingPhase.DRONE_ACCELERATING:
            if camera_motion > 10 or tracker_confidence < 0.4:
                return True

        # Reinit if tracker failed
        if not tracker_success and lost_frames > 3:
            return True

        return False

    def get_weight_debug(self, phase: TrackingPhase) -> Dict[str, float]:
        """Get current weights for debugging."""
        return self.phase_weights.get(phase, self.phase_weights[TrackingPhase.UNKNOWN]).copy()
