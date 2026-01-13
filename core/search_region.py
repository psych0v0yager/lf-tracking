"""
Adaptive Search Region Generator

Dynamically expands the search region based on:
- Tracking phase (stationary/following/accelerating)
- Object acceleration magnitude
- Tracker confidence
- EKF uncertainty
"""

import numpy as np
from typing import Tuple, Optional
from trackers.phase_detector import TrackingPhase


class AdaptiveSearchRegion:
    """
    Generates adaptive search regions for tracker reinitialization.

    Key insight: During drone acceleration (phase 3), we need to expand
    the search region aggressively in the direction of motion.
    """

    def __init__(self,
                 min_expansion: float = 1.0,
                 max_expansion: float = 4.0,
                 base_expansion_stationary: float = 1.2,
                 base_expansion_following: float = 1.5,
                 base_expansion_accelerating: float = 2.0):
        """
        Args:
            min_expansion: Minimum expansion factor
            max_expansion: Maximum expansion factor
            base_expansion_*: Base expansion for each phase
        """
        self.min_expansion = min_expansion
        self.max_expansion = max_expansion
        self.base_expansions = {
            TrackingPhase.STATIONARY: base_expansion_stationary,
            TrackingPhase.CAMERA_FOLLOWING: base_expansion_following,
            TrackingPhase.DRONE_ACCELERATING: base_expansion_accelerating,
            TrackingPhase.UNKNOWN: base_expansion_following,
        }

    def compute(self,
                bbox: Tuple[int, int, int, int],
                phase: TrackingPhase,
                confidence: float,
                acceleration: Tuple[float, float] = (0, 0),
                velocity: Tuple[float, float] = (0, 0),
                uncertainty: Tuple[float, float] = (0, 0),
                frame_shape: Tuple[int, int] = (1080, 1920)) -> Tuple[int, int, int, int]:
        """
        Compute adaptive search region.

        Args:
            bbox: Current bounding box (x, y, w, h)
            phase: Current tracking phase
            confidence: Tracker confidence [0, 1]
            acceleration: (ax, ay) from EKF
            velocity: (vx, vy) from EKF
            uncertainty: (sigma_x, sigma_y) from EKF
            frame_shape: (height, width)

        Returns:
            Search region (x, y, w, h)
        """
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        fh, fw = frame_shape

        # Base expansion from phase
        base_expand = self.base_expansions.get(phase, 1.5)

        # Confidence modifier: lower confidence = larger search
        # confidence in [0, 1], we want expansion in [1, 2]
        confidence_expand = 1.0 + (1.0 - confidence)

        # Acceleration modifier: higher acceleration = larger search
        ax, ay = acceleration
        accel_mag = np.sqrt(ax**2 + ay**2)
        accel_expand = 1.0 + accel_mag / 50.0  # 50 px/frame^2 = double expansion

        # Velocity modifier for directional expansion
        vx, vy = velocity
        speed = np.sqrt(vx**2 + vy**2)

        # Uncertainty modifier
        sigma_x, sigma_y = uncertainty
        uncertainty_expand = 1.0 + (sigma_x + sigma_y) / 100.0

        # Combine factors
        if phase == TrackingPhase.DRONE_ACCELERATING:
            # In acceleration phase, be very aggressive
            total_expand = base_expand * confidence_expand * accel_expand * uncertainty_expand
        elif phase == TrackingPhase.CAMERA_FOLLOWING:
            # Moderate expansion
            total_expand = base_expand * confidence_expand
        else:
            # Stationary: minimal expansion
            total_expand = base_expand

        # Clamp expansion
        total_expand = max(self.min_expansion, min(self.max_expansion, total_expand))

        # Compute search region size
        search_w = w * total_expand
        search_h = h * total_expand

        # For acceleration phase, bias search region in direction of motion
        if phase == TrackingPhase.DRONE_ACCELERATING and speed > 5:
            # Predict where object will be
            predict_frames = 2  # Look 2 frames ahead
            predict_x = cx + vx * predict_frames + 0.5 * ax * predict_frames**2
            predict_y = cy + vy * predict_frames + 0.5 * ay * predict_frames**2

            # Center search region between current and predicted position
            search_cx = (cx + predict_x) / 2
            search_cy = (cy + predict_y) / 2

            # Expand in direction of motion
            dx = abs(predict_x - cx)
            dy = abs(predict_y - cy)
            search_w = max(search_w, w + dx * 2)
            search_h = max(search_h, h + dy * 2)
        else:
            search_cx = cx
            search_cy = cy

        # Compute search region bounds
        search_x = int(search_cx - search_w / 2)
        search_y = int(search_cy - search_h / 2)
        search_w = int(search_w)
        search_h = int(search_h)

        # Clamp to frame bounds
        search_x = max(0, min(search_x, fw - search_w))
        search_y = max(0, min(search_y, fh - search_h))
        search_w = min(search_w, fw - search_x)
        search_h = min(search_h, fh - search_y)

        return (search_x, search_y, search_w, search_h)

    def compute_for_recovery(self,
                             last_bbox: Tuple[int, int, int, int],
                             lost_frames: int,
                             velocity: Tuple[float, float],
                             frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Compute search region for recovery after losing track.

        Args:
            last_bbox: Last known bounding box
            lost_frames: Number of frames since track was lost
            velocity: Last known velocity
            frame_shape: (height, width)

        Returns:
            Search region for recovery
        """
        x, y, w, h = last_bbox
        cx, cy = x + w/2, y + h/2
        vx, vy = velocity
        fh, fw = frame_shape

        # Predict where object might be based on last velocity
        predict_cx = cx + vx * lost_frames
        predict_cy = cy + vy * lost_frames

        # Expansion increases with lost frames
        expand = 2.0 + 0.2 * lost_frames
        expand = min(expand, 6.0)  # Cap at 6x expansion

        # Also add velocity-based expansion
        speed = np.sqrt(vx**2 + vy**2)
        velocity_expand = speed * lost_frames * 0.5

        search_w = w * expand + velocity_expand
        search_h = h * expand + velocity_expand

        # Center on predicted position
        search_x = int(predict_cx - search_w / 2)
        search_y = int(predict_cy - search_h / 2)
        search_w = int(search_w)
        search_h = int(search_h)

        # Clamp to frame
        search_x = max(0, min(search_x, fw - 1))
        search_y = max(0, min(search_y, fh - 1))
        search_w = min(search_w, fw - search_x)
        search_h = min(search_h, fh - search_y)

        return (search_x, search_y, search_w, search_h)
