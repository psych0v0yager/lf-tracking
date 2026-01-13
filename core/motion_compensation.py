"""
Motion Compensation Module

Estimates global camera motion using sparse optical flow and RANSAC homography.
Adapted from best_tracker_v3.py with improvements for velocity estimation.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class MotionCompensator:
    """
    Estimates camera motion and provides bbox transformation.

    Uses:
    - Sparse optical flow (Lucas-Kanade) on background features
    - RANSAC homography for robust motion estimation
    - Excludes object region from motion estimation
    """

    def __init__(self,
                 max_corners: int = 300,
                 quality_level: float = 0.01,
                 min_distance: int = 20,
                 win_size: Tuple[int, int] = (21, 21),
                 max_level: int = 4):
        """
        Args:
            max_corners: Maximum feature points to track
            quality_level: Quality threshold for feature detection
            min_distance: Minimum distance between features
            win_size: Optical flow window size
            max_level: Pyramid levels for optical flow
        """
        self.prev_gray = None
        self.prev_H = None

        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=7
        )

    def update(self, gray: np.ndarray,
               bbox: Tuple[int, int, int, int]) -> Tuple[Optional[np.ndarray], float]:
        """
        Estimate camera motion from current frame.

        Args:
            gray: Grayscale frame
            bbox: Current object bounding box (to exclude from motion estimation)

        Returns:
            (homography_matrix, motion_magnitude)
            Homography is None if estimation failed
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None, 0.0

        h, w = gray.shape

        # Create mask excluding object region with margin
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = [int(v) for v in bbox]
        margin = max(30, bw // 2, bh // 2)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
        mask[y1:y2, x1:x2] = 0

        # Find features in previous frame
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)

        if prev_pts is None or len(prev_pts) < 8:
            self.prev_gray = gray.copy()
            return self.prev_H, 0.0

        # Track features to current frame
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            self.prev_gray = gray.copy()
            return self.prev_H, 0.0

        # Filter good matches
        good_mask = status.flatten() == 1
        good_prev = prev_pts[good_mask]
        good_curr = curr_pts[good_mask]

        if len(good_prev) < 8:
            self.prev_gray = gray.copy()
            return self.prev_H, 0.0

        # Compute homography with RANSAC
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)

        # Compute motion magnitude from inliers
        motion = 0.0
        if inliers is not None:
            inlier_mask = inliers.flatten() == 1
            if inlier_mask.sum() > 0:
                displacements = good_curr[inlier_mask] - good_prev[inlier_mask]
                motion = float(np.median(np.linalg.norm(displacements, axis=1)))

        self.prev_gray = gray.copy()
        if H is not None:
            self.prev_H = H

        return H, motion

    def transform_bbox(self, bbox: Tuple[int, int, int, int],
                       H: np.ndarray,
                       frame_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Transform bounding box using homography.

        Args:
            bbox: (x, y, w, h)
            H: Homography matrix
            frame_shape: (height, width)

        Returns:
            Transformed bbox clamped to frame bounds
        """
        x, y, w, h = bbox
        corners = np.array([
            [x, y],
            [x + w, y],
            [x + w, y + h],
            [x, y + h]
        ], dtype=np.float32).reshape(-1, 1, 2)

        try:
            transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        except cv2.error:
            return bbox

        x_new = transformed[:, 0].min()
        y_new = transformed[:, 1].min()
        w_new = transformed[:, 0].max() - x_new
        h_new = transformed[:, 1].max() - y_new

        # Clamp to frame bounds
        fh, fw = frame_shape
        x_new = max(0, min(x_new, fw - w_new - 1))
        y_new = max(0, min(y_new, fh - h_new - 1))
        w_new = max(1, min(w_new, fw - x_new))
        h_new = max(1, min(h_new, fh - y_new))

        return (int(x_new), int(y_new), int(w_new), int(h_new))

    def get_translation(self, H: Optional[np.ndarray]) -> Tuple[float, float]:
        """
        Extract translation component from homography.

        Args:
            H: Homography matrix

        Returns:
            (dx, dy) translation
        """
        if H is None:
            return (0.0, 0.0)

        # Translation is in the last column
        dx = H[0, 2]
        dy = H[1, 2]
        return (float(dx), float(dy))

    def reset(self):
        """Reset motion estimator state."""
        self.prev_gray = None
        self.prev_H = None
