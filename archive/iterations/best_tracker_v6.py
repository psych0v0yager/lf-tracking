"""
Robust Single Object Tracker - Version 6

Different recovery approach: Motion-based detection.
Instead of template matching (which fails with appearance changes),
detect the drone by finding what's moving differently from the background.

After camera motion compensation, the drone is the only thing still moving.

Usage:
    python best_tracker_v6.py test_track_dron1.mp4 1314 623 73 46 --no-download
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from urllib.request import urlretrieve
import argparse
from collections import deque


# =============================================================================
# Tracker Creation
# =============================================================================

def create_tracker(force_csrt=False):
    return cv2.TrackerCSRT_create(), "CSRT"


# =============================================================================
# Camera Motion Estimator
# =============================================================================

class CameraMotionEstimator:
    def __init__(self):
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7
        )

    def update(self, gray, bbox):
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None, 0, None

        h, w = gray.shape

        # Mask out object region
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = [int(v) for v in bbox]
        margin = max(30, bw, bh)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
        mask[y1:y2, x1:x2] = 0

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)

        if prev_pts is None or len(prev_pts) < 8:
            prev = self.prev_gray.copy()
            self.prev_gray = gray.copy()
            return None, 0, prev

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            prev = self.prev_gray.copy()
            self.prev_gray = gray.copy()
            return None, 0, prev

        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            prev = self.prev_gray.copy()
            self.prev_gray = gray.copy()
            return None, 0, prev

        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)

        motion = 0
        if inliers is not None:
            inlier_mask = inliers.flatten() == 1
            if inlier_mask.sum() > 0:
                motion = np.median(np.linalg.norm(
                    good_curr[inlier_mask] - good_prev[inlier_mask], axis=1
                ))

        prev = self.prev_gray.copy()
        self.prev_gray = gray.copy()
        return H, motion, prev


# =============================================================================
# Motion-Based Object Detector
# =============================================================================

class MotionDetector:
    """
    Detects moving objects by comparing motion-compensated frames.
    After warping previous frame to align with current, any remaining
    differences are moving objects (like our drone).
    """

    def __init__(self, target_size):
        self.target_w, self.target_h = target_size
        self.min_area = (self.target_w * self.target_h) * 0.3
        self.max_area = (self.target_w * self.target_h) * 5.0

    def detect(self, prev_gray, curr_gray, H, search_region=None):
        """
        Find moving objects after camera motion compensation.

        Returns list of (x, y, w, h, score) candidates sorted by score.
        """
        if H is None or prev_gray is None:
            return []

        fh, fw = curr_gray.shape

        # Warp previous frame to align with current (compensate camera motion)
        try:
            prev_warped = cv2.warpPerspective(prev_gray, H, (fw, fh))
        except Exception:
            return []

        # Compute absolute difference - moving objects will show up
        diff = cv2.absdiff(curr_gray, prev_warped)

        # Apply search region mask if provided
        if search_region is not None:
            sx, sy, sw, sh = [int(v) for v in search_region]
            mask = np.zeros_like(diff)
            mask[sy:sy+sh, sx:sx+sw] = 255
            diff = cv2.bitwise_and(diff, mask)

        # Threshold and clean up
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Morphological operations to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter by area (should be similar to drone size)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Filter by aspect ratio (drone is roughly square-ish)
            aspect = w / h if h > 0 else 0
            if aspect < 0.3 or aspect > 3.0:
                continue

            # Score based on area match and intensity in diff image
            area_match = 1.0 - abs(area - self.target_w * self.target_h) / (self.target_w * self.target_h)
            intensity = np.mean(diff[y:y+h, x:x+w]) / 255.0

            score = 0.5 * max(0, area_match) + 0.5 * intensity

            candidates.append((x, y, w, h, score))

        # Sort by score
        candidates.sort(key=lambda c: c[4], reverse=True)
        return candidates


# =============================================================================
# Utility Functions
# =============================================================================

def transform_bbox(bbox, H, frame_shape):
    x, y, w, h = bbox
    corners = np.array([
        [x, y], [x+w, y], [x+w, y+h], [x, y+h]
    ], dtype=np.float32).reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    x_new = transformed[:, 0].min()
    y_new = transformed[:, 1].min()
    w_new = transformed[:, 0].max() - x_new
    h_new = transformed[:, 1].max() - y_new

    fh, fw = frame_shape[:2]
    x_new = max(0, min(x_new, fw - w_new - 1))
    y_new = max(0, min(y_new, fh - h_new - 1))

    return (int(x_new), int(y_new), int(w_new), int(h_new))


def clamp_bbox(bbox, frame_shape):
    x, y, w, h = bbox
    fh, fw = frame_shape[:2]
    x = max(0, min(int(x), fw - w - 1))
    y = max(0, min(int(y), fh - h - 1))
    return (x, y, int(w), int(h))


# =============================================================================
# Main Tracker
# =============================================================================

class RobustTrackerV6:
    """
    CSRT tracker with motion-based recovery.

    When CSRT fails, instead of template matching, we detect the drone
    by finding what's moving differently from the camera-compensated background.
    """

    def __init__(self, motion_threshold=20):
        self.motion_threshold = motion_threshold

        self.tracker = None
        self.tracker_name = None
        self.camera_motion = CameraMotionEstimator()
        self.motion_detector = None

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None
        self.prev_gray = None

        # Velocity tracking
        self.velocity = (0, 0)
        self.position_history = deque(maxlen=10)

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray.copy()

        self.camera_motion.update(gray, self.bbox)
        self.motion_detector = MotionDetector(self.original_size)

        self.tracker, self.tracker_name = create_tracker()

        try:
            success = self.tracker.init(frame, self.bbox)
            if not success:
                print("Warning: Tracker init returned False")
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

    def _update_velocity(self):
        if len(self.position_history) >= 2:
            p1 = self.position_history[-2]
            p2 = self.position_history[-1]
            self.velocity = (p2[0] - p1[0], p2[1] - p1[1])

    def update(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        self.recovery_method = None

        # Get camera motion and previous frame
        H, cam_motion, prev_gray_for_diff = self.camera_motion.update(gray, self.bbox)

        # Camera-compensated prediction
        if H is not None:
            try:
                cam_pred = transform_bbox(self.bbox, H, frame.shape)
            except Exception:
                cam_pred = self.bbox
        else:
            cam_pred = self.bbox

        # Reinitialize on large camera motion
        if cam_motion > self.motion_threshold:
            self.tracker, _ = create_tracker()
            self.tracker.init(frame, cam_pred)

        # Run primary tracker
        try:
            success, box = self.tracker.update(frame)
        except Exception:
            success = False
            box = self.bbox

        if success:
            # Constrain size
            x, y, w, h = [int(v) for v in box]
            orig_w, orig_h = self.original_size
            w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
            h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
            cx, cy = x + box[2]/2, y + box[3]/2
            self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

            self.lost_count = 0

            # Track position for velocity
            self.position_history.append((self.bbox[0] + self.bbox[2]/2,
                                          self.bbox[1] + self.bbox[3]/2))
            self._update_velocity()

        else:
            self.lost_count += 1
            orig_w, orig_h = self.original_size

            # RECOVERY: Use motion detection
            recovered = False

            if prev_gray_for_diff is not None and H is not None:
                # Define search region based on prediction + velocity
                pred_cx = cam_pred[0] + orig_w / 2 + self.velocity[0] * self.lost_count
                pred_cy = cam_pred[1] + orig_h / 2 + self.velocity[1] * self.lost_count

                # Expand search region
                expand = 3.0 + self.lost_count * 0.5
                search_w = orig_w * expand
                search_h = orig_h * expand

                search_region = (
                    int(max(0, pred_cx - search_w)),
                    int(max(0, pred_cy - search_h)),
                    int(min(search_w * 2, fw)),
                    int(min(search_h * 2, fh))
                )

                # Detect moving objects
                candidates = self.motion_detector.detect(
                    prev_gray_for_diff, gray, H, search_region
                )

                if candidates:
                    # Take best candidate
                    cx, cy, cw, ch, score = candidates[0]

                    # Use detected center with original size
                    det_cx = cx + cw / 2
                    det_cy = cy + ch / 2
                    self.bbox = clamp_bbox(
                        (det_cx - orig_w/2, det_cy - orig_h/2, orig_w, orig_h),
                        frame.shape
                    )

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"motion({score:.2f})"
                    recovered = True

                    self.position_history.append((det_cx, det_cy))
                    self._update_velocity()

            # Fallback: wider search without region constraint
            if not recovered and prev_gray_for_diff is not None and H is not None:
                candidates = self.motion_detector.detect(
                    prev_gray_for_diff, gray, H, None  # Full frame
                )

                # Filter candidates by proximity to last known position
                if candidates:
                    last_cx = cam_pred[0] + orig_w / 2
                    last_cy = cam_pred[1] + orig_h / 2

                    # Sort by distance to prediction
                    def dist_score(c):
                        cx = c[0] + c[2] / 2
                        cy = c[1] + c[3] / 2
                        dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)
                        return c[4] - dist / 500  # Balance score and distance

                    candidates.sort(key=dist_score, reverse=True)

                    if candidates:
                        cx, cy, cw, ch, score = candidates[0]
                        det_cx = cx + cw / 2
                        det_cy = cy + ch / 2
                        self.bbox = clamp_bbox(
                            (det_cx - orig_w/2, det_cy - orig_h/2, orig_w, orig_h),
                            frame.shape
                        )

                        self.tracker, _ = create_tracker()
                        self.tracker.init(frame, self.bbox)

                        self.recovery_method = f"wide({score:.2f})"
                        recovered = True

            if not recovered:
                # Last resort: use prediction
                self.bbox = clamp_bbox(cam_pred, frame.shape)
                self.tracker, _ = create_tracker()
                self.tracker.init(frame, self.bbox)
                self.recovery_method = "predict"

        self.prev_gray = gray.copy()
        return success or (self.recovery_method is not None), self.bbox, cam_motion


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V6")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--motion-threshold", type=float, default=20)
    parser.add_argument("--output", "-o", help="Output video path")

    args = parser.parse_args()
    bbox = (args.x, args.y, args.w, args.h)

    print(f"OpenCV version: {cv2.__version__}")
    print(f"Video: {args.video}")
    print(f"Initial bbox: {bbox}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)

    fh, fw = frame.shape[:2]

    tracker = RobustTrackerV6(args.motion_threshold)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"Recovery: Motion-based detection")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v6.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    frame_num = 0
    success_count = 0
    recovery_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        success, bbox, motion = tracker.update(frame)

        if success:
            success_count += 1
        if tracker.recovery_method:
            recovery_count += 1

        x, y, w, h = bbox

        if tracker.recovery_method:
            color = (0, 255, 255)
        elif success:
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        # Draw velocity
        vx, vy = tracker.velocity
        if abs(vx) > 1 or abs(vy) > 1:
            cx, cy = x + w//2, y + h//2
            cv2.arrowedLine(frame, (cx, cy), (int(cx + vx*3), int(cy + vy*3)), (255, 0, 255), 2)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        cv2.putText(frame, f"Frame {frame_num} | {status} | Motion: {motion:.1f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"V6: CSRT + MotionDetection",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Tracking V6", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Results: {success_count}/{frame_num} frames ({100*success_count/frame_num:.1f}%)")
    print(f"Recoveries: {recovery_count}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
