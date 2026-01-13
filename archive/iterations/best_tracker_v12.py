"""
Robust Single Object Tracker - Version 12

Key insight: CSRT reports success=True even when tracking wrong thing.
The bbox drifts to background but tracker doesn't know.

Solution: VERIFY each tracking result against original template.
If verification fails, trigger recovery even if CSRT says success.

Usage:
    python best_tracker_v12.py test_track_dron1.mp4 1314 623 73 46
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import argparse
from collections import deque


def create_tracker():
    return cv2.TrackerCSRT_create(), "CSRT"


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
            return None, 0

        h, w = gray.shape

        mask = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = [int(v) for v in bbox]
        margin = max(30, bw, bh)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
        mask[y1:y2, x1:x2] = 0

        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)

        if prev_pts is None or len(prev_pts) < 8:
            self.prev_gray = gray.copy()
            return None, 0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            self.prev_gray = gray.copy()
            return None, 0

        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            self.prev_gray = gray.copy()
            return None, 0

        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)

        motion = 0
        if inliers is not None:
            inlier_mask = inliers.flatten() == 1
            if inlier_mask.sum() > 0:
                motion = np.median(np.linalg.norm(
                    good_curr[inlier_mask] - good_prev[inlier_mask], axis=1
                ))

        self.prev_gray = gray.copy()
        return H, motion


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
    x = max(0, min(int(x), fw - int(w) - 1))
    y = max(0, min(int(y), fh - int(h) - 1))
    return (x, y, int(w), int(h))


class TemplateVerifier:
    """
    Verifies that tracked region still matches the target.
    Detects when tracker has drifted to wrong object.
    """

    def __init__(self):
        self.reference_template = None  # Original template (never changes)
        self.adaptive_template = None   # Slowly updated template
        self.template_size = None

    def set_reference(self, gray, bbox):
        """Set the reference template from initial bbox"""
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)

        if w > 0 and h > 0:
            self.reference_template = gray[y:y+h, x:x+w].copy()
            self.adaptive_template = self.reference_template.copy()
            self.template_size = (w, h)

    def verify(self, gray, bbox, threshold=0.3):
        """
        Check if bbox region matches the reference template.
        Returns (is_valid, match_score)
        """
        if self.reference_template is None:
            return True, 1.0

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)

        if w <= 0 or h <= 0:
            return False, 0.0

        # Extract current region
        current = gray[y:y+h, x:x+w]

        if current.size == 0:
            return False, 0.0

        # Resize to match template if needed
        if current.shape != self.reference_template.shape:
            try:
                current = cv2.resize(current, (self.template_size[0], self.template_size[1]))
            except Exception:
                return False, 0.0

        # Compare with reference using normalized cross-correlation
        try:
            result = cv2.matchTemplate(current, self.reference_template, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0] if result.size > 0 else 0.0
        except Exception:
            # If sizes don't match for template matching, use a different metric
            try:
                # Use correlation coefficient
                score = np.corrcoef(current.flatten(), self.reference_template.flatten())[0, 1]
                if np.isnan(score):
                    score = 0.0
            except Exception:
                score = 0.0

        return score >= threshold, score

    def update_adaptive(self, gray, bbox, alpha=0.05):
        """Slowly update adaptive template when verification passes"""
        if self.adaptive_template is None:
            return

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)

        if w <= 0 or h <= 0:
            return

        current = gray[y:y+h, x:x+w]

        if current.shape != self.adaptive_template.shape:
            try:
                current = cv2.resize(current, (self.template_size[0], self.template_size[1]))
            except Exception:
                return

        if current.shape == self.adaptive_template.shape:
            self.adaptive_template = cv2.addWeighted(
                self.adaptive_template, 1 - alpha,
                current, alpha, 0
            )

    def search(self, gray, search_region, threshold=0.25):
        """Search for template in region"""
        if self.reference_template is None:
            return None

        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)

        th, tw = self.reference_template.shape[:2]

        if sw <= tw or sh <= th:
            return None

        region = gray[sy:sy+sh, sx:sx+sw]

        best_val = 0
        best_loc = None
        best_scale = 1.0

        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new_h, new_w = int(th * scale), int(tw * scale)
            if new_h >= region.shape[0] or new_w >= region.shape[1]:
                continue
            if new_h < 5 or new_w < 5:
                continue

            try:
                scaled = cv2.resize(self.reference_template, (new_w, new_h))
                result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale
            except Exception:
                continue

        if best_val >= threshold and best_loc is not None:
            x = sx + best_loc[0]
            y = sy + best_loc[1]
            w = int(tw * best_scale)
            h = int(th * best_scale)
            return (x, y, w, h, best_val)

        return None


class RobustTrackerV12:
    def __init__(self, motion_threshold=20, verify_threshold=0.25):
        self.motion_threshold = motion_threshold
        self.verify_threshold = verify_threshold

        self.tracker = None
        self.tracker_name = None
        self.camera_motion = CameraMotionEstimator()
        self.verifier = TemplateVerifier()

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None
        self.verify_score = 1.0

        self.last_good_bbox = None

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])
        self.last_good_bbox = self.bbox

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.camera_motion.update(gray, self.bbox)

        # Set reference template
        self.verifier.set_reference(gray, self.bbox)

        self.tracker, self.tracker_name = create_tracker()

        try:
            self.tracker.init(frame, self.bbox)
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

    def update(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        self.recovery_method = None

        # Camera motion estimation
        H, cam_motion = self.camera_motion.update(gray, self.bbox)

        # Motion-compensated prediction
        if H is not None:
            try:
                motion_pred = transform_bbox(self.bbox, H, frame.shape)
            except Exception:
                motion_pred = self.bbox
        else:
            motion_pred = self.bbox

        # Reinitialize on high camera motion
        if cam_motion > self.motion_threshold:
            self.tracker, _ = create_tracker()
            self.tracker.init(frame, motion_pred)

        # Run primary tracker
        try:
            tracker_success, box = self.tracker.update(frame)
        except Exception:
            tracker_success = False
            box = self.bbox

        # CRITICAL: Verify the tracking result
        if tracker_success:
            temp_bbox = tuple(int(v) for v in box)
            is_valid, self.verify_score = self.verifier.verify(gray, temp_bbox, self.verify_threshold)

            if is_valid:
                # Tracking is good
                x, y, w, h = temp_bbox
                orig_w, orig_h = self.original_size

                w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
                h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
                cx, cy = x + box[2]/2, y + box[3]/2
                self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

                self.lost_count = 0
                self.last_good_bbox = self.bbox

                # Update adaptive template slowly
                self.verifier.update_adaptive(gray, self.bbox)

                actual_success = True
            else:
                # CSRT says success but verification FAILED - trigger recovery
                tracker_success = False
                actual_success = False
        else:
            actual_success = False
            self.verify_score = 0.0

        # Recovery if tracking failed or verification failed
        if not actual_success:
            self.lost_count += 1
            orig_w, orig_h = self.original_size

            # Search around motion-compensated prediction
            pred_cx = motion_pred[0] + orig_w / 2
            pred_cy = motion_pred[1] + orig_h / 2

            expand = 4.0 + self.lost_count * 0.5
            search_w = orig_w * expand
            search_h = orig_h * expand

            search_region = (
                int(max(0, pred_cx - search_w / 2)),
                int(max(0, pred_cy - search_h / 2)),
                int(min(search_w, fw)),
                int(min(search_h, fh))
            )

            match = self.verifier.search(gray, search_region, threshold=0.2)

            if match:
                mx, my, mw, mh, score = match
                self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                self.verify_score = score

                self.tracker, _ = create_tracker()
                self.tracker.init(frame, self.bbox)

                self.recovery_method = f"search({score:.2f})"
                self.last_good_bbox = self.bbox
                actual_success = True

            else:
                # Try around last good position
                if self.last_good_bbox is not None:
                    lg_cx = self.last_good_bbox[0] + orig_w / 2
                    lg_cy = self.last_good_bbox[1] + orig_h / 2

                    search_region = (
                        int(max(0, lg_cx - search_w / 2)),
                        int(max(0, lg_cy - search_h / 2)),
                        int(min(search_w, fw)),
                        int(min(search_h, fh))
                    )

                    match = self.verifier.search(gray, search_region, threshold=0.2)

                    if match:
                        mx, my, mw, mh, score = match
                        self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                        self.verify_score = score

                        self.tracker, _ = create_tracker()
                        self.tracker.init(frame, self.bbox)

                        self.recovery_method = f"local({score:.2f})"
                        self.last_good_bbox = self.bbox
                        actual_success = True

            # Wide search as last resort
            if not actual_success:
                match = self.verifier.search(gray, (0, 0, fw, fh), threshold=0.15)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                    self.verify_score = score

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"wide({score:.2f})"
                    self.last_good_bbox = self.bbox
                    actual_success = True
                else:
                    # Hold position
                    self.bbox = motion_pred
                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)
                    self.recovery_method = "hold"

        return actual_success, self.bbox, cam_motion, self.verify_score


def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V12")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--verify-threshold", type=float, default=0.25,
                        help="Template verification threshold")
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

    tracker = RobustTrackerV12(verify_threshold=args.verify_threshold)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"V12: Template verification (threshold={args.verify_threshold})")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v12.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    frame_num = 0
    success_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        success, bbox, cam_motion, verify_score = tracker.update(frame)

        if success:
            success_count += 1

        x, y, w, h = bbox

        # Color based on verify score
        if tracker.recovery_method:
            color = (0, 255, 255)  # Yellow - recovery
        elif verify_score > 0.4:
            color = (0, 255, 0)   # Green - good match
        elif verify_score > 0.25:
            color = (0, 200, 200) # Yellow-green - marginal
        else:
            color = (0, 165, 255) # Orange - poor match

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        cv2.putText(frame, f"Frame {frame_num} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Motion: {cam_motion:.1f} | Verify: {verify_score:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Tracking V12", frame)

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
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
