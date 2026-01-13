"""
Robust Single Object Tracker - Version 9

Key insight:
- Initial pan: drone stationary, moves WITH background → full motion comp helps
- Frame 600: drone moving independently, moves LESS than background → motion comp overshoots

Solution: PARTIAL motion compensation
- Don't apply 100% of predicted motion
- Blend between old position and motion-compensated position
- This prevents overshoot while still helping with camera pans

Usage:
    python best_tracker_v9.py test_track_dron1.mp4 1314 623 73 46
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


def blend_bbox(bbox1, bbox2, alpha, frame_shape):
    """Blend two bboxes: alpha * bbox1 + (1-alpha) * bbox2"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    cx1, cy1 = x1 + w1/2, y1 + h1/2
    cx2, cy2 = x2 + w2/2, y2 + h2/2

    cx = alpha * cx1 + (1 - alpha) * cx2
    cy = alpha * cy1 + (1 - alpha) * cy2
    w = alpha * w1 + (1 - alpha) * w2
    h = alpha * h1 + (1 - alpha) * h2

    return clamp_bbox((cx - w/2, cy - h/2, w, h), frame_shape)


class RobustTrackerV9:
    def __init__(self, motion_threshold=20, motion_blend=0.6):
        self.motion_threshold = motion_threshold
        self.motion_blend = motion_blend  # How much to trust motion comp (0-1)

        self.tracker = None
        self.tracker_name = None
        self.camera_motion = CameraMotionEstimator()

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None

        # Track object velocity to detect independent motion
        self.position_history = deque(maxlen=10)
        self.object_velocity = (0, 0)

        # Template for recovery
        self.template = None
        self.last_good_bbox = None

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])
        self.last_good_bbox = self.bbox

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.camera_motion.update(gray, self.bbox)

        x, y, w, h = self.bbox
        self.template = gray[y:y+h, x:x+w].copy()

        self.tracker, self.tracker_name = create_tracker()

        try:
            self.tracker.init(frame, self.bbox)
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

    def _update_velocity(self):
        """Compute object's velocity in frame coordinates"""
        if len(self.position_history) >= 2:
            p1 = self.position_history[-2]
            p2 = self.position_history[-1]
            vx = p2[0] - p1[0]
            vy = p2[1] - p1[1]
            # Smooth
            self.object_velocity = (
                0.7 * vx + 0.3 * self.object_velocity[0],
                0.7 * vy + 0.3 * self.object_velocity[1]
            )

    def _get_object_speed(self):
        return np.sqrt(self.object_velocity[0]**2 + self.object_velocity[1]**2)

    def _template_search(self, gray, search_region, threshold=0.25):
        if self.template is None:
            return None

        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)

        if sw <= self.template.shape[1] or sh <= self.template.shape[0]:
            return None

        region = gray[sy:sy+sh, sx:sx+sw]

        best_val = 0
        best_loc = None
        best_scale = 1.0
        th, tw = self.template.shape[:2]

        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new_h, new_w = int(th * scale), int(tw * scale)
            if new_h >= region.shape[0] or new_w >= region.shape[1]:
                continue
            if new_h < 5 or new_w < 5:
                continue

            try:
                scaled = cv2.resize(self.template, (new_w, new_h))
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

    def update(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        self.recovery_method = None

        H, cam_motion = self.camera_motion.update(gray, self.bbox)

        # Compute motion-compensated prediction
        if H is not None:
            try:
                motion_pred = transform_bbox(self.bbox, H, frame.shape)
            except Exception:
                motion_pred = self.bbox
        else:
            motion_pred = self.bbox

        # KEY: Adaptive blend based on object velocity
        # If object is moving fast (independent motion), trust motion comp LESS
        # If object is slow/stationary, trust motion comp MORE
        obj_speed = self._get_object_speed()

        # High object speed = moving independently = less motion comp trust
        # Low object speed = moving with camera = more motion comp trust
        if obj_speed > 10:
            effective_blend = 0.3  # Object moving fast, don't trust motion comp much
        elif obj_speed > 5:
            effective_blend = 0.5
        else:
            effective_blend = self.motion_blend  # Default: trust motion comp

        # Reinitialize on high camera motion, but with BLENDED position
        if cam_motion > self.motion_threshold:
            # Blend current bbox with motion prediction
            blended_bbox = blend_bbox(motion_pred, self.bbox, effective_blend, frame.shape)

            self.tracker, _ = create_tracker()
            self.tracker.init(frame, blended_bbox)

        # Run primary tracker
        try:
            success, box = self.tracker.update(frame)
        except Exception:
            success = False
            box = self.bbox

        if success:
            x, y, w, h = [int(v) for v in box]
            orig_w, orig_h = self.original_size

            w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
            h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
            cx, cy = x + box[2]/2, y + box[3]/2
            self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

            self.lost_count = 0
            self.last_good_bbox = self.bbox

            # Update position history and velocity
            self.position_history.append((self.bbox[0] + self.bbox[2]/2,
                                          self.bbox[1] + self.bbox[3]/2))
            self._update_velocity()

            # Update template
            if self.frame_count % 15 == 0:
                x, y, w, h = self.bbox
                if x >= 0 and y >= 0 and x+w <= fw and y+h <= fh:
                    new_template = gray[y:y+h, x:x+w]
                    if new_template.shape[0] > 0 and new_template.shape[1] > 0:
                        if self.template is not None and new_template.shape == self.template.shape:
                            self.template = cv2.addWeighted(self.template, 0.8, new_template, 0.2, 0)
                        else:
                            self.template = new_template.copy()

        else:
            self.lost_count += 1
            orig_w, orig_h = self.original_size

            # Recovery: search around last good position first
            if self.last_good_bbox is not None:
                lgx, lgy, lgw, lgh = self.last_good_bbox
                lg_cx, lg_cy = lgx + lgw/2, lgy + lgh/2

                expand = 4.0 + self.lost_count * 0.5
                search_w = orig_w * expand
                search_h = orig_h * expand

                search_region = (
                    int(max(0, lg_cx - search_w / 2)),
                    int(max(0, lg_cy - search_h / 2)),
                    int(min(search_w, fw)),
                    int(min(search_h, fh))
                )

                match = self._template_search(gray, search_region, threshold=0.2)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"local({score:.2f})"
                    self.last_good_bbox = self.bbox
                    self.position_history.append((mx + orig_w/2, my + orig_h/2))

            # Try motion-compensated search
            if self.recovery_method is None and H is not None:
                pred_cx = motion_pred[0] + orig_w / 2
                pred_cy = motion_pred[1] + orig_h / 2

                expand = 5.0 + self.lost_count * 0.5
                search_w = orig_w * expand
                search_h = orig_h * expand

                search_region = (
                    int(max(0, pred_cx - search_w / 2)),
                    int(max(0, pred_cy - search_h / 2)),
                    int(min(search_w, fw)),
                    int(min(search_h, fh))
                )

                match = self._template_search(gray, search_region, threshold=0.2)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"motion({score:.2f})"
                    self.last_good_bbox = self.bbox

            # Wide search
            if self.recovery_method is None:
                match = self._template_search(gray, (0, 0, fw, fh), threshold=0.15)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"wide({score:.2f})"
                    self.last_good_bbox = self.bbox

            if self.recovery_method is None:
                if self.last_good_bbox is not None:
                    self.bbox = self.last_good_bbox
                self.tracker, _ = create_tracker()
                self.tracker.init(frame, self.bbox)
                self.recovery_method = "hold"

        return success or (self.recovery_method is not None), self.bbox, cam_motion, obj_speed


def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V9")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
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

    tracker = RobustTrackerV9()
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"V9: Adaptive motion blend based on object velocity")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v9.mp4")
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
        success, bbox, cam_motion, obj_speed = tracker.update(frame)

        if success:
            success_count += 1

        x, y, w, h = bbox

        if tracker.recovery_method:
            color = (0, 255, 255)
        elif success:
            color = (0, 255, 0)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        cv2.putText(frame, f"Frame {frame_num} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"CamMotion: {cam_motion:.1f} | ObjSpeed: {obj_speed:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Tracking V9", frame)

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
