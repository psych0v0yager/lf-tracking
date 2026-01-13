"""
Robust Single Object Tracker - Version 7

Key insight: When camera accelerates (catches up to drone), the drone's
apparent velocity in the frame suddenly flips. We must detect this and
NOT trust velocity predictions during camera acceleration.

Strategy:
- Track camera acceleration (change in camera motion)
- During camera acceleration: use wide symmetric search, ignore velocity
- During stable camera: use velocity-guided search

Usage:
    python best_tracker_v7.py test_track_dron1.mp4 1314 623 73 46
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
        self.motion_history = deque(maxlen=10)
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
        """Returns (H, motion_magnitude, camera_acceleration)"""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.motion_history.append(0)
            return None, 0, 0

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
            self.motion_history.append(0)
            return None, 0, 0

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            self.prev_gray = gray.copy()
            self.motion_history.append(0)
            return None, 0, 0

        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            self.prev_gray = gray.copy()
            self.motion_history.append(0)
            return None, 0, 0

        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)

        motion = 0
        if inliers is not None:
            inlier_mask = inliers.flatten() == 1
            if inlier_mask.sum() > 0:
                motion = np.median(np.linalg.norm(
                    good_curr[inlier_mask] - good_prev[inlier_mask], axis=1
                ))

        # Compute camera acceleration (change in motion)
        self.motion_history.append(motion)
        if len(self.motion_history) >= 3:
            recent = list(self.motion_history)
            # Acceleration = change in motion magnitude
            cam_accel = abs(recent[-1] - recent[-2])
        else:
            cam_accel = 0

        self.prev_gray = gray.copy()
        return H, motion, cam_accel


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


class RobustTrackerV7:
    def __init__(self, motion_threshold=20):
        self.motion_threshold = motion_threshold

        self.tracker = None
        self.tracker_name = None
        self.camera_motion = CameraMotionEstimator()

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None

        # Position/velocity tracking
        self.position_history = deque(maxlen=10)
        self.velocity = (0, 0)

        # Camera acceleration state
        self.cam_accel_cooldown = 0

        # Template for recovery
        self.template = None

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.camera_motion.update(gray, self.bbox)

        # Save template
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
        if len(self.position_history) >= 2:
            p1 = self.position_history[-2]
            p2 = self.position_history[-1]
            # Smooth velocity update
            new_vx = p2[0] - p1[0]
            new_vy = p2[1] - p1[1]
            # Blend with old velocity for stability
            alpha = 0.7
            self.velocity = (
                alpha * new_vx + (1-alpha) * self.velocity[0],
                alpha * new_vy + (1-alpha) * self.velocity[1]
            )

    def _template_search(self, gray, search_region, threshold=0.25):
        """Simple template matching in search region"""
        if self.template is None:
            return None

        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)

        if sw <= self.template.shape[1] or sh <= self.template.shape[0]:
            return None

        region = gray[sy:sy+sh, sx:sx+sw]

        try:
            result = cv2.matchTemplate(region, self.template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= threshold:
                x = sx + max_loc[0]
                y = sy + max_loc[1]
                return (x, y, self.template.shape[1], self.template.shape[0], max_val)
        except Exception:
            pass

        return None

    def update(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        self.recovery_method = None

        # Get camera motion and acceleration
        H, cam_motion, cam_accel = self.camera_motion.update(gray, self.bbox)

        # Detect camera acceleration (camera catching up)
        # This is when we should NOT trust velocity
        is_cam_accelerating = cam_accel > 10 or cam_motion > 30

        if is_cam_accelerating:
            self.cam_accel_cooldown = 5  # Don't trust velocity for next 5 frames
        elif self.cam_accel_cooldown > 0:
            self.cam_accel_cooldown -= 1

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
            x, y, w, h = [int(v) for v in box]
            orig_w, orig_h = self.original_size
            w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
            h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
            cx, cy = x + box[2]/2, y + box[3]/2
            self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

            self.lost_count = 0

            # Update position history and velocity
            self.position_history.append((self.bbox[0] + self.bbox[2]/2,
                                          self.bbox[1] + self.bbox[3]/2))
            self._update_velocity()

            # Update template periodically
            if self.frame_count % 15 == 0:
                x, y, w, h = self.bbox
                if x >= 0 and y >= 0 and x+w <= fw and y+h <= fh:
                    new_template = gray[y:y+h, x:x+w]
                    if new_template.shape == self.template.shape:
                        self.template = cv2.addWeighted(self.template, 0.8, new_template, 0.2, 0)

        else:
            self.lost_count += 1
            orig_w, orig_h = self.original_size

            # RECOVERY STRATEGY
            # Key insight: during camera acceleration, DON'T use velocity prediction
            # because apparent velocity just flipped

            pred_cx = cam_pred[0] + orig_w / 2
            pred_cy = cam_pred[1] + orig_h / 2

            # Only add velocity if camera is stable
            if self.cam_accel_cooldown == 0:
                pred_cx += self.velocity[0] * self.lost_count * 0.5
                pred_cy += self.velocity[1] * self.lost_count * 0.5

            # During camera acceleration: use WIDE SYMMETRIC search
            # During stable camera: use narrower velocity-guided search
            if self.cam_accel_cooldown > 0:
                # Wide symmetric search - don't trust velocity direction
                expand = 5.0 + self.lost_count
                search_w = orig_w * expand
                search_h = orig_h * expand
                search_cx = cam_pred[0] + orig_w / 2  # Just use camera-compensated position
                search_cy = cam_pred[1] + orig_h / 2
            else:
                # Normal search with velocity guidance
                expand = 3.0 + self.lost_count * 0.5
                search_w = orig_w * expand
                search_h = orig_h * expand
                search_cx = pred_cx
                search_cy = pred_cy

            search_region = (
                int(max(0, search_cx - search_w / 2)),
                int(max(0, search_cy - search_h / 2)),
                int(min(search_w, fw)),
                int(min(search_h, fh))
            )

            # Template search
            match = self._template_search(gray, search_region, threshold=0.2)

            if match:
                mx, my, mw, mh, score = match
                self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)

                self.tracker, _ = create_tracker()
                self.tracker.init(frame, self.bbox)

                self.recovery_method = f"template({score:.2f})"

                # Reset velocity during camera acceleration
                if self.cam_accel_cooldown > 0:
                    self.velocity = (0, 0)
                    self.position_history.clear()

                self.position_history.append((self.bbox[0] + orig_w/2, self.bbox[1] + orig_h/2))

            else:
                # Fallback: very wide search
                wide_region = (0, 0, fw, fh)
                match = self._template_search(gray, wide_region, threshold=0.15)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"wide({score:.2f})"
                    self.velocity = (0, 0)
                    self.position_history.clear()
                else:
                    # Last resort: use camera prediction only
                    self.bbox = clamp_bbox((cam_pred[0], cam_pred[1], orig_w, orig_h), frame.shape)
                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)
                    self.recovery_method = "predict"

        return success or (self.recovery_method is not None), self.bbox, cam_motion, cam_accel


def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V7")
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

    tracker = RobustTrackerV7(args.motion_threshold)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"V7: Camera acceleration detection")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v7.mp4")
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
        success, bbox, cam_motion, cam_accel = tracker.update(frame)

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

        # Draw velocity (only if not in acceleration cooldown)
        if tracker.cam_accel_cooldown == 0:
            vx, vy = tracker.velocity
            if abs(vx) > 1 or abs(vy) > 1:
                cx, cy = x + w//2, y + h//2
                cv2.arrowedLine(frame, (cx, cy), (int(cx + vx*3), int(cy + vy*3)), (255, 0, 255), 2)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        accel_warn = " CAM_ACCEL!" if tracker.cam_accel_cooldown > 0 else ""
        cv2.putText(frame, f"Frame {frame_num} | {status} | Motion: {cam_motion:.1f} Accel: {cam_accel:.1f}{accel_warn}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"V7: CamAccel detection + Wide search",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Tracking V7", frame)

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
