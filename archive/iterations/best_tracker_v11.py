"""
Robust Single Object Tracker - Version 11

Strategy: Use v3's motion compensation approach (which works for initial pan),
but ONLY apply world velocity correction when world velocity is significant.

- Initial pan: world_vel ≈ 0 → no correction → works like v3
- Frame 600: world_vel is significant → apply correction → prevent overshoot

Usage:
    python best_tracker_v11.py test_track_dron1.mp4 1314 623 73 46
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
        """Returns (H, motion_magnitude, motion_vector)"""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None, 0, (0, 0)

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
            return None, 0, (0, 0)

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )

        if curr_pts is None:
            self.prev_gray = gray.copy()
            return None, 0, (0, 0)

        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]

        if len(good_prev) < 8:
            self.prev_gray = gray.copy()
            return None, 0, (0, 0)

        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)

        motion = 0
        motion_vec = (0, 0)
        if inliers is not None:
            inlier_mask = inliers.flatten() == 1
            if inlier_mask.sum() > 1:
                displacements = good_curr[inlier_mask] - good_prev[inlier_mask]
                if displacements.ndim == 2 and displacements.shape[1] >= 2:
                    motion = np.median(np.linalg.norm(displacements, axis=1))
                    motion_vec = (float(np.median(displacements[:, 0])),
                                  float(np.median(displacements[:, 1])))

        self.prev_gray = gray.copy()
        return H, motion, motion_vec


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


class RobustTrackerV11:
    def __init__(self, motion_threshold=20):
        self.motion_threshold = motion_threshold

        self.tracker = None
        self.tracker_name = None
        self.camera_motion = CameraMotionEstimator()

        self.bbox = None
        self.prev_bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None

        # Track positions for world velocity estimation
        self.position_history = deque(maxlen=15)
        self.cam_motion_history = deque(maxlen=15)

        self.template = None
        self.last_good_bbox = None

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.prev_bbox = self.bbox
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

    def _get_world_velocity(self):
        """
        Estimate world velocity from position history and camera motion.
        Only returns significant velocity if we have enough confident data.
        """
        if len(self.position_history) < 5 or len(self.cam_motion_history) < 5:
            return (0, 0), 0  # Not enough data

        # Frame velocities from position changes
        positions = list(self.position_history)
        frame_vels = []
        for i in range(1, len(positions)):
            vx = positions[i][0] - positions[i-1][0]
            vy = positions[i][1] - positions[i-1][1]
            frame_vels.append((vx, vy))

        # Average frame velocity
        avg_frame_vx = np.mean([v[0] for v in frame_vels[-5:]])
        avg_frame_vy = np.mean([v[1] for v in frame_vels[-5:]])

        # Average camera motion
        cam_motions = list(self.cam_motion_history)[-5:]
        avg_cam_vx = np.mean([v[0] for v in cam_motions])
        avg_cam_vy = np.mean([v[1] for v in cam_motions])

        # World velocity = frame velocity - background motion
        world_vx = avg_frame_vx - avg_cam_vx
        world_vy = avg_frame_vy - avg_cam_vy

        magnitude = np.sqrt(world_vx**2 + world_vy**2)

        return (world_vx, world_vy), magnitude

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

        # Get camera motion
        H, cam_motion, cam_motion_vec = self.camera_motion.update(gray, self.bbox)
        self.cam_motion_history.append(cam_motion_vec)

        # Compute motion-compensated prediction (like v3)
        if H is not None:
            try:
                motion_pred = transform_bbox(self.bbox, H, frame.shape)
            except Exception:
                motion_pred = self.bbox
        else:
            motion_pred = self.bbox

        # Get world velocity estimate
        world_vel, world_vel_mag = self._get_world_velocity()

        # KEY: Only apply correction if world velocity is SIGNIFICANT
        # This means: initial pan (world_vel ≈ 0) uses pure motion comp
        #            frame 600 (world_vel significant) applies correction
        WORLD_VEL_THRESHOLD = 3.0  # Only correct if moving > 3 px/frame in world

        if world_vel_mag > WORLD_VEL_THRESHOLD:
            # Apply world velocity correction
            corrected_pred = (
                motion_pred[0] + world_vel[0],
                motion_pred[1] + world_vel[1],
                motion_pred[2],
                motion_pred[3]
            )
            corrected_pred = clamp_bbox(corrected_pred, frame.shape)
            using_correction = True
        else:
            # Use pure motion compensation (like v3)
            corrected_pred = motion_pred
            using_correction = False

        # Reinitialize on high camera motion
        if cam_motion > self.motion_threshold:
            self.tracker, _ = create_tracker()
            self.tracker.init(frame, corrected_pred)

        # Run primary tracker
        try:
            success, box = self.tracker.update(frame)
        except Exception:
            success = False
            box = self.bbox

        self.prev_bbox = self.bbox

        if success:
            x, y, w, h = [int(v) for v in box]
            orig_w, orig_h = self.original_size

            w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
            h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
            cx, cy = x + box[2]/2, y + box[3]/2
            self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

            self.lost_count = 0
            self.last_good_bbox = self.bbox

            # Track position
            self.position_history.append((self.bbox[0] + self.bbox[2]/2,
                                          self.bbox[1] + self.bbox[3]/2))

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

            # Recovery
            pred_cx = corrected_pred[0] + orig_w / 2
            pred_cy = corrected_pred[1] + orig_h / 2

            expand = 4.0 + self.lost_count * 0.5
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

                self.recovery_method = f"search({score:.2f})"
                self.last_good_bbox = self.bbox
                self.position_history.append((mx + orig_w/2, my + orig_h/2))

            else:
                match = self._template_search(gray, (0, 0, fw, fh), threshold=0.15)

                if match:
                    mx, my, mw, mh, score = match
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)

                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"wide({score:.2f})"
                    self.last_good_bbox = self.bbox
                else:
                    self.bbox = corrected_pred
                    self.tracker, _ = create_tracker()
                    self.tracker.init(frame, self.bbox)
                    self.recovery_method = "hold"

        return success or (self.recovery_method is not None), self.bbox, cam_motion, world_vel_mag, using_correction


def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V11")
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

    tracker = RobustTrackerV11()
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"V11: Conditional world velocity correction")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v11.mp4")
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
        success, bbox, cam_motion, world_vel_mag, using_corr = tracker.update(frame)

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

        corr_str = "CORR" if using_corr else "RAW"
        cv2.putText(frame, f"Frame {frame_num} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"CamMotion: {cam_motion:.1f} | WorldVel: {world_vel_mag:.1f} | {corr_str}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Tracking V11", frame)

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
