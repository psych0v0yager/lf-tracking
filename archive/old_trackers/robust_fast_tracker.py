"""
Robust Fast Tracker with Jitter Handling and Velocity Prediction

Addresses:
1. Camera jitter causing tracker loss → Temporal smoothing of motion estimation
2. Fast-moving object → Kalman filter velocity prediction + dynamic search region

Usage:
    python robust_fast_tracker.py test_track_dron1.mp4 1314 623 73 46
"""

import cv2
import numpy as np
import sys
import time
import argparse
from pathlib import Path
from collections import deque


class KalmanBoxTracker:
    """
    Kalman filter for bounding box tracking.
    Predicts position and velocity to handle fast-moving objects.
    
    State: [x, y, w, h, vx, vy, vw, vh]
    """
    
    def __init__(self, bbox):
        # State: [x, y, w, h, vx, vy, vw, vh]
        self.kf = cv2.KalmanFilter(8, 4)
        
        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy
            [0, 0, 1, 0, 0, 0, 1, 0],  # w = w + vw
            [0, 0, 0, 1, 0, 0, 0, 1],  # h = h + vh
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vw = vw
            [0, 0, 0, 0, 0, 0, 0, 1],  # vh = vh
        ], dtype=np.float32)
        
        # Measurement matrix (we observe x, y, w, h)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(8, dtype=np.float32) * 0.1
        self.kf.processNoiseCov[4:, 4:] *= 0.5  # Lower noise for velocity
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1.0
        
        # Initial state
        x, y, w, h = bbox
        self.kf.statePost = np.array([x, y, w, h, 0, 0, 0, 0], dtype=np.float32)
        self.kf.errorCovPost = np.eye(8, dtype=np.float32)
    
    def predict(self):
        """Predict next state, returns predicted bbox"""
        state = self.kf.predict().flatten()
        return (int(state[0]), int(state[1]), int(state[2]), int(state[3]))
    
    def update(self, bbox):
        """Update with measurement"""
        measurement = np.array([[bbox[0]], [bbox[1]], [bbox[2]], [bbox[3]]], dtype=np.float32)
        self.kf.correct(measurement)
    
    def get_velocity(self):
        """Get current velocity estimate"""
        state = self.kf.statePost.flatten()
        return (float(state[4]), float(state[5]))  # vx, vy
    
    def get_speed(self):
        """Get speed magnitude"""
        vx, vy = self.get_velocity()
        return float(np.sqrt(vx**2 + vy**2))


class SmoothedMotionEstimator:
    """
    Motion estimator with temporal smoothing to handle jitter.
    
    Key insight: Camera jitter is high-frequency, small amplitude.
    Real camera pans are lower-frequency, larger amplitude.
    We smooth out the jitter while still responding to real pans.
    """
    
    def __init__(self, smooth_alpha=0.3, jitter_threshold=5.0):
        self.prev_gray = None
        self.smooth_alpha = smooth_alpha  # EMA smoothing factor
        self.jitter_threshold = jitter_threshold  # Motion below this is considered jitter
        
        self.smoothed_H = None
        self.motion_history = deque(maxlen=5)
        
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=4,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        self.feature_params = dict(
            maxCorners=200,
            qualityLevel=0.01,
            minDistance=30,
            blockSize=7
        )
    
    def _is_jitter(self, motion):
        """Detect if current motion is jitter based on history"""
        if len(self.motion_history) < 3:
            return False
        
        # Jitter: small, inconsistent motion direction
        recent = list(self.motion_history)
        
        # Check if motion is small
        if motion < self.jitter_threshold:
            return True
        
        # Check variance - jitter has high variance
        variance = np.var(recent)
        mean_motion = np.mean(recent)
        
        # High relative variance = jitter
        if mean_motion > 0 and variance / mean_motion > 0.5:
            return True
        
        return False
    
    def update(self, gray, bbox):
        """
        Estimate motion with jitter filtering.
        Returns (homography, motion_magnitude, is_jitter)
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None, 0, False
        
        h, w = gray.shape
        
        # Mask excluding object
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = [int(v) for v in bbox]
        margin = max(30, bw, bh)
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
        mask[y1:y2, x1:x2] = 0
        
        # Find features
        prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
        
        if prev_pts is None or len(prev_pts) < 10:
            self.prev_gray = gray.copy()
            return self.smoothed_H, 0, False
        
        # Track
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            self.prev_gray = gray.copy()
            return self.smoothed_H, 0, False
        
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        
        if len(good_prev) < 8:
            self.prev_gray = gray.copy()
            return self.smoothed_H, 0, False
        
        # Compute homography
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        
        # Compute motion magnitude
        motion = 0
        if inliers is not None and inliers.sum() > 0:
            m = inliers.flatten() == 1
            motion = np.median(np.linalg.norm(good_curr[m] - good_prev[m], axis=1))
        
        self.motion_history.append(motion)
        is_jitter = self._is_jitter(motion)
        
        # Smooth the homography
        if H is not None:
            if self.smoothed_H is None:
                self.smoothed_H = H.copy()
            else:
                # For jitter, use more smoothing (lower alpha)
                alpha = self.smooth_alpha * 0.3 if is_jitter else self.smooth_alpha
                self.smoothed_H = alpha * H + (1 - alpha) * self.smoothed_H
        
        self.prev_gray = gray.copy()
        
        # Return smoothed homography for jitter, raw for real motion
        output_H = self.smoothed_H if is_jitter else H
        return output_H, motion, is_jitter


def transform_bbox(bbox, H, frame_shape):
    """Transform bbox using homography"""
    x, y, w, h = bbox
    corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    corners = corners.reshape(-1, 1, 2)
    
    try:
        transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except:
        return bbox
    
    x_new = transformed[:, 0].min()
    y_new = transformed[:, 1].min()
    w_new = transformed[:, 0].max() - x_new
    h_new = transformed[:, 1].max() - y_new
    
    fh, fw = frame_shape[:2]
    x_new = max(0, min(x_new, fw - w_new - 1))
    y_new = max(0, min(y_new, fh - h_new - 1))
    
    return (int(x_new), int(y_new), int(max(1, w_new)), int(max(1, h_new)))


class RobustFastTracker:
    """
    Tracker combining:
    1. Fast base tracker (KCF/MOSSE)
    2. Smoothed motion estimation (handles jitter)
    3. Kalman filter (handles fast-moving objects)
    4. Dynamic search region (expands when object moving fast)
    """
    
    def __init__(self, tracker_type="kcf", motion_threshold=15):
        self.tracker_type = tracker_type.lower()
        self.motion_threshold = motion_threshold
        
        self.tracker = None
        self.motion_estimator = SmoothedMotionEstimator(
            smooth_alpha=0.4,
            jitter_threshold=8.0
        )
        self.kalman = None
        
        self.bbox = None
        self.lost_count = 0
        self.consecutive_jitter = 0
        self.frame_count = 0
        self.warmup_frames = 30  # Be extra aggressive during warmup
    
    def _create_tracker(self):
        if self.tracker_type == "mosse":
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == "kcf":
            return cv2.TrackerKCF_create()
        elif self.tracker_type == "csrt":
            return cv2.TrackerCSRT_create()
        else:
            return cv2.TrackerKCF_create()
    
    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.motion_estimator.prev_gray = gray.copy()
        
        self.kalman = KalmanBoxTracker(self.bbox)
        
        self.tracker = self._create_tracker()
        self.tracker.init(frame, self.bbox)
        
        self.lost_count = 0
        self.consecutive_jitter = 0
        self.frame_count = 0
        return True
    
    def update(self, frame):
        t0 = time.perf_counter()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        
        self.frame_count += 1
        is_warmup = self.frame_count <= self.warmup_frames
        
        # 1. Kalman predict (for fast-moving objects)
        kalman_pred = self.kalman.predict()
        object_speed = self.kalman.get_speed()
        
        # 2. Estimate camera motion (with jitter filtering)
        H, camera_motion, is_jitter = self.motion_estimator.update(gray, self.bbox)
        
        # During warmup, don't treat anything as jitter - respond to all motion
        if is_warmup:
            is_jitter = False
        
        if is_jitter:
            self.consecutive_jitter += 1
        else:
            self.consecutive_jitter = 0
        
        # 3. Compute predicted bbox
        # During warmup or real motion: always apply motion compensation
        if H is not None and (is_warmup or not is_jitter):
            motion_pred = transform_bbox(self.bbox, H, frame.shape)
        else:
            motion_pred = self.bbox
        
        # Blend Kalman and motion prediction based on object speed
        # During warmup: trust motion compensation more
        if is_warmup:
            alpha = 0.2  # Trust motion compensation during warmup
        else:
            alpha = min(1.0, object_speed / 20.0)
        
        predicted_bbox = (
            int(alpha * kalman_pred[0] + (1 - alpha) * motion_pred[0]),
            int(alpha * kalman_pred[1] + (1 - alpha) * motion_pred[1]),
            int(alpha * kalman_pred[2] + (1 - alpha) * motion_pred[2]),
            int(alpha * kalman_pred[3] + (1 - alpha) * motion_pred[3]),
        )
        
        # 4. Dynamic search region based on object speed
        base_expand = 1.0
        speed_expand = min(0.5, object_speed / 30.0)
        
        # Extra expansion during warmup
        if is_warmup:
            base_expand = 1.5
        
        # 5. Decide whether to reinitialize tracker
        should_reinit = False
        
        # During warmup: always reinit if any significant motion
        if is_warmup and camera_motion > 5:
            should_reinit = True
        # After warmup: reinit if large non-jitter camera motion
        elif camera_motion > self.motion_threshold and not is_jitter:
            should_reinit = True
        # Reinit if object is moving fast
        elif object_speed > 15:
            should_reinit = True
        
        # Don't reinit during jitter (unless warmup)
        if self.consecutive_jitter > 2 and not is_warmup:
            should_reinit = False
        
        if should_reinit:
            self.tracker = self._create_tracker()
            
            # Expand search region for fast objects
            x, y, w, h = predicted_bbox
            expand = base_expand + speed_expand
            cx, cy = x + w/2, y + h/2
            new_w, new_h = w * (1 + expand), h * (1 + expand)
            
            init_bbox = (
                int(max(0, cx - new_w/2)),
                int(max(0, cy - new_h/2)),
                int(min(new_w, fw - 1)),
                int(min(new_h, fh - 1))
            )
            self.tracker.init(frame, init_bbox)
        
        # 6. Run tracker
        success, box = self.tracker.update(frame)
        
        if success:
            self.bbox = tuple(int(v) for v in box)
            self.kalman.update(self.bbox)
            self.lost_count = 0
        else:
            self.lost_count += 1
            
            # Use Kalman prediction when lost
            self.bbox = kalman_pred
            self.bbox = (
                max(0, min(self.bbox[0], fw - self.bbox[2])),
                max(0, min(self.bbox[1], fh - self.bbox[3])),
                self.bbox[2],
                self.bbox[3]
            )
            
            if self.lost_count < 30:
                self.tracker = self._create_tracker()
                expand = 1.5 + 0.1 * self.lost_count
                x, y, w, h = self.bbox
                cx, cy = x + w/2, y + h/2
                expanded = (
                    int(max(0, cx - w*expand/2)),
                    int(max(0, cy - h*expand/2)),
                    int(min(w*expand, fw)),
                    int(min(h*expand, fh))
                )
                self.tracker.init(frame, expanded)
        
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        debug_info = {
            "camera_motion": camera_motion,
            "is_jitter": is_jitter,
            "object_speed": object_speed,
            "consecutive_jitter": self.consecutive_jitter,
            "is_warmup": is_warmup,
        }
        
        return success, self.bbox, fps, debug_info


def main():
    parser = argparse.ArgumentParser(description="Robust Fast Tracker")
    parser.add_argument("video", help="Video file")
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("w", type=int)
    parser.add_argument("h", type=int)
    parser.add_argument("--tracker", "-t", choices=["mosse", "kcf", "csrt"],
                        default="kcf", help="Base tracker (default: kcf)")
    parser.add_argument("--motion-threshold", type=float, default=15)
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--no-display", action="store_true")
    
    args = parser.parse_args()
    bbox = (args.x, args.y, args.w, args.h)
    
    print(f"Video: {args.video}")
    print(f"Base tracker: {args.tracker.upper()}")
    print(f"Initial bbox: {bbox}")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)
    
    tracker = RobustFastTracker(
        tracker_type=args.tracker,
        motion_threshold=args.motion_threshold
    )
    tracker.initialize(frame, bbox)
    
    print(f"\n✓ Initialized with jitter handling + Kalman prediction")
    print("Press 'q' to quit, SPACE to pause\n")
    
    fh, fw = frame.shape[:2]
    out_path = args.output or (Path(args.video).stem + f"_robust_{args.tracker}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, video_fps, (fw, fh))
    
    frame_num = 0
    success_count = 0
    fps_history = deque(maxlen=30)
    
    total_start = time.perf_counter()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        success, bbox, fps, debug = tracker.update(frame)
        
        if success:
            success_count += 1
        
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # Draw
        x, y, w, h = bbox
        color = (0, 255, 0) if success else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)
        
        # Info
        status = "OK" if success else f"LOST({tracker.lost_count})"
        jitter_str = "JITTER" if debug["is_jitter"] else ""
        warmup_str = "WARMUP" if debug["is_warmup"] else ""
        
        cv2.putText(frame, f"Frame {frame_num} | {status} | {avg_fps:.0f} fps {warmup_str}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"CamMotion: {debug['camera_motion']:.1f} | ObjSpeed: {debug['object_speed']:.1f} {jitter_str}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        writer.write(frame)
        
        if not args.no_display:
            cv2.imshow("Robust Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
    
    total_time = time.perf_counter() - total_start
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"\n{'='*50}")
    print(f"Results:")
    print(f"  Frames: {frame_num}")
    print(f"  Success: {success_count}/{frame_num} ({100*success_count/frame_num:.1f}%)")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average FPS: {frame_num/total_time:.1f}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
