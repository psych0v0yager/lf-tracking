"""
Fast Tracker with Robust Motion Compensation

Uses KCF/MOSSE for speed but keeps the same quality motion compensation
from best_tracker_v3 that worked well.

Usage:
    python fast_tracker_v2.py test_track_dron1.mp4 1314 623 73 46
    python fast_tracker_v2.py test_track_dron1.mp4 1314 623 73 46 --tracker mosse
"""

import cv2
import numpy as np
import sys
import time
import argparse
from pathlib import Path
from collections import deque


class MotionEstimator:
    """
    Same robust motion estimator from best_tracker_v3.
    Not optimized for speed - prioritizes accuracy.
    """
    
    def __init__(self):
        self.prev_gray = None
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
    
    def update(self, gray, bbox):
        """
        Estimate motion from previous frame.
        Returns (homography, motion_magnitude)
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None, 0
        
        h, w = gray.shape
        
        # Mask excluding object region
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
            return None, 0
        
        # Track
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
        
        # Homography
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        
        motion = 0
        if inliers is not None and inliers.sum() > 0:
            m = inliers.flatten() == 1
            motion = np.median(np.linalg.norm(good_curr[m] - good_prev[m], axis=1))
        
        self.prev_gray = gray.copy()
        return H, motion


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


class FastTrackerV2:
    """
    Fast tracker (KCF/MOSSE) with robust motion compensation.
    Same motion compensation quality as best_tracker_v3.
    """
    
    def __init__(self, tracker_type="kcf", motion_threshold=20):
        self.tracker_type = tracker_type.lower()
        self.motion_threshold = motion_threshold
        
        self.tracker = None
        self.motion_estimator = MotionEstimator()
        
        self.bbox = None
        self.lost_count = 0
    
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
        
        self.tracker = self._create_tracker()
        self.tracker.init(frame, self.bbox)
        
        self.lost_count = 0
        return True
    
    def update(self, frame):
        t0 = time.perf_counter()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        
        # Estimate camera motion
        H, motion = self.motion_estimator.update(gray, self.bbox)
        
        # Predict bbox based on camera motion
        predicted_bbox = self.bbox
        if H is not None:
            predicted_bbox = transform_bbox(self.bbox, H, frame.shape)
        
        # If significant motion, reinitialize at predicted position
        if motion > self.motion_threshold:
            self.tracker = self._create_tracker()
            self.tracker.init(frame, predicted_bbox)
        
        # Run tracker
        success, box = self.tracker.update(frame)
        
        if success:
            self.bbox = tuple(int(v) for v in box)
            self.lost_count = 0
        else:
            self.lost_count += 1
            self.bbox = predicted_bbox
            
            # Try to recover
            if self.lost_count < 30:
                self.tracker = self._create_tracker()
                x, y, w, h = self.bbox
                expand = 1.0 + 0.1 * self.lost_count
                cx, cy = x + w/2, y + h/2
                fh, fw = frame.shape[:2]
                expanded = (
                    int(max(0, cx - w*expand/2)),
                    int(max(0, cy - h*expand/2)),
                    int(min(w*expand, fw - 1)),
                    int(min(h*expand, fh - 1))
                )
                self.tracker.init(frame, expanded)
        
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        return success, self.bbox, motion, fps


def main():
    parser = argparse.ArgumentParser(description="Fast Tracker V2")
    parser.add_argument("video", help="Video file")
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("w", type=int)
    parser.add_argument("h", type=int)
    parser.add_argument("--tracker", "-t", choices=["mosse", "kcf", "csrt"],
                        default="kcf", help="Base tracker (default: kcf)")
    parser.add_argument("--motion-threshold", type=float, default=20,
                        help="Motion threshold for reinitialization")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--no-display", action="store_true")
    
    args = parser.parse_args()
    bbox = (args.x, args.y, args.w, args.h)
    
    print(f"Video: {args.video}")
    print(f"Tracker: {args.tracker.upper()}")
    print(f"Motion threshold: {args.motion_threshold}")
    print(f"Initial bbox: {bbox}")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)
    
    tracker = FastTrackerV2(
        tracker_type=args.tracker,
        motion_threshold=args.motion_threshold
    )
    tracker.initialize(frame, bbox)
    
    print(f"\n✓ Initialized")
    print("Press 'q' to quit, SPACE to pause\n")
    
    fh, fw = frame.shape[:2]
    out_path = args.output or (Path(args.video).stem + f"_fast_v2_{args.tracker}.mp4")
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
        success, bbox, motion, fps = tracker.update(frame)
        
        if success:
            success_count += 1
        
        fps_history.append(fps)
        avg_fps = np.mean(fps_history)
        
        # Draw
        x, y, w, h = bbox
        color = (0, 255, 0) if success else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)
        
        status = "OK" if success else f"LOST({tracker.lost_count})"
        cv2.putText(frame, f"Frame {frame_num} | {status} | {avg_fps:.0f} fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {args.tracker.upper()} | Motion: {motion:.1f}px",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        writer.write(frame)
        
        if not args.no_display:
            cv2.imshow("Fast Tracker V2", frame)
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
