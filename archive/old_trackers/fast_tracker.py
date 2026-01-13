"""
Fast Real-Time Object Tracker

Optimized for speed while maintaining accuracy via motion compensation.

Speed tricks:
1. KCF/MOSSE instead of CSRT (10-20x faster)
2. Downscaled tracking (optional)
3. Reduced motion estimation overhead
4. Frame skipping with interpolation (optional)

Usage:
    python fast_tracker.py test_track_dron1.mp4 1314 623 73 46
    
    # Even faster with downscaling:
    python fast_tracker.py test_track_dron1.mp4 1314 623 73 46 --scale 0.5
    
    # Fastest (MOSSE):
    python fast_tracker.py test_track_dron1.mp4 1314 623 73 46 --tracker mosse
"""

import cv2
import numpy as np
import sys
import time
import argparse
from pathlib import Path


class FastMotionEstimator:
    """Lightweight motion estimator optimized for speed"""
    
    def __init__(self, max_corners=100, grid_size=50):
        self.prev_gray = None
        self.max_corners = max_corners
        self.grid_size = grid_size  # For grid-based features (faster than goodFeaturesToTrack)
        
        self.lk_params = dict(
            winSize=(15, 15),  # Smaller window = faster
            maxLevel=3,        # Fewer pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
    
    def _get_grid_points(self, shape, bbox, margin=20):
        """Get evenly spaced grid points (faster than feature detection)"""
        h, w = shape
        x, y, bw, bh = [int(v) for v in bbox]
        
        # Create grid
        xs = np.arange(0, w, self.grid_size)
        ys = np.arange(0, h, self.grid_size)
        xx, yy = np.meshgrid(xs, ys)
        points = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
        
        # Filter out points inside bbox + margin
        x1, y1 = x - margin, y - margin
        x2, y2 = x + bw + margin, y + bh + margin
        
        mask = ~((points[:, 0] >= x1) & (points[:, 0] <= x2) & 
                 (points[:, 1] >= y1) & (points[:, 1] <= y2))
        points = points[mask]
        
        return points.reshape(-1, 1, 2)
    
    def update(self, gray, bbox):
        """Estimate motion. Returns (homography, motion_magnitude)"""
        if self.prev_gray is None:
            self.prev_gray = gray
            return None, 0
        
        # Get grid points (faster than goodFeaturesToTrack)
        prev_pts = self._get_grid_points(gray.shape, bbox)
        
        if len(prev_pts) < 8:
            self.prev_gray = gray
            return None, 0
        
        # Track
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            self.prev_gray = gray
            return None, 0
        
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        
        if len(good_prev) < 8:
            self.prev_gray = gray
            return None, 0
        
        # Homography
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        
        motion = 0
        if inliers is not None and inliers.sum() > 0:
            m = inliers.flatten() == 1
            motion = np.median(np.linalg.norm(good_curr[m] - good_prev[m], axis=1))
        
        self.prev_gray = gray
        return H, motion


def transform_bbox(bbox, H, frame_shape):
    """Transform bbox using homography"""
    x, y, w, h = bbox
    corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
    corners = corners.reshape(-1, 1, 2)
    
    transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    
    x_new = transformed[:, 0].min()
    y_new = transformed[:, 1].min()
    w_new = transformed[:, 0].max() - x_new
    h_new = transformed[:, 1].max() - y_new
    
    fh, fw = frame_shape[:2]
    x_new = max(0, min(x_new, fw - w_new - 1))
    y_new = max(0, min(y_new, fh - h_new - 1))
    
    return (int(x_new), int(y_new), int(max(1, w_new)), int(max(1, h_new)))


class FastTracker:
    """
    Fast motion-compensated tracker.
    
    Tracker speed comparison (approximate):
        MOSSE: 500+ fps
        KCF:   150-300 fps  
        CSRT:  25-50 fps
    """
    
    def __init__(self, tracker_type="kcf", motion_threshold=15, scale=1.0):
        self.tracker_type = tracker_type.lower()
        self.motion_threshold = motion_threshold
        self.scale = scale
        
        self.tracker = None
        self.motion_estimator = FastMotionEstimator()
        self.bbox = None
        self.original_bbox = None  # For scale conversion
        self.lost_count = 0
    
    def _create_tracker(self):
        """Create tracker instance"""
        if self.tracker_type == "mosse":
            return cv2.legacy.TrackerMOSSE_create()
        elif self.tracker_type == "kcf":
            return cv2.TrackerKCF_create()
        elif self.tracker_type == "csrt":
            return cv2.TrackerCSRT_create()
        else:
            return cv2.TrackerKCF_create()
    
    def _scale_bbox(self, bbox, scale):
        """Scale bbox coordinates"""
        x, y, w, h = bbox
        return (int(x * scale), int(y * scale), int(w * scale), int(h * scale))
    
    def _scale_frame(self, frame):
        """Downscale frame for faster processing"""
        if self.scale == 1.0:
            return frame
        return cv2.resize(frame, None, fx=self.scale, fy=self.scale, 
                          interpolation=cv2.INTER_LINEAR)
    
    def initialize(self, frame, bbox):
        """Initialize tracker"""
        self.original_bbox = tuple(int(v) for v in bbox)
        
        # Scale for processing
        if self.scale != 1.0:
            frame = self._scale_frame(frame)
            bbox = self._scale_bbox(bbox, self.scale)
        
        self.bbox = tuple(int(v) for v in bbox)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.motion_estimator.prev_gray = gray
        
        self.tracker = self._create_tracker()
        self.tracker.init(frame, self.bbox)
        self.lost_count = 0
        
        return True
    
    def update(self, frame):
        """
        Update tracker.
        Returns (success, bbox_in_original_scale, motion, fps_estimate)
        """
        t0 = time.perf_counter()
        
        # Scale frame
        if self.scale != 1.0:
            frame = self._scale_frame(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Estimate motion
        H, motion = self.motion_estimator.update(gray, self.bbox)
        
        # Predict bbox
        predicted_bbox = self.bbox
        if H is not None:
            try:
                predicted_bbox = transform_bbox(self.bbox, H, frame.shape)
            except:
                pass
        
        # Reinitialize if large motion
        if motion > self.motion_threshold:
            self.tracker = self._create_tracker()
            self.tracker.init(frame, predicted_bbox)
        
        # Track
        success, box = self.tracker.update(frame)
        
        if success:
            self.bbox = tuple(int(v) for v in box)
            self.lost_count = 0
        else:
            self.lost_count += 1
            self.bbox = predicted_bbox
            
            if self.lost_count < 20:
                self.tracker = self._create_tracker()
                x, y, w, h = self.bbox
                expand = 1.3
                cx, cy = x + w/2, y + h/2
                fh, fw = frame.shape[:2]
                expanded = (
                    int(max(0, cx - w*expand/2)),
                    int(max(0, cy - h*expand/2)),
                    int(min(w*expand, fw)),
                    int(min(h*expand, fh))
                )
                self.tracker.init(frame, expanded)
        
        # Convert back to original scale
        if self.scale != 1.0:
            output_bbox = self._scale_bbox(self.bbox, 1.0 / self.scale)
        else:
            output_bbox = self.bbox
        
        elapsed = time.perf_counter() - t0
        fps = 1.0 / elapsed if elapsed > 0 else 0
        
        return success, output_bbox, motion, fps


def main():
    parser = argparse.ArgumentParser(description="Fast Real-Time Tracker")
    parser.add_argument("video", help="Video file")
    parser.add_argument("x", type=int)
    parser.add_argument("y", type=int)
    parser.add_argument("w", type=int)
    parser.add_argument("h", type=int)
    parser.add_argument("--tracker", "-t", choices=["mosse", "kcf", "csrt"], 
                        default="kcf", help="Tracker type (default: kcf)")
    parser.add_argument("--scale", "-s", type=float, default=1.0,
                        help="Downscale factor (e.g., 0.5 for half resolution)")
    parser.add_argument("--motion-threshold", type=float, default=15,
                        help="Motion threshold for reinitialization")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--no-display", action="store_true", help="Don't show window")
    
    args = parser.parse_args()
    bbox = (args.x, args.y, args.w, args.h)
    
    print(f"Video: {args.video}")
    print(f"Tracker: {args.tracker.upper()}")
    print(f"Scale: {args.scale}")
    print(f"Initial bbox: {bbox}")
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)
    
    # Initialize
    tracker = FastTracker(
        tracker_type=args.tracker,
        motion_threshold=args.motion_threshold,
        scale=args.scale
    )
    tracker.initialize(frame, bbox)
    
    print(f"\n✓ Initialized")
    print("Press 'q' to quit, SPACE to pause\n")
    
    # Output
    fh, fw = frame.shape[:2]
    out_path = args.output or (Path(args.video).stem + f"_tracked_{args.tracker}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, video_fps, (fw, fh))
    
    frame_num = 0
    success_count = 0
    fps_history = []
    
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
        if len(fps_history) > 30:
            fps_history.pop(0)
        avg_fps = np.mean(fps_history)
        
        # Draw
        x, y, w, h = bbox
        color = (0, 255, 0) if success else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)
        
        # Info
        status = "OK" if success else f"LOST"
        info = f"Frame {frame_num} | {status} | {avg_fps:.0f} fps | Motion: {motion:.1f}px"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {args.tracker.upper()} (scale={args.scale})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        writer.write(frame)
        
        if not args.no_display:
            cv2.imshow("Fast Tracker", frame)
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
