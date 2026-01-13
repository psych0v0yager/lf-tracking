"""
Best Pip-Installable Single Object Tracker

Uses state-of-the-art deep learning trackers that are easy to install.

RECOMMENDED INSTALL (pick one):

    # Option 1: OpenCV with DL trackers built-in (easiest)
    pip install opencv-contrib-python

    # Option 2: Norfair (lightweight, good for SOT)
    pip install norfair

    # Option 3: SuperGradients (YOLO-NAS based, very accurate)
    pip install super-gradients

Usage:
    python best_tracker.py video.mp4 100 200 50 50
"""

import cv2
import numpy as np
import sys
from pathlib import Path


def get_best_available_tracker():
    """
    Try to get the best available tracker in order of preference.
    Returns (tracker_instance, tracker_name)
    """
    
    # Option 1: DaSiamRPN - Deep learning tracker in opencv-contrib
    # This is MUCH better than CSRT for your use case
    try:
        tracker = cv2.TrackerDaSiamRPN_create()
        return tracker, "DaSiamRPN"
    except AttributeError:
        pass
    
    # Option 2: Nano tracker - lightweight DL tracker
    try:
        tracker = cv2.TrackerNano_create()
        return tracker, "Nano"
    except AttributeError:
        pass
    
    # Option 3: GOTURN - another DL option
    try:
        tracker = cv2.TrackerGOTURN_create()
        return tracker, "GOTURN"
    except AttributeError:
        pass
    
    # Fallback: CSRT
    tracker = cv2.TrackerCSRT_create()
    return tracker, "CSRT"


class MotionCompensatedDLTracker:
    """
    Combines deep learning tracker with motion compensation.
    Best of both worlds for small objects + camera motion.
    """
    
    def __init__(self):
        self.tracker, self.tracker_name = get_best_available_tracker()
        print(f"Using tracker: {self.tracker_name}")
        
        self.prev_gray = None
        self.bbox = None
        self.lost_count = 0
        
        # Optical flow params
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
    
    def _estimate_camera_motion(self, prev_gray, curr_gray, bbox):
        """Estimate global camera motion, excluding object region"""
        h, w = prev_gray.shape
        
        # Create mask excluding object
        mask = np.ones((h, w), dtype=np.uint8) * 255
        x, y, bw, bh = [int(v) for v in bbox]
        margin = 30
        x1, y1 = max(0, x - margin), max(0, y - margin)
        x2, y2 = min(w, x + bw + margin), min(h, y + bh + margin)
        mask[y1:y2, x1:x2] = 0
        
        # Find features
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=mask, **self.feature_params)
        if prev_pts is None or len(prev_pts) < 8:
            return None, 0
        
        # Track features
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            return None, 0
        
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        
        if len(good_prev) < 8:
            return None, 0
        
        # Compute homography
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        
        # Compute motion magnitude
        motion = 0
        if inliers is not None:
            mask = inliers.flatten() == 1
            if mask.sum() > 0:
                motion = np.median(np.linalg.norm(good_curr[mask] - good_prev[mask], axis=1))
        
        return H, motion
    
    def _transform_bbox(self, bbox, H, frame_shape):
        """Transform bbox using homography and clamp to frame"""
        x, y, w, h = bbox
        corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
        corners = corners.reshape(-1, 1, 2)
        
        transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
        
        x_new = max(0, transformed[:, 0].min())
        y_new = max(0, transformed[:, 1].min())
        w_new = transformed[:, 0].max() - x_new
        h_new = transformed[:, 1].max() - y_new
        
        # Clamp
        fh, fw = frame_shape[:2]
        x_new = min(x_new, fw - w_new)
        y_new = min(y_new, fh - h_new)
        
        return (int(x_new), int(y_new), int(w_new), int(h_new))
    
    def initialize(self, frame, bbox):
        """Initialize tracker"""
        self.bbox = tuple(int(v) for v in bbox)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.tracker.init(frame, self.bbox)
        self.lost_count = 0
        return True
    
    def update(self, frame):
        """Update tracker, returns (success, bbox, motion_magnitude)"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Estimate camera motion
        H, motion = self._estimate_camera_motion(self.prev_gray, gray, self.bbox)
        
        # Predict bbox based on camera motion
        predicted_bbox = self.bbox
        if H is not None:
            predicted_bbox = self._transform_bbox(self.bbox, H, frame.shape)
        
        # If large motion, reinitialize at predicted location
        if motion > 20:
            self.tracker, _ = get_best_available_tracker()
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
                self.tracker, _ = get_best_available_tracker()
                # Expand search region
                x, y, w, h = self.bbox
                expand = 1.5
                cx, cy = x + w/2, y + h/2
                new_w, new_h = w * expand, h * expand
                expanded = (int(cx - new_w/2), int(cy - new_h/2), int(new_w), int(new_h))
                self.tracker.init(frame, expanded)
        
        self.prev_gray = gray
        return success, self.bbox, motion


def main():
    if len(sys.argv) < 6:
        print("Usage: python best_tracker.py <video> <x> <y> <w> <h>")
        print("\nFirst, install opencv-contrib for best results:")
        print("  pip install opencv-contrib-python")
        sys.exit(1)
    
    video_path = sys.argv[1]
    bbox = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    
    # Check what's available
    print("\nChecking available trackers...")
    print(f"OpenCV version: {cv2.__version__}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Cannot open: {video_path}")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Cannot read first frame")
        sys.exit(1)
    
    # Initialize
    tracker = MotionCompensatedDLTracker()
    tracker.initialize(frame, bbox)
    
    print(f"\nTracking with initial bbox: {bbox}")
    print("Press 'q' to quit, SPACE to pause\n")
    
    frame_num = 0
    success_count = 0
    
    # For saving output
    out_path = Path(video_path).stem + "_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_num += 1
        success, bbox, motion = tracker.update(frame)
        
        if success:
            success_count += 1
        
        # Draw
        x, y, bw, bh = bbox
        color = (0, 255, 0) if success else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.circle(frame, (x + bw//2, y + bh//2), 4, color, -1)
        
        status = "TRACKING" if success else f"LOST ({tracker.lost_count})"
        cv2.putText(frame, f"Frame {frame_num} | {status} | Motion: {motion:.1f}px", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {tracker.tracker_name}", 
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        writer.write(frame)
        cv2.imshow("Tracking", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)
    
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    
    print(f"\nResults: {success_count}/{frame_num} frames ({100*success_count/frame_num:.1f}%)")
    print(f"Output saved: {out_path}")


if __name__ == "__main__":
    main()
