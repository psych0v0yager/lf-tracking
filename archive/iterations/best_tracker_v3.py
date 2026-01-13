"""
Robust Single Object Tracker - Fixed Version 3

Now with correct model URLs from OpenCV Zoo and proper fallback.

Usage:
    python best_tracker_v3.py test_track_dron1.mp4 1314 623 73 46
    
    # Skip model download, use CSRT directly:
    python best_tracker_v3.py test_track_dron1.mp4 1314 623 73 46 --no-download
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve
import argparse


# =============================================================================
# Correct Model URLs from OpenCV Zoo
# =============================================================================

# https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_dasiamrpn
DASIAMRPN_MODELS = {
    "dasiamrpn_model.onnx": 
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_model_2021nov.onnx",
    "dasiamrpn_kernel_r1.onnx": 
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_kernel_r1_2021nov.onnx",
    "dasiamrpn_kernel_cls1.onnx": 
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx",
}

# https://github.com/opencv/opencv_zoo/tree/main/models/object_tracking_vittrack
VITTRACK_MODEL = {
    "vittrack.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
}


def download_file(url, filepath):
    """Download a file with progress indication"""
    print(f"  Downloading {filepath.name}...")
    try:
        urlretrieve(url, filepath)
        print(f"  ✓ Done")
        return True
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_dasiamrpn(model_dir):
    """Download DaSiamRPN models"""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    success = True
    for filename, url in DASIAMRPN_MODELS.items():
        filepath = model_dir / filename
        if not filepath.exists():
            if not download_file(url, filepath):
                success = False
    return success


def download_vittrack(model_dir):
    """Download ViTTrack model (newer, single file)"""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    for filename, url in VITTRACK_MODEL.items():
        filepath = model_dir / filename
        if not filepath.exists():
            return download_file(url, filepath)
    return True


# =============================================================================
# Tracker Creation
# =============================================================================

def create_tracker(model_dir=".", force_csrt=False):
    """
    Create the best available tracker.
    Returns (tracker, name) or raises exception.

    Args:
        model_dir: Directory containing model files
        force_csrt: If True, always use CSRT regardless of available models
    """
    if force_csrt:
        tracker = cv2.TrackerCSRT_create()
        return tracker, "CSRT"

    model_dir = Path(model_dir)

    # Option 1: ViTTrack (newest, best single-file option)
    vittrack_path = model_dir / "vittrack.onnx"
    if vittrack_path.exists():
        try:
            params = cv2.TrackerVit_Params()
            params.net = str(vittrack_path)
            tracker = cv2.TrackerVit_create(params)
            return tracker, "ViTTrack"
        except Exception as e:
            print(f"ViTTrack failed: {e}")

    # Option 2: DaSiamRPN
    model_path = model_dir / "dasiamrpn_model.onnx"
    kernel_r1 = model_dir / "dasiamrpn_kernel_r1.onnx"
    kernel_cls1 = model_dir / "dasiamrpn_kernel_cls1.onnx"

    if model_path.exists() and kernel_r1.exists() and kernel_cls1.exists():
        try:
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = str(model_path)
            params.kernel_r1 = str(kernel_r1)
            params.kernel_cls1 = str(kernel_cls1)
            tracker = cv2.TrackerDaSiamRPN_create(params)
            return tracker, "DaSiamRPN"
        except Exception as e:
            print(f"DaSiamRPN failed: {e}")

    # Option 3: CSRT (always available, no models needed)
    tracker = cv2.TrackerCSRT_create()
    return tracker, "CSRT"


# =============================================================================
# Motion Compensation
# =============================================================================

class MotionEstimator:
    """Estimates global camera motion using optical flow"""
    
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
        """
        Estimate motion from previous frame.
        Returns (homography, motion_magnitude) or (None, 0)
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None, 0
        
        h, w = gray.shape
        
        # Create mask excluding object region
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
            return None, 0
        
        # Track features
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            self.prev_gray = gray.copy()
            return None, 0
        
        # Filter good matches
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        
        if len(good_prev) < 8:
            self.prev_gray = gray.copy()
            return None, 0
        
        # Compute homography
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        
        # Compute motion magnitude
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
    """Transform bounding box using homography"""
    x, y, w, h = bbox
    corners = np.array([
        [x, y], [x+w, y], [x+w, y+h], [x, y+h]
    ], dtype=np.float32).reshape(-1, 1, 2)
    
    transformed = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    
    x_new = transformed[:, 0].min()
    y_new = transformed[:, 1].min()
    w_new = transformed[:, 0].max() - x_new
    h_new = transformed[:, 1].max() - y_new
    
    # Clamp to frame
    fh, fw = frame_shape[:2]
    x_new = max(0, min(x_new, fw - w_new - 1))
    y_new = max(0, min(y_new, fh - h_new - 1))
    w_new = max(1, min(w_new, fw - x_new))
    h_new = max(1, min(h_new, fh - y_new))
    
    return (int(x_new), int(y_new), int(w_new), int(h_new))


# =============================================================================
# Main Tracker Class
# =============================================================================

class RobustTracker:
    """Motion-compensated tracker for small objects with camera motion"""

    def __init__(self, model_dir="./models", motion_threshold=20, force_csrt=False):
        self.model_dir = Path(model_dir)
        self.motion_threshold = motion_threshold
        self.force_csrt = force_csrt

        self.tracker = None
        self.tracker_name = None
        self.motion_estimator = MotionEstimator()

        self.bbox = None
        self.lost_count = 0

    def initialize(self, frame, bbox):
        """Initialize with first frame and bounding box (x, y, w, h)"""
        self.bbox = tuple(int(v) for v in bbox)

        # Initialize motion estimator
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.motion_estimator.update(gray, self.bbox)

        # Create and initialize tracker
        self.tracker, self.tracker_name = create_tracker(self.model_dir, self.force_csrt)
        
        try:
            success = self.tracker.init(frame, self.bbox)
            if not success:
                print(f"Warning: Tracker init returned False")
            self.lost_count = 0
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False
    
    def update(self, frame):
        """
        Update tracker.
        Returns (success, bbox, motion_magnitude)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Estimate camera motion
        H, motion = self.motion_estimator.update(gray, self.bbox)
        
        # Predict bbox based on camera motion
        predicted_bbox = self.bbox
        if H is not None:
            try:
                predicted_bbox = transform_bbox(self.bbox, H, frame.shape)
            except Exception:
                predicted_bbox = self.bbox
        
        # If large motion, reinitialize at predicted position
        if motion > self.motion_threshold:
            self.tracker, _ = create_tracker(self.model_dir, self.force_csrt)
            self.tracker.init(frame, predicted_bbox)
        
        # Run tracker
        try:
            success, box = self.tracker.update(frame)
        except Exception as e:
            print(f"Tracker update error: {e}")
            success = False
            box = self.bbox
        
        if success:
            self.bbox = tuple(int(v) for v in box)
            self.lost_count = 0
        else:
            self.lost_count += 1
            self.bbox = predicted_bbox
            
            # Try to recover
            if self.lost_count < 30:
                self.tracker, _ = create_tracker(self.model_dir, self.force_csrt)
                # Expand search region
                x, y, w, h = self.bbox
                expand = 1.0 + 0.1 * self.lost_count
                cx, cy = x + w/2, y + h/2
                new_w, new_h = w * expand, h * expand
                fh, fw = frame.shape[:2]
                expanded = (
                    int(max(0, cx - new_w/2)),
                    int(max(0, cy - new_h/2)),
                    int(min(new_w, fw - 1)),
                    int(min(new_h, fh - 1))
                )
                self.tracker.init(frame, expanded)
        
        return success, self.bbox, motion


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--no-download", action="store_true", 
                        help="Skip model download, use CSRT")
    parser.add_argument("--motion-threshold", type=float, default=20,
                        help="Motion threshold for reinitialization")
    parser.add_argument("--output", "-o", help="Output video path")
    
    args = parser.parse_args()
    bbox = (args.x, args.y, args.w, args.h)
    
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Video: {args.video}")
    print(f"Initial bbox: {bbox}")
    
    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)
    
    # Check for / download models
    if not args.no_download:
        tracker_test, name = create_tracker(model_dir)
        
        if name == "CSRT":
            print("\nNo DL models found. Download options:")
            print("  1. ViTTrack (~4MB) - Newest, recommended")
            print("  2. DaSiamRPN (~100MB) - Classic, accurate")
            print("  3. Skip - Use CSRT with motion compensation")
            
            choice = input("\nChoice [1/2/3]: ").strip()
            
            if choice == "1":
                if download_vittrack(model_dir):
                    print("✓ ViTTrack downloaded!")
                else:
                    print("Download failed, using CSRT")
            elif choice == "2":
                if download_dasiamrpn(model_dir):
                    print("✓ DaSiamRPN downloaded!")
                else:
                    print("Download failed, using CSRT")
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)
    
    # Validate bbox
    fh, fw = frame.shape[:2]
    if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > fw or bbox[1] + bbox[3] > fh:
        print(f"Warning: bbox {bbox} may be outside frame {fw}x{fh}")
    
    # Initialize tracker
    tracker = RobustTracker(model_dir, args.motion_threshold, force_csrt=args.no_download)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)
    
    print(f"\n✓ Tracker: {tracker.tracker_name}")
    print(f"✓ Motion threshold: {args.motion_threshold}px")
    print("\nPress 'q' to quit, SPACE to pause\n")
    
    # Output video
    out_path = args.output or (Path(args.video).stem + "_tracked.mp4")
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
        success, bbox, motion = tracker.update(frame)
        
        if success:
            success_count += 1
        
        # Draw
        x, y, w, h = bbox
        color = (0, 255, 0) if success else (0, 165, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)
        
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
    
    print(f"\n{'='*50}")
    print(f"Results: {success_count}/{frame_num} frames ({100*success_count/frame_num:.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
