"""
Robust Single Object Tracker - Fixed Version

Handles missing model files gracefully and provides download instructions.

Usage:
    python best_tracker_v2.py test_track_dron1.mp4 1314 623 73 46
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve


# =============================================================================
# Model Download Helper
# =============================================================================

DASIAMRPN_MODELS = {
    "model": "https://github.com/peterj80/models/raw/main/dasiamrpn_model.onnx",
    "kernel_r1": "https://github.com/peterj80/models/raw/main/dasiamrpn_kernel_r1.onnx",
    "kernel_cls1": "https://github.com/peterj80/models/raw/main/dasiamrpn_kernel_cls1.onnx",
}

NANO_MODELS = {
    "backbone": "https://github.com/peterj80/models/raw/main/nanotrack_backbone_sim.onnx",
    "head": "https://github.com/peterj80/models/raw/main/nanotrack_head_sim.onnx",
}


def download_models(model_dict, model_dir="."):
    """Download model files if they don't exist"""
    model_dir = Path(model_dir)
    model_dir.mkdir(exist_ok=True)
    
    for name, url in model_dict.items():
        filename = url.split("/")[-1]
        filepath = model_dir / filename
        
        if not filepath.exists():
            print(f"Downloading {filename}...")
            try:
                urlretrieve(url, filepath)
                print(f"  ✓ Downloaded {filename}")
            except Exception as e:
                print(f"  ✗ Failed to download {filename}: {e}")
                return False
    return True


def get_best_available_tracker(model_dir="."):
    """
    Try to get the best available tracker in order of preference.
    Returns (tracker_instance, tracker_name) or (None, error_message)
    """
    model_dir = Path(model_dir)
    
    # Option 1: DaSiamRPN - Best deep learning tracker
    try:
        model_path = model_dir / "dasiamrpn_model.onnx"
        kernel_r1 = model_dir / "dasiamrpn_kernel_r1.onnx"
        kernel_cls1 = model_dir / "dasiamrpn_kernel_cls1.onnx"
        
        if model_path.exists() and kernel_r1.exists() and kernel_cls1.exists():
            params = cv2.TrackerDaSiamRPN_Params()
            params.model = str(model_path)
            params.kernel_r1 = str(kernel_r1)
            params.kernel_cls1 = str(kernel_cls1)
            tracker = cv2.TrackerDaSiamRPN_create(params)
            return tracker, "DaSiamRPN"
    except Exception as e:
        print(f"DaSiamRPN not available: {e}")
    
    # Option 2: NanoTrack - Lightweight DL tracker
    try:
        backbone = model_dir / "nanotrack_backbone_sim.onnx"
        head = model_dir / "nanotrack_head_sim.onnx"
        
        if backbone.exists() and head.exists():
            params = cv2.TrackerNano_Params()
            params.backbone = str(backbone)
            params.neckhead = str(head)
            tracker = cv2.TrackerNano_create(params)
            return tracker, "NanoTrack"
    except Exception as e:
        print(f"NanoTrack not available: {e}")
    
    # Option 3: CSRT - Best non-DL tracker (always available)
    try:
        tracker = cv2.TrackerCSRT_create()
        return tracker, "CSRT"
    except Exception as e:
        print(f"CSRT not available: {e}")
    
    # Option 4: KCF - Fast fallback
    try:
        tracker = cv2.TrackerKCF_create()
        return tracker, "KCF"
    except Exception as e:
        print(f"KCF not available: {e}")
    
    return None, "No trackers available"


# =============================================================================
# Motion Compensated Tracker
# =============================================================================

class MotionCompensatedTracker:
    """
    Combines any OpenCV tracker with global motion compensation.
    Critical for handling camera pans with small objects.
    """
    
    def __init__(self, model_dir="."):
        self.model_dir = Path(model_dir)
        self.tracker = None
        self.tracker_name = None
        self.prev_gray = None
        self.bbox = None
        self.lost_count = 0
        
        # Optical flow params for motion estimation
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
    
    def _create_tracker(self):
        """Create a new tracker instance"""
        tracker, name = get_best_available_tracker(self.model_dir)
        if tracker is None:
            raise RuntimeError(f"No tracker available: {name}")
        return tracker, name
    
    def _get_background_features(self, gray, bbox, margin=30):
        """Get features outside the object region for motion estimation"""
        h, w = gray.shape
        mask = np.ones((h, w), dtype=np.uint8) * 255
        
        x, y, bw, bh = [int(v) for v in bbox]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(w, x + bw + margin)
        y2 = min(h, y + bh + margin)
        mask[y1:y2, x1:x2] = 0
        
        return cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
    
    def _estimate_homography(self, prev_gray, curr_gray, bbox):
        """Estimate camera motion as homography transform"""
        prev_pts = self._get_background_features(prev_gray, bbox)
        
        if prev_pts is None or len(prev_pts) < 8:
            return None, 0
        
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, prev_pts, None, **self.lk_params
        )
        
        if curr_pts is None:
            return None, 0
        
        # Filter good matches
        good_prev = prev_pts[status.flatten() == 1]
        good_curr = curr_pts[status.flatten() == 1]
        
        if len(good_prev) < 8:
            return None, 0
        
        # RANSAC homography
        H, inliers = cv2.findHomography(good_prev, good_curr, cv2.RANSAC, 5.0)
        
        # Compute motion magnitude
        motion = 0
        if inliers is not None:
            inlier_mask = inliers.flatten() == 1
            if inlier_mask.sum() > 0:
                displacements = np.linalg.norm(
                    good_curr[inlier_mask] - good_prev[inlier_mask], axis=1
                )
                motion = np.median(displacements)
        
        return H, motion
    
    def _transform_bbox(self, bbox, H, frame_shape):
        """Transform bbox using homography"""
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
        x_new = max(0, min(x_new, fw - w_new))
        y_new = max(0, min(y_new, fh - h_new))
        
        return (int(x_new), int(y_new), int(w_new), int(h_new))
    
    def initialize(self, frame, bbox):
        """Initialize tracker with first frame and bounding box"""
        self.bbox = tuple(int(v) for v in bbox)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        self.tracker, self.tracker_name = self._create_tracker()
        success = self.tracker.init(frame, self.bbox)
        
        self.lost_count = 0
        return success
    
    def update(self, frame):
        """
        Update tracker with new frame.
        Returns (success, bbox, motion_magnitude)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Estimate camera motion
        H, motion = self._estimate_homography(self.prev_gray, gray, self.bbox)
        
        # Predict bbox based on camera motion
        predicted_bbox = self.bbox
        if H is not None:
            predicted_bbox = self._transform_bbox(self.bbox, H, frame.shape)
        
        # If large motion detected, reinitialize tracker at predicted location
        if motion > 20:
            self.tracker, _ = self._create_tracker()
            self.tracker.init(frame, predicted_bbox)
        
        # Run tracker
        success, box = self.tracker.update(frame)
        
        if success:
            self.bbox = tuple(int(v) for v in box)
            self.lost_count = 0
        else:
            self.lost_count += 1
            self.bbox = predicted_bbox
            
            # Try to recover with expanded search
            if self.lost_count < 30:
                self.tracker, _ = self._create_tracker()
                x, y, w, h = self.bbox
                expand = 1.5
                cx, cy = x + w/2, y + h/2
                new_w, new_h = w * expand, h * expand
                fh, fw = frame.shape[:2]
                expanded = (
                    int(max(0, cx - new_w/2)),
                    int(max(0, cy - new_h/2)),
                    int(min(new_w, fw)),
                    int(min(new_h, fh))
                )
                self.tracker.init(frame, expanded)
        
        self.prev_gray = gray
        return success, self.bbox, motion


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 6:
        print("Usage: python best_tracker_v2.py <video> <x> <y> <w> <h>")
        print("Example: python best_tracker_v2.py video.mp4 100 200 50 50")
        sys.exit(1)
    
    video_path = sys.argv[1]
    bbox = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
    
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Video: {video_path}")
    print(f"Initial bbox: {bbox}")
    
    # Check/download models
    model_dir = Path("./models")
    model_dir.mkdir(exist_ok=True)
    
    print("\nChecking for deep learning models...")
    tracker_test, name = get_best_available_tracker(model_dir)
    
    if name == "CSRT":
        print("\nNo DL models found. Would you like to download them? (y/n)")
        print("  - DaSiamRPN (~100MB) - Best accuracy")
        print("  - NanoTrack (~5MB) - Lightweight")
        
        response = input("\nDownload DaSiamRPN? [y/n]: ").strip().lower()
        if response == 'y':
            if download_models(DASIAMRPN_MODELS, model_dir):
                print("✓ DaSiamRPN models downloaded!")
            else:
                print("Download failed, falling back to CSRT")
        else:
            response = input("Download NanoTrack instead? [y/n]: ").strip().lower()
            if response == 'y':
                if download_models(NANO_MODELS, model_dir):
                    print("✓ NanoTrack models downloaded!")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video '{video_path}'")
        sys.exit(1)
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)
    
    # Initialize tracker
    tracker = MotionCompensatedTracker(model_dir)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)
    
    print(f"\n✓ Using tracker: {tracker.tracker_name}")
    print("Press 'q' to quit, SPACE to pause\n")
    
    # Setup output
    out_path = Path(video_path).stem + "_tracked.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    
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
        
        # Draw results
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
    
    print(f"\n{'='*50}")
    print(f"Results: {success_count}/{frame_num} frames ({100*success_count/frame_num:.1f}%)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
