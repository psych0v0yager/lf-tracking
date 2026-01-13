"""
Robust Single Object Tracker - Version 4

Based on v3 with added template matching recovery for when CSRT loses track.

Key improvements:
- Template matching recovery when tracker confidence drops
- Multi-scale template search
- Velocity-based position prediction

Usage:
    python best_tracker_v4.py test_track_dron1.mp4 1314 623 73 46

    # Skip model download, use CSRT directly:
    python best_tracker_v4.py test_track_dron1.mp4 1314 623 73 46 --no-download
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
from urllib.request import urlretrieve
import argparse
from collections import deque


# =============================================================================
# Correct Model URLs from OpenCV Zoo
# =============================================================================

DASIAMRPN_MODELS = {
    "dasiamrpn_model.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_model_2021nov.onnx",
    "dasiamrpn_kernel_r1.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_kernel_r1_2021nov.onnx",
    "dasiamrpn_kernel_cls1.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_dasiamrpn/object_tracking_dasiamrpn_kernel_cls1_2021nov.onnx",
}

VITTRACK_MODEL = {
    "vittrack.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
}


def download_file(url, filepath):
    """Download a file with progress indication"""
    print(f"  Downloading {filepath.name}...")
    try:
        urlretrieve(url, filepath)
        print(f"  Done")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
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
    """
    if force_csrt:
        tracker = cv2.TrackerCSRT_create()
        return tracker, "CSRT"

    model_dir = Path(model_dir)

    # Option 1: ViTTrack
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

    # Option 3: CSRT (always available)
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
# Template Matcher for Recovery
# =============================================================================

class TemplateMatcher:
    """Multi-scale template matching for recovery when tracker fails"""

    def __init__(self, scales=None):
        self.template = None
        self.template_size = None
        self.scales = scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        self.update_alpha = 0.1  # Template update rate

    def set_template(self, gray, bbox):
        """Set initial template from bbox region"""
        x, y, w, h = [int(v) for v in bbox]

        # Bounds check
        fh, fw = gray.shape
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = min(w, fw - x)
        h = min(h, fh - y)

        if w > 0 and h > 0:
            self.template = gray[y:y+h, x:x+w].copy()
            self.template_size = (w, h)

    def update_template(self, gray, bbox):
        """Slowly update template with new observation (EMA)"""
        if self.template is None:
            self.set_template(gray, bbox)
            return

        x, y, w, h = [int(v) for v in bbox]

        # Bounds check
        fh, fw = gray.shape
        x = max(0, min(x, fw - 1))
        y = max(0, min(y, fh - 1))
        w = min(w, fw - x)
        h = min(h, fh - y)

        if w <= 0 or h <= 0:
            return

        new_patch = gray[y:y+h, x:x+w]

        # Resize to match template size if needed
        if new_patch.shape != self.template.shape:
            if self.template_size:
                new_patch = cv2.resize(new_patch, self.template_size)

        if new_patch.shape == self.template.shape:
            self.template = cv2.addWeighted(
                self.template, 1 - self.update_alpha,
                new_patch, self.update_alpha, 0
            )

    def match(self, gray, search_region, threshold=0.4):
        """
        Find template in search region using multi-scale matching.

        Args:
            gray: Grayscale frame
            search_region: (x, y, w, h) region to search in
            threshold: Minimum match score

        Returns:
            (x, y, w, h, score) if found, None otherwise
        """
        if self.template is None:
            return None

        sx, sy, sw, sh = [int(v) for v in search_region]

        # Bounds check
        fh, fw = gray.shape
        sx = max(0, min(sx, fw - 1))
        sy = max(0, min(sy, fh - 1))
        sw = min(sw, fw - sx)
        sh = min(sh, fh - sy)

        if sw <= 0 or sh <= 0:
            return None

        region = gray[sy:sy+sh, sx:sx+sw]

        if region.size == 0:
            return None

        best_val = 0
        best_loc = None
        best_scale = 1.0

        th, tw = self.template.shape[:2]

        for scale in self.scales:
            new_h, new_w = int(th * scale), int(tw * scale)

            # Skip if template larger than search region
            if new_h >= region.shape[0] or new_w >= region.shape[1]:
                continue
            if new_h < 5 or new_w < 5:
                continue

            try:
                scaled_template = cv2.resize(self.template, (new_w, new_h))
                result = cv2.matchTemplate(region, scaled_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)

                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale
            except Exception:
                continue

        if best_val >= threshold and best_loc is not None:
            # Convert to full frame coordinates
            w = int(tw * best_scale)
            h = int(th * best_scale)
            x = sx + best_loc[0]
            y = sy + best_loc[1]
            return (x, y, w, h, best_val)

        return None


# =============================================================================
# Velocity Estimator
# =============================================================================

class VelocityEstimator:
    """Estimates object velocity from recent positions"""

    def __init__(self, history_len=5):
        self.positions = deque(maxlen=history_len)

    def update(self, bbox):
        """Update with new bbox position"""
        x, y, w, h = bbox
        cx, cy = x + w/2, y + h/2
        self.positions.append((cx, cy))

    def get_velocity(self):
        """Get estimated velocity (vx, vy) in pixels per frame"""
        if len(self.positions) < 2:
            return (0, 0)

        # Use last few positions to estimate velocity
        positions = list(self.positions)

        # Simple: use difference between last two
        vx = positions[-1][0] - positions[-2][0]
        vy = positions[-1][1] - positions[-2][1]

        return (vx, vy)

    def predict_position(self, bbox, frames_ahead=1):
        """Predict future position based on velocity"""
        vx, vy = self.get_velocity()
        x, y, w, h = bbox

        pred_x = x + vx * frames_ahead
        pred_y = y + vy * frames_ahead

        return (int(pred_x), int(pred_y), w, h)


# =============================================================================
# Main Tracker Class
# =============================================================================

class RobustTrackerV4:
    """
    Motion-compensated tracker with template matching recovery.

    Recovery strategy when tracker fails:
    1. Use motion compensation to predict position
    2. Use velocity estimation to refine prediction
    3. Use template matching in expanded search region
    4. Reinitialize tracker at best match
    """

    def __init__(self, model_dir="./models", motion_threshold=20, force_csrt=False):
        self.model_dir = Path(model_dir)
        self.motion_threshold = motion_threshold
        self.force_csrt = force_csrt

        self.tracker = None
        self.tracker_name = None
        self.motion_estimator = MotionEstimator()
        self.template_matcher = TemplateMatcher()
        self.velocity_estimator = VelocityEstimator()

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None  # For debugging

    def initialize(self, frame, bbox):
        """Initialize with first frame and bounding box (x, y, w, h)"""
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])

        # Initialize motion estimator
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.motion_estimator.update(gray, self.bbox)

        # Initialize template
        self.template_matcher.set_template(gray, self.bbox)

        # Initialize velocity estimator
        self.velocity_estimator.update(self.bbox)

        # Create and initialize tracker
        self.tracker, self.tracker_name = create_tracker(self.model_dir, self.force_csrt)

        try:
            success = self.tracker.init(frame, self.bbox)
            if not success:
                print(f"Warning: Tracker init returned False")
            self.lost_count = 0
            self.frame_count = 0
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

    def update(self, frame):
        """
        Update tracker.
        Returns (success, bbox, motion_magnitude)
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        self.recovery_method = None

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

            # Constrain size to original (prevent drift)
            x, y, w, h = self.bbox
            orig_w, orig_h = self.original_size

            # Allow some size variation but not too much
            w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
            h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))

            # Re-center with constrained size
            cx, cy = x + self.bbox[2]/2, y + self.bbox[3]/2
            self.bbox = (int(cx - w/2), int(cy - h/2), w, h)

            self.lost_count = 0

            # Update template periodically when tracking is good
            if self.frame_count % 10 == 0:
                self.template_matcher.update_template(gray, self.bbox)

            # Update velocity estimator
            self.velocity_estimator.update(self.bbox)

        else:
            self.lost_count += 1

            # Try to recover using template matching
            recovered = False

            if self.lost_count <= 30:
                # Build search region: combine motion prediction + velocity prediction
                vx, vy = self.velocity_estimator.get_velocity()

                # Predicted center from motion compensation
                pred_x, pred_y, pred_w, pred_h = predicted_bbox
                pred_cx = pred_x + pred_w / 2
                pred_cy = pred_y + pred_h / 2

                # Add velocity prediction
                pred_cx += vx * self.lost_count * 0.5
                pred_cy += vy * self.lost_count * 0.5

                # Search region expands with lost count
                expand = 2.0 + 0.3 * self.lost_count
                orig_w, orig_h = self.original_size
                search_w = orig_w * expand
                search_h = orig_h * expand

                search_region = (
                    int(max(0, pred_cx - search_w / 2)),
                    int(max(0, pred_cy - search_h / 2)),
                    int(min(search_w, fw)),
                    int(min(search_h, fh))
                )

                # Try template matching
                match_result = self.template_matcher.match(
                    gray, search_region, threshold=0.35
                )

                if match_result is not None:
                    mx, my, mw, mh, score = match_result

                    # Use original size, matched position
                    self.bbox = (mx, my, orig_w, orig_h)

                    # Reinitialize tracker at matched position
                    self.tracker, _ = create_tracker(self.model_dir, self.force_csrt)
                    self.tracker.init(frame, self.bbox)

                    self.recovery_method = f"template({score:.2f})"
                    recovered = True

                    # Update velocity with recovered position
                    self.velocity_estimator.update(self.bbox)

            if not recovered:
                # Fall back to predicted position
                self.bbox = predicted_bbox

                # Ensure original size
                x, y, _, _ = self.bbox
                orig_w, orig_h = self.original_size
                self.bbox = (x, y, orig_w, orig_h)

                # Reinitialize tracker at predicted position
                self.tracker, _ = create_tracker(self.model_dir, self.force_csrt)
                self.tracker.init(frame, self.bbox)

                self.recovery_method = "predict"

        return success or (self.recovery_method is not None), self.bbox, motion


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V4")
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
                    print("ViTTrack downloaded!")
                else:
                    print("Download failed, using CSRT")
            elif choice == "2":
                if download_dasiamrpn(model_dir):
                    print("DaSiamRPN downloaded!")
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
    tracker = RobustTrackerV4(model_dir, args.motion_threshold, force_csrt=args.no_download)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\n Tracker: {tracker.tracker_name}")
    print(f" Motion threshold: {args.motion_threshold}px")
    print(f" Template matching recovery: enabled")
    print("\nPress 'q' to quit, SPACE to pause\n")

    # Output video
    out_path = args.output or (Path(args.video).stem + "_tracked_v4.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    frame_num = 0
    success_count = 0
    recovery_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        success, bbox, motion = tracker.update(frame)

        if success:
            success_count += 1
        if tracker.recovery_method:
            recovery_count += 1

        # Draw
        x, y, w, h = bbox

        # Color: green=tracking, yellow=recovered, orange=lost
        if tracker.recovery_method:
            color = (0, 255, 255)  # Yellow for recovery
        elif success:
            color = (0, 255, 0)   # Green for tracking
        else:
            color = (0, 165, 255)  # Orange for lost

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        cv2.putText(frame, f"Frame {frame_num} | {status} | Motion: {motion:.1f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {tracker.tracker_name} + TemplateRecovery",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        writer.write(frame)
        cv2.imshow("Tracking V4", frame)

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
    print(f"Recoveries: {recovery_count}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
