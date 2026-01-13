"""
Robust Single Object Tracker - Version 5

Key improvement: Object-level optical flow tracking.
Instead of just tracking camera motion, we also track the drone's pixels
directly using dense optical flow. This captures the drone's independent
motion even when it accelerates faster than the camera.

Recovery strategy:
1. Primary: CSRT tracker with motion compensation
2. On failure: Use object optical flow to predict where drone moved
3. Search in a directional region (cone along velocity vector)
4. Template matching as final fallback

Usage:
    python best_tracker_v5.py test_track_dron1.mp4 1314 623 73 46 --no-download
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from urllib.request import urlretrieve
import argparse
from collections import deque


# =============================================================================
# Model URLs
# =============================================================================

VITTRACK_MODEL = {
    "vittrack.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/object_tracking_vittrack/object_tracking_vittrack_2023sep.onnx"
}


def download_file(url, filepath):
    print(f"  Downloading {filepath.name}...")
    try:
        urlretrieve(url, filepath)
        print(f"  Done")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False


def download_vittrack(model_dir):
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
    if force_csrt:
        return cv2.TrackerCSRT_create(), "CSRT"

    model_dir = Path(model_dir)
    vittrack_path = model_dir / "vittrack.onnx"
    if vittrack_path.exists():
        try:
            params = cv2.TrackerVit_Params()
            params.net = str(vittrack_path)
            return cv2.TrackerVit_create(params), "ViTTrack"
        except Exception as e:
            print(f"ViTTrack failed: {e}")

    return cv2.TrackerCSRT_create(), "CSRT"


# =============================================================================
# Camera Motion Estimator (background optical flow)
# =============================================================================

class CameraMotionEstimator:
    """Estimates global camera motion from background features"""

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
        """Returns (homography, motion_magnitude) or (None, 0)"""
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return None, 0

        h, w = gray.shape

        # Mask out object region
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


# =============================================================================
# Object Optical Flow Tracker
# =============================================================================

class ObjectFlowTracker:
    """
    Tracks the object's pixels directly using dense optical flow.
    This captures the object's independent motion (separate from camera).

    Includes outlier rejection to prevent bad frames from corrupting velocity.
    """

    def __init__(self, history_len=10):
        self.prev_gray = None
        self.prev_bbox = None
        self.velocity_history = deque(maxlen=history_len)
        self.last_good_velocity = (0, 0)
        self.confidence = 1.0

        # Farneback parameters for dense flow
        self.flow_params = dict(
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

    def _is_velocity_outlier(self, vx, vy):
        """
        Check if new velocity is an outlier (sudden direction reversal).
        A real object can't reverse direction instantly.
        """
        if len(self.velocity_history) < 2:
            return False

        # Get recent velocity trend
        recent = list(self.velocity_history)[-3:]
        avg_vx = np.mean([v[0] for v in recent])
        avg_vy = np.mean([v[1] for v in recent])

        old_speed = np.sqrt(avg_vx**2 + avg_vy**2)
        new_speed = np.sqrt(vx**2 + vy**2)

        # If both are slow, not an outlier
        if old_speed < 2 and new_speed < 2:
            return False

        # Check direction change using dot product
        if old_speed > 2 and new_speed > 2:
            # Normalize
            old_dir = (avg_vx / old_speed, avg_vy / old_speed)
            new_dir = (vx / new_speed, vy / new_speed)

            # Dot product: 1 = same direction, -1 = opposite
            dot = old_dir[0] * new_dir[0] + old_dir[1] * new_dir[1]

            # If direction flipped more than 120 degrees, it's an outlier
            if dot < -0.5:
                return True

        # Check for sudden speed change (more than 3x)
        if old_speed > 3 and new_speed > old_speed * 3:
            return True

        return False

    def update(self, gray, bbox):
        """
        Compute object's motion using dense optical flow on bbox region.
        Returns (vx, vy) - object velocity in pixels/frame
        """
        if self.prev_gray is None or self.prev_bbox is None:
            self.prev_gray = gray.copy()
            self.prev_bbox = bbox
            return (0, 0)

        # Extract regions
        px, py, pw, ph = [int(v) for v in self.prev_bbox]
        cx, cy, cw, ch = [int(v) for v in bbox]

        fh, fw = gray.shape

        # Expand region slightly to capture motion
        margin = max(pw, ph) // 2

        # Previous frame region (expanded)
        p_x1 = max(0, px - margin)
        p_y1 = max(0, py - margin)
        p_x2 = min(fw, px + pw + margin)
        p_y2 = min(fh, py + ph + margin)

        # Current frame: search in larger region
        search_margin = margin * 3
        c_x1 = max(0, cx - search_margin)
        c_y1 = max(0, cy - search_margin)
        c_x2 = min(fw, cx + cw + search_margin)
        c_y2 = min(fh, cy + ch + search_margin)

        # Make regions same size for flow computation
        # Use intersection approach
        x1 = max(p_x1, c_x1)
        y1 = max(p_y1, c_y1)
        x2 = min(p_x2, c_x2)
        y2 = min(p_y2, c_y2)

        if x2 - x1 < 20 or y2 - y1 < 20:
            self.prev_gray = gray.copy()
            self.prev_bbox = bbox
            return self._get_smoothed_velocity()

        prev_region = self.prev_gray[y1:y2, x1:x2]
        curr_region = gray[y1:y2, x1:x2]

        try:
            # Compute dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                prev_region, curr_region, None, **self.flow_params
            )

            # Focus on the object region within the flow
            obj_y1 = max(0, py - y1)
            obj_y2 = min(flow.shape[0], py + ph - y1)
            obj_x1 = max(0, px - x1)
            obj_x2 = min(flow.shape[1], px + pw - x1)

            if obj_x2 > obj_x1 and obj_y2 > obj_y1:
                obj_flow = flow[obj_y1:obj_y2, obj_x1:obj_x2]

                # Use median flow in object region (robust to outliers)
                vx = np.median(obj_flow[:, :, 0])
                vy = np.median(obj_flow[:, :, 1])

                # Check for outlier (sudden direction reversal)
                if self._is_velocity_outlier(vx, vy):
                    # Reject this measurement, use last good velocity with decay
                    self.confidence *= 0.8
                    vx, vy = self.last_good_velocity
                    vx *= 0.95  # Slight decay
                    vy *= 0.95
                else:
                    self.confidence = min(1.0, self.confidence + 0.1)
                    self.last_good_velocity = (vx, vy)

                self.velocity_history.append((vx, vy))

        except Exception:
            pass

        self.prev_gray = gray.copy()
        self.prev_bbox = bbox

        return self._get_smoothed_velocity()

    def _get_smoothed_velocity(self):
        """Get smoothed velocity from history with momentum"""
        if len(self.velocity_history) == 0:
            return self.last_good_velocity

        vels = list(self.velocity_history)

        # Weight recent samples more, but use median for robustness
        if len(vels) >= 3:
            # Use last 3-5 samples with exponential weighting
            recent = vels[-5:] if len(vels) >= 5 else vels[-3:]
            weights = [0.5 ** (len(recent) - 1 - i) for i in range(len(recent))]
            total_weight = sum(weights)

            vx = sum(v[0] * w for v, w in zip(recent, weights)) / total_weight
            vy = sum(v[1] * w for v, w in zip(recent, weights)) / total_weight
        else:
            vx = np.median([v[0] for v in vels])
            vy = np.median([v[1] for v in vels])

        # Blend with last good velocity for stability
        blend = self.confidence
        vx = blend * vx + (1 - blend) * self.last_good_velocity[0]
        vy = blend * vy + (1 - blend) * self.last_good_velocity[1]

        return (vx, vy)

    def predict_position(self, bbox, frames_ahead=1):
        """Predict future bbox position using object velocity"""
        vx, vy = self._get_smoothed_velocity()
        x, y, w, h = bbox

        pred_x = x + vx * frames_ahead
        pred_y = y + vy * frames_ahead

        return (pred_x, pred_y, w, h)

    def get_velocity_magnitude(self):
        """Get speed of object"""
        vx, vy = self._get_smoothed_velocity()
        return np.sqrt(vx**2 + vy**2)


# =============================================================================
# Template Matcher
# =============================================================================

class TemplateMatcher:
    """Multi-scale template matching"""

    def __init__(self):
        self.template = None
        self.template_size = None
        self.scales = [0.8, 0.9, 1.0, 1.1, 1.2]

    def set_template(self, gray, bbox):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w > 0 and h > 0:
            self.template = gray[y:y+h, x:x+w].copy()
            self.template_size = (w, h)

    def update_template(self, gray, bbox, alpha=0.1):
        if self.template is None:
            self.set_template(gray, bbox)
            return

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 0 or h <= 0:
            return

        new_patch = gray[y:y+h, x:x+w]
        if new_patch.shape != self.template.shape and self.template_size:
            new_patch = cv2.resize(new_patch, self.template_size)

        if new_patch.shape == self.template.shape:
            self.template = cv2.addWeighted(
                self.template, 1 - alpha, new_patch, alpha, 0
            )

    def match(self, gray, search_region, threshold=0.3):
        """Find template in search region. Returns (x, y, w, h, score) or None"""
        if self.template is None:
            return None

        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)

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
            w = int(tw * best_scale)
            h = int(th * best_scale)
            x = sx + best_loc[0]
            y = sy + best_loc[1]
            return (x, y, w, h, best_val)

        return None


# =============================================================================
# Utility Functions
# =============================================================================

def transform_bbox(bbox, H, frame_shape):
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

    fh, fw = frame_shape[:2]
    x_new = max(0, min(x_new, fw - w_new - 1))
    y_new = max(0, min(y_new, fh - h_new - 1))

    return (int(x_new), int(y_new), int(w_new), int(h_new))


def clamp_bbox(bbox, frame_shape):
    """Clamp bbox to frame bounds"""
    x, y, w, h = bbox
    fh, fw = frame_shape[:2]
    x = max(0, min(int(x), fw - w - 1))
    y = max(0, min(int(y), fh - h - 1))
    return (x, y, int(w), int(h))


# =============================================================================
# Main Tracker
# =============================================================================

class RobustTrackerV5:
    """
    Motion-compensated tracker with object optical flow prediction.

    Key insight: Camera motion compensation tells us where the background moved,
    but the drone can move independently. By tracking the drone's pixels directly
    with dense optical flow, we capture its independent motion.
    """

    def __init__(self, model_dir="./models", motion_threshold=20, force_csrt=False):
        self.model_dir = Path(model_dir)
        self.motion_threshold = motion_threshold
        self.force_csrt = force_csrt

        self.tracker = None
        self.tracker_name = None

        # Motion estimation
        self.camera_motion = CameraMotionEstimator()
        self.object_flow = ObjectFlowTracker()
        self.template_matcher = TemplateMatcher()

        # State
        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None

        # Jitter detection
        self.motion_history = deque(maxlen=5)
        self.jitter_cooldown = 0
        self.last_stable_bbox = None

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize all components
        self.camera_motion.update(gray, self.bbox)
        self.object_flow.update(gray, self.bbox)
        self.template_matcher.set_template(gray, self.bbox)

        self.tracker, self.tracker_name = create_tracker(self.model_dir, self.force_csrt)

        try:
            success = self.tracker.init(frame, self.bbox)
            if not success:
                print("Warning: Tracker init returned False")
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

    def update(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fh, fw = frame.shape[:2]
        self.recovery_method = None

        # 1. Estimate camera motion (background)
        H, cam_motion = self.camera_motion.update(gray, self.bbox)

        # Jitter detection: sudden spike in camera motion
        self.motion_history.append(cam_motion)
        avg_motion = np.mean(list(self.motion_history)[:-1]) if len(self.motion_history) > 1 else cam_motion
        is_jitter = cam_motion > max(40, avg_motion * 3)

        if is_jitter:
            self.jitter_cooldown = 10  # Frames to be extra careful
        elif self.jitter_cooldown > 0:
            self.jitter_cooldown -= 1

        # Save stable position before jitter
        if self.jitter_cooldown == 0 and self.lost_count == 0:
            self.last_stable_bbox = self.bbox

        # 2. Get object's independent velocity from dense flow
        obj_vx, obj_vy = self.object_flow.update(gray, self.bbox)
        obj_speed = self.object_flow.get_velocity_magnitude()

        # 3. Compute camera-compensated prediction
        if H is not None:
            try:
                cam_pred = transform_bbox(self.bbox, H, frame.shape)
            except Exception:
                cam_pred = self.bbox
        else:
            cam_pred = self.bbox

        # 4. Reinitialize tracker on large camera motion
        if cam_motion > self.motion_threshold:
            self.tracker, _ = create_tracker(self.model_dir, self.force_csrt)
            self.tracker.init(frame, cam_pred)

        # 5. Run primary tracker
        try:
            success, box = self.tracker.update(frame)
        except Exception:
            success = False
            box = self.bbox

        if success:
            # Constrain to original size
            x, y, w, h = [int(v) for v in box]
            orig_w, orig_h = self.original_size
            w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
            h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
            cx, cy = x + box[2]/2, y + box[3]/2
            self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

            self.lost_count = 0

            # Update template periodically
            if self.frame_count % 10 == 0:
                self.template_matcher.update_template(gray, self.bbox)

        else:
            self.lost_count += 1

            # RECOVERY: Use object optical flow to predict position
            # This is the key difference from v4 - we use the object's
            # own motion, not just camera motion

            orig_w, orig_h = self.original_size

            # Start from camera-compensated position
            pred_x, pred_y = cam_pred[0], cam_pred[1]
            pred_cx = pred_x + orig_w / 2
            pred_cy = pred_y + orig_h / 2

            # Add object's independent velocity (this is what v4 was missing)
            # Scale by lost_count since the object keeps moving
            pred_cx += obj_vx * self.lost_count
            pred_cy += obj_vy * self.lost_count

            # Create directional search region - elongated in direction of motion
            # This is a cone/ellipse shape along the velocity vector
            base_expand = 2.0 + 0.5 * self.lost_count

            # Direction of motion
            if obj_speed > 1:
                # Normalize velocity
                dir_x = obj_vx / obj_speed
                dir_y = obj_vy / obj_speed

                # Expand more in direction of motion
                expand_forward = base_expand * (1 + obj_speed / 10)
                expand_perpendicular = base_expand

                # Calculate search region bounds
                # Forward direction: extend further
                # Perpendicular: normal expansion
                search_w = orig_w * max(expand_forward * abs(dir_x) + expand_perpendicular * abs(dir_y), base_expand)
                search_h = orig_h * max(expand_forward * abs(dir_y) + expand_perpendicular * abs(dir_x), base_expand)

                # Offset search center in direction of motion
                offset_scale = obj_speed * self.lost_count * 0.3
                search_cx = pred_cx + dir_x * offset_scale
                search_cy = pred_cy + dir_y * offset_scale
            else:
                # No clear direction, use symmetric search
                search_w = orig_w * base_expand
                search_h = orig_h * base_expand
                search_cx = pred_cx
                search_cy = pred_cy

            search_region = (
                int(max(0, search_cx - search_w / 2)),
                int(max(0, search_cy - search_h / 2)),
                int(min(search_w, fw)),
                int(min(search_h, fh))
            )

            # Try template matching in velocity-predicted region
            match_result = self.template_matcher.match(gray, search_region, threshold=0.25)

            # FALLBACK: If no match and we had jitter, search around last stable position
            if match_result is None and self.last_stable_bbox is not None:
                stable_x, stable_y, _, _ = self.last_stable_bbox
                stable_cx = stable_x + orig_w / 2
                stable_cy = stable_y + orig_h / 2

                # Large search around last stable position
                fallback_expand = 4.0 + self.lost_count * 0.5
                fallback_w = orig_w * fallback_expand
                fallback_h = orig_h * fallback_expand

                fallback_region = (
                    int(max(0, stable_cx - fallback_w / 2)),
                    int(max(0, stable_cy - fallback_h / 2)),
                    int(min(fallback_w, fw)),
                    int(min(fallback_h, fh))
                )

                match_result = self.template_matcher.match(gray, fallback_region, threshold=0.2)
                if match_result:
                    self.recovery_method = "fallback"  # Mark that we used fallback

            # LAST RESORT: Search even wider region on left side where drone was heading
            if match_result is None and self.lost_count > 3:
                # Drone was moving left, search left portion of frame
                if obj_vx < 0:
                    wide_region = (0, 0, fw // 2, fh)
                elif obj_vx > 0:
                    wide_region = (fw // 2, 0, fw // 2, fh)
                else:
                    wide_region = (0, 0, fw, fh)

                match_result = self.template_matcher.match(gray, wide_region, threshold=0.2)

            if match_result is not None:
                mx, my, mw, mh, score = match_result
                self.bbox = (mx, my, orig_w, orig_h)
                self.bbox = clamp_bbox(self.bbox, frame.shape)

                self.tracker, _ = create_tracker(self.model_dir, self.force_csrt)
                self.tracker.init(frame, self.bbox)

                # Keep existing recovery_method if set to "fallback", otherwise set template
                if self.recovery_method != "fallback":
                    self.recovery_method = f"template({score:.2f})"
                else:
                    self.recovery_method = f"fallback({score:.2f})"

            else:
                # Fall back to flow-based prediction
                self.bbox = clamp_bbox((pred_cx - orig_w/2, pred_cy - orig_h/2, orig_w, orig_h), frame.shape)

                self.tracker, _ = create_tracker(self.model_dir, self.force_csrt)
                self.tracker.init(frame, self.bbox)

                self.recovery_method = f"flow(v={obj_speed:.1f})"

        return success or (self.recovery_method is not None), self.bbox, cam_motion


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V5")
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

    if not args.no_download:
        tracker_test, name = create_tracker(model_dir)
        if name == "CSRT":
            print("\nNo DL models found. Download ViTTrack? [y/n]: ", end="")
            if input().strip().lower() == 'y':
                download_vittrack(model_dir)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)

    fh, fw = frame.shape[:2]

    tracker = RobustTrackerV5(model_dir, args.motion_threshold, force_csrt=args.no_download)
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"Motion threshold: {args.motion_threshold}px")
    print(f"Object flow tracking: enabled")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v5.mp4")
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

        if tracker.recovery_method:
            color = (0, 255, 255)  # Yellow
        elif success:
            color = (0, 255, 0)    # Green
        else:
            color = (0, 165, 255)  # Orange

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        # Draw velocity vector
        obj_vx, obj_vy = tracker.object_flow._get_smoothed_velocity()
        if abs(obj_vx) > 1 or abs(obj_vy) > 1:
            cx, cy = x + w//2, y + h//2
            end_x = int(cx + obj_vx * 5)
            end_y = int(cy + obj_vy * 5)
            cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 0, 255), 2)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        cv2.putText(frame, f"Frame {frame_num} | {status} | CamMotion: {motion:.1f}px",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Tracker: {tracker.tracker_name} + ObjectFlow",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        vel_conf = tracker.object_flow.confidence
        jitter_str = " JITTER!" if tracker.jitter_cooldown > 0 else ""
        cv2.putText(frame, f"ObjVel: ({obj_vx:.1f}, {obj_vy:.1f}) conf={vel_conf:.2f}{jitter_str}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

        writer.write(frame)
        cv2.imshow("Tracking V5", frame)

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
