"""
Robust Single Object Tracker - Version 17v5 (ViTTrack)

Changes from v17v2:
- ViTTrack (Vision Transformer) instead of CSRT
- Deep learning based tracker - should be more robust
- Expected: ~10-15ms on GPU, ~30-50ms on CPU
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import argparse
import threading
import queue
import time
from collections import defaultdict


class TimingStats:
    """Track timing statistics for each pipeline step"""

    def __init__(self):
        self.times = defaultdict(list)
        self.current_frame_times = {}

    def start(self, name):
        """Start timing a step"""
        self.current_frame_times[name] = time.perf_counter()

    def stop(self, name):
        """Stop timing a step and record"""
        if name in self.current_frame_times:
            elapsed = (time.perf_counter() - self.current_frame_times[name]) * 1000  # ms
            self.times[name].append(elapsed)
            del self.current_frame_times[name]
            return elapsed
        return 0

    def get_stats(self, name):
        """Get min/avg/max/total for a step"""
        if name not in self.times or len(self.times[name]) == 0:
            return 0, 0, 0, 0
        data = self.times[name]
        return min(data), np.mean(data), max(data), sum(data)

    def print_summary(self):
        """Print timing summary"""
        print("\n" + "=" * 70)
        print("TIMING SUMMARY (all times in milliseconds)")
        print("=" * 70)
        print(f"{'Step':<25} {'Min':>8} {'Avg':>8} {'Max':>8} {'Total':>10} {'Count':>7}")
        print("-" * 70)

        total_avg = 0
        for name in sorted(self.times.keys()):
            min_t, avg_t, max_t, total_t = self.get_stats(name)
            count = len(self.times[name])
            print(f"{name:<25} {min_t:>8.2f} {avg_t:>8.2f} {max_t:>8.2f} {total_t:>10.1f} {count:>7}")
            if name != "frame_total":
                total_avg += avg_t

        print("-" * 70)
        frame_min, frame_avg, frame_max, _ = self.get_stats("frame_total")
        if frame_avg > 0:
            fps = 1000.0 / frame_avg
            print(f"{'Frame Total':<25} {frame_min:>8.2f} {frame_avg:>8.2f} {frame_max:>8.2f}")
            print(f"{'Effective FPS':<25} {1000/frame_max:>8.1f} {fps:>8.1f} {1000/frame_min:>8.1f}")
        print("=" * 70)


def create_tracker(model_path="models/vittrack.onnx"):
    """Create ViTTrack tracker"""
    params = cv2.TrackerVit_Params()
    params.net = model_path
    return cv2.TrackerVit_create(params), "ViTTrack"


class CameraMotionEstimator:
    """
    Optimized Motion Estimator using Grid Flow on Downscaled Image.
    Target speed: <15ms
    """
    def __init__(self):
        self.prev_gray_small = None
        self.target_width = 480    # Increased from 320 for better accuracy
        self.grid_step = 25        # Grid spacing in small image pixels
        self.grid_points = None    # Cache for grid
        self.grid_shape = None

        # Balanced LK params
        self.lk_params = dict(
            winSize=(11, 11),      # Reduced from 21x21
            maxLevel=3,            # Increased from 2 for better large motion handling
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
        )

    def _get_grid(self, h, w):
        """Generate or retrieve cached grid points"""
        if self.grid_points is None or self.grid_shape != (h, w):
            # Create meshgrid with margin
            margin = 15
            xs = np.arange(margin, w - margin, self.grid_step)
            ys = np.arange(margin, h - margin, self.grid_step)
            if len(xs) > 0 and len(ys) > 0:
                mesh_x, mesh_y = np.meshgrid(xs, ys)
                self.grid_points = np.float32(np.dstack((mesh_x, mesh_y)).reshape(-1, 1, 2))
                self.grid_shape = (h, w)
            else:
                return None
        return self.grid_points

    def update(self, gray, bbox):
        # 1. Downscale input frame
        fh, fw = gray.shape
        scale = self.target_width / float(fw)
        sh, sw = int(fh * scale), int(fw * scale)
        
        # Use Linear (fast and decent)
        gray_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_LINEAR)

        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            return None, 0

        # 2. Get Grid Points (Eliminates feature detection cost)
        all_pts = self._get_grid(sh, sw)
        if all_pts is None:
            self.prev_gray_small = gray_small
            return None, 0

        # 3. Filter points inside the object bbox (Masking)
        # We must scale the bbox down to match our small image
        bx, by, bw, bh = [v * scale for v in bbox]
        
        # Apply margin (scaled)
        margin = max(10, bw, bh) 
        x1 = bx - margin
        y1 = by - margin
        x2 = bx + bw + margin
        y2 = by + bh + margin
        
        # Vectorized filter: Keep points OUTSIDE the exclusion zone
        # pts is (N, 1, 2) -> x is [:, 0, 0], y is [:, 0, 1]
        px = all_pts[:, 0, 0]
        py = all_pts[:, 0, 1]
        
        # Keep if: x < x1 OR x > x2 OR y < y1 OR y > y2
        mask_keep = (px < x1) | (px > x2) | (py < y1) | (py > y2)
        p0 = all_pts[mask_keep]

        if len(p0) < 8:
            self.prev_gray_small = gray_small
            return None, 0

        # 4. Optical Flow (Lucas Kanade) on small image
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray_small, gray_small, p0, None, **self.lk_params
        )
        
        if p1 is None:
            self.prev_gray_small = gray_small
            return None, 0

        good_p0 = p0[status.flatten() == 1]
        good_p1 = p1[status.flatten() == 1]

        if len(good_p0) < 8:
            self.prev_gray_small = gray_small
            return None, 0

        # 5. RANSAC Homography (on small coords)
        # Threshold 2.0 on small image is roughly 6-10px on full image, robust enough
        H_small, inliers = cv2.findHomography(good_p0, good_p1, cv2.RANSAC, 3.0)

        H_full = None
        motion = 0
        
        if H_small is not None:
            # 6. Scale Homography UP to Full Resolution
            # H_full = S^-1 * H_small * S
            # Translation (0,2 and 1,2) divides by scale (gets bigger)
            # Perspective (2,0 and 2,1) multiplies by scale (gets smaller)
            H_full = H_small.copy()
            H_full[0, 2] /= scale
            H_full[1, 2] /= scale
            H_full[2, 0] *= scale
            H_full[2, 1] *= scale

            # Calculate motion magnitude (scaled back to full pixels)
            if inliers is not None:
                valid = inliers.flatten() == 1
                if np.sum(valid) > 0:
                    diff = good_p1[valid] - good_p0[valid]
                    motion_small = np.median(np.linalg.norm(diff, axis=1))
                    motion = motion_small / scale

        self.prev_gray_small = gray_small
        return H_full, motion


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


class TemplateVerifier:
    def __init__(self):
        self.reference_template = None
        self.original_template = None  # Keep original for fallback
        self.template_size = None
        self.update_alpha = 0.05  # Slow template adaptation

    def set_reference(self, gray, bbox, save_path="debug_template.png"):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w > 0 and h > 0:
            self.reference_template = gray[y:y+h, x:x+w].copy()
            self.original_template = self.reference_template.copy()
            self.template_size = (w, h)
            # Save template for debugging
            if save_path:
                cv2.imwrite(save_path, self.reference_template)
                print(f"Template saved to: {save_path} (size: {w}x{h})")

    def update_template(self, gray, bbox, alpha=None):
        """Slowly adapt template when tracking is confident"""
        if self.reference_template is None:
            return
        if alpha is None:
            alpha = self.update_alpha

        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)

        if w <= 0 or h <= 0:
            return

        current = gray[y:y+h, x:x+w]
        if current.shape != self.reference_template.shape:
            try:
                current = cv2.resize(current, (self.template_size[0], self.template_size[1]))
            except:
                return

        # Blend current appearance into template
        self.reference_template = cv2.addWeighted(
            self.reference_template, 1 - alpha,
            current, alpha, 0
        )

    def verify(self, gray, bbox, threshold=0.3, frame_num=0, save_debug=False):
        if self.reference_template is None:
            return True, 1.0
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 0 or h <= 0:
            return False, 0.0
        current = gray[y:y+h, x:x+w]
        if current.size == 0:
            return False, 0.0
        if current.shape != self.reference_template.shape:
            try:
                current = cv2.resize(current, (self.template_size[0], self.template_size[1]))
            except:
                return False, 0.0
        try:
            result = cv2.matchTemplate(current, self.reference_template, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0] if result.size > 0 else 0.0
        except:
            score = 0.0

        return score >= threshold, score

    def verify_bbox(self, gray, bbox):
        """Verify a bbox and return score (no threshold)"""
        _, score = self.verify(gray, bbox, threshold=0.0)
        return score

    def _search_with_template(self, gray, search_region, template, threshold):
        """Search using a specific template"""
        if template is None:
            return None
        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)
        th, tw = template.shape[:2]
        if sw <= tw or sh <= th:
            return None
        region = gray[sy:sy+sh, sx:sx+sw]
        best_val = 0
        best_loc = None
        best_scale = 1.0
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            new_h, new_w = int(th * scale), int(tw * scale)
            if new_h >= region.shape[0] or new_w >= region.shape[1]:
                continue
            if new_h < 5 or new_w < 5:
                continue
            try:
                scaled = cv2.resize(template, (new_w, new_h))
                result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale
            except:
                continue
        if best_val >= threshold and best_loc is not None:
            x = sx + best_loc[0]
            y = sy + best_loc[1]
            w = int(tw * best_scale)
            h = int(th * best_scale)
            return (x, y, w, h, best_val)
        return None

    def search(self, gray, search_region, threshold=0.25):
        """Search using both adapted and original templates, return best"""
        # Try adapted template
        match1 = self._search_with_template(gray, search_region, self.reference_template, threshold)

        # Try original template
        match2 = self._search_with_template(gray, search_region, self.original_template, threshold)

        # Return best match
        if match1 is None and match2 is None:
            return None
        if match1 is None:
            return match2
        if match2 is None:
            return match1
        return match1 if match1[4] >= match2[4] else match2


class YOLODetector:
    """
    YOLO-based object detector for recovery.
    Runs on cropped regions, not full frame.
    """

    def __init__(self, model_name="yolo11n.pt", conf_threshold=0.3):
        self.model = None
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.last_request_time = 0
        self.cooldown = 0.1  # 100ms between requests
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            print(f"YOLO model loaded: {self.model_name}")
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            self.model = None

    def detect_in_region(self, frame, search_region, target_size):
        """
        Run YOLO on a cropped region and return candidates.
        """
        if self.model is None:
            return []

        # Rate limiting
        now = time.time()
        if now - self.last_request_time < self.cooldown:
            return []
        self.last_request_time = now

        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = frame.shape[:2]
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)

        if sw < 50 or sh < 50:
            return []

        # Crop region
        crop = frame[sy:sy+sh, sx:sx+sw]

        try:
            # Run YOLO with low confidence to get more candidates
            results = self.model(crop, conf=self.conf_threshold, verbose=False)

            candidates = []
            target_w, target_h = target_size
            target_area = target_w * target_h

            # Debug: show what YOLO found
            total_detections = sum(len(r.boxes) if r.boxes is not None else 0 for r in results)
            
            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    # Get bbox in crop coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])

                    w = x2 - x1
                    h = y2 - y1
                    area = w * h

                    # Filter by size (0.3x to 5x target area)
                    if area < target_area * 0.3 or area > target_area * 5:
                        continue

                    # Convert to full frame coordinates
                    fx = sx + x1
                    fy = sy + y1

                    candidates.append((int(fx), int(fy), int(w), int(h), conf))

            return candidates

        except Exception as e:
            print(f"YOLO error: {e}")
            return []


class RobustTrackerV17v5:
    def __init__(self, motion_threshold=20, verify_threshold=0.5, use_yolo=True, model_path="models/vittrack.onnx"):
        self.motion_threshold = motion_threshold
        self.verify_threshold = verify_threshold
        self.use_yolo = use_yolo
        self.model_path = model_path

        self.tracker = None
        self.tracker_name = None
        self.camera_motion = CameraMotionEstimator()
        self.verifier = TemplateVerifier()
        self.yolo = YOLODetector() if use_yolo else None

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None
        self.verify_score = 1.0

        self.last_good_bbox = None
        self.yolo_cooldown = 0  # Frames to wait before next YOLO

        # Score history for drop detection
        self.score_history = []
        self.score_history_len = 5

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])
        self.last_good_bbox = self.bbox

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.camera_motion.update(gray, self.bbox)
        self.verifier.set_reference(gray, self.bbox)

        self.tracker, self.tracker_name = create_tracker(self.model_path)

        try:
            self.tracker.init(frame, self.bbox)
            return True
        except Exception as e:
            print(f"Tracker init failed: {e}")
            return False

    def _try_yolo_recovery(self, frame, gray, search_center):
        """Try to recover using YOLO detection"""
        if self.yolo is None or self.yolo_cooldown > 0:
            return None

        orig_w, orig_h = self.original_size
        fh, fw = frame.shape[:2]

        # Create search region around predicted position - search wide!
        expand = 10.0 + self.lost_count * 1.0
        search_w = orig_w * expand
        search_h = orig_h * expand

        search_region = (
            int(max(0, search_center[0] - search_w / 2)),
            int(max(0, search_center[1] - search_h / 2)),
            int(min(search_w, fw)),
            int(min(search_h, fh))
        )

        # Run YOLO
        candidates = self.yolo.detect_in_region(frame, search_region, self.original_size)

        if not candidates:
            self.yolo_cooldown = 2  # Wait 2 frames before trying again
            return None

        # Verify each candidate against template, pick best
        best_match = None
        best_score = 0

        for cx, cy, cw, ch, conf in candidates:
            # Create bbox with original size centered on detection
            det_cx = cx + cw / 2
            det_cy = cy + ch / 2
            test_bbox = (int(det_cx - orig_w/2), int(det_cy - orig_h/2), orig_w, orig_h)
            test_bbox = clamp_bbox(test_bbox, frame.shape)

            score = self.verifier.verify_bbox(gray, test_bbox)

            # Combine YOLO confidence and template score
            combined_score = 0.3 * conf + 0.7 * score

            if combined_score > best_score:
                best_score = combined_score
                best_match = (test_bbox, score, conf)

        if best_match and best_match[1] > 0.15:  # Template score threshold
            self.yolo_cooldown = 1  # Short cooldown on success
            return best_match

        self.yolo_cooldown = 2  # Short cooldown - keep trying
        return None

    def update(self, frame, timing=None):
        self.frame_count += 1

        # BGR to Gray
        if timing:
            timing.start("cvt_gray")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if timing:
            timing.stop("cvt_gray")

        fh, fw = frame.shape[:2]
        self.recovery_method = None

        if self.yolo_cooldown > 0:
            self.yolo_cooldown -= 1

        # Camera motion estimation
        if timing:
            timing.start("motion_est")
        H, cam_motion = self.camera_motion.update(gray, self.bbox)
        if timing:
            timing.stop("motion_est")

        # Motion-compensated prediction
        if timing:
            timing.start("motion_comp")
        if H is not None:
            try:
                motion_pred = transform_bbox(self.bbox, H, frame.shape)
            except:
                motion_pred = self.bbox
        else:
            motion_pred = self.bbox
        if timing:
            timing.stop("motion_comp")

        # Reinitialize on high camera motion
        if timing:
            timing.start("tracker_reinit")
        if cam_motion > self.motion_threshold:
            self.tracker, _ = create_tracker(self.model_path)
            self.tracker.init(frame, motion_pred)
        if timing:
            timing.stop("tracker_reinit")

        # Run primary tracker (ViTTrack)
        if timing:
            timing.start("vit_update")
        try:
            tracker_success, box = self.tracker.update(frame)
        except:
            tracker_success = False
            box = self.bbox
        if timing:
            timing.stop("vit_update")

        # Verify
        actual_success = False
        if tracker_success:
            temp_bbox = tuple(int(v) for v in box)

            if timing:
                timing.start("verify")
            is_valid, self.verify_score = self.verifier.verify(
                gray, temp_bbox, self.verify_threshold,
                frame_num=self.frame_count, save_debug=False  # Disable debug saving for timing
            )
            if timing:
                timing.stop("verify")

            # Also check for sudden score drop (indicates drift to wrong target)
            if len(self.score_history) >= 3:
                recent_avg = np.mean(self.score_history[-3:])
                # If score drops more than 0.25 from recent average, reject
                if self.verify_score < recent_avg - 0.25:
                    is_valid = False

            if is_valid:
                x, y, w, h = temp_bbox
                orig_w, orig_h = self.original_size
                w = max(int(orig_w * 0.7), min(int(orig_w * 1.3), w))
                h = max(int(orig_h * 0.7), min(int(orig_h * 1.3), h))
                cx, cy = x + box[2]/2, y + box[3]/2
                self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

                self.lost_count = 0
                self.last_good_bbox = self.bbox
                actual_success = True

                # Adapt template when tracking is very confident
                if self.verify_score > 0.65:
                    if timing:
                        timing.start("template_update")
                    self.verifier.update_template(gray, self.bbox, alpha=0.03)
                    if timing:
                        timing.stop("template_update")

                # Track score history for good frames only
                self.score_history.append(self.verify_score)
                if len(self.score_history) > self.score_history_len:
                    self.score_history.pop(0)
        else:
            self.verify_score = 0.0

        # Recovery
        if not actual_success:
            self.lost_count += 1
            orig_w, orig_h = self.original_size

            if timing:
                timing.start("recovery")

            # IMMEDIATE full-frame search with HIGH threshold
            # This avoids false matches on clouds/bushes
            match = self.verifier.search(gray, (0, 0, fw, fh), threshold=0.55)

            if match:
                mx, my, mw, mh, score = match
                self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                self.verify_score = score

                self.tracker, _ = create_tracker(self.model_path)
                self.tracker.init(frame, self.bbox)

                self.recovery_method = f"fullsearch({score:.2f})"
                self.last_good_bbox = self.bbox
                self.score_history = [score]
                actual_success = True

            # If high-threshold search fails, try lower threshold but verify result
            if not actual_success:
                match = self.verifier.search(gray, (0, 0, fw, fh), threshold=0.35)

                if match:
                    mx, my, mw, mh, score = match
                    # Only accept if score is reasonably good
                    if score > 0.45:
                        self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                        self.verify_score = score

                        self.tracker, _ = create_tracker(self.model_path)
                        self.tracker.init(frame, self.bbox)

                        self.recovery_method = f"search({score:.2f})"
                        self.last_good_bbox = self.bbox
                        self.score_history = [score]
                        actual_success = True

            # Hold position as last resort
            if not actual_success:
                self.bbox = self.last_good_bbox if self.last_good_bbox else motion_pred
                self.tracker, _ = create_tracker(self.model_path)
                self.tracker.init(frame, self.bbox)
                self.recovery_method = "hold"

            if timing:
                timing.stop("recovery")

        return actual_success, self.bbox, cam_motion, self.verify_score


def main():
    parser = argparse.ArgumentParser(description="Robust Object Tracker V17v5 (ViTTrack)")
    parser.add_argument("video", help="Video file path")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--no-yolo", action="store_true", help="Disable YOLO recovery")
    parser.add_argument("--verify-threshold", type=float, default=0.25)
    parser.add_argument("--model", default="models/vittrack.onnx", help="ViTTrack model path")
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

    tracker = RobustTrackerV17v5(
        verify_threshold=args.verify_threshold,
        use_yolo=not args.no_yolo,
        model_path=args.model
    )
    if not tracker.initialize(frame, bbox):
        print("Error: Failed to initialize tracker")
        sys.exit(1)

    print(f"\nTracker: {tracker.tracker_name}")
    print(f"YOLO recovery: {'enabled' if tracker.use_yolo else 'disabled'}")
    print("\nPress 'q' to quit, SPACE to pause\n")

    out_path = args.output or (Path(args.video).stem + "_tracked_v17v5.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    writer = cv2.VideoWriter(out_path, fourcc, fps, (fw, fh))

    frame_num = 0
    success_count = 0
    yolo_recoveries = 0

    # Timing stats
    timing = TimingStats()

    print("\nRunning with timing enabled...\n")

    while True:
        timing.start("frame_total")

        # Frame read
        timing.start("frame_read")
        ret, frame = cap.read()
        timing.stop("frame_read")

        if not ret:
            break

        frame_num += 1

        # Tracker update (includes multiple sub-timings)
        success, bbox, cam_motion, verify_score = tracker.update(frame, timing)

        if success:
            success_count += 1
        if tracker.recovery_method and "YOLO" in tracker.recovery_method:
            yolo_recoveries += 1

        # Drawing
        timing.start("drawing")
        x, y, w, h = bbox

        if tracker.recovery_method and "YOLO" in tracker.recovery_method:
            color = (255, 0, 255)  # Magenta for YOLO recovery
        elif tracker.recovery_method:
            color = (0, 255, 255)  # Yellow for template recovery
        elif verify_score > 0.4:
            color = (0, 255, 0)
        elif verify_score > 0.25:
            color = (0, 200, 200)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        status = "TRACKING" if success and not tracker.recovery_method else f"LOST({tracker.lost_count})"
        if tracker.recovery_method:
            status = f"RECOVERED: {tracker.recovery_method}"

        # Show FPS on frame
        frame_min, frame_avg, frame_max, _ = timing.get_stats("frame_total")
        current_fps = 1000.0 / frame_avg if frame_avg > 0 else 0

        cv2.putText(frame, f"Frame {frame_num} | {status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Motion: {cam_motion:.1f} | Verify: {verify_score:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {current_fps:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        timing.stop("drawing")

        # Video write
        timing.start("video_write")
        writer.write(frame)
        timing.stop("video_write")

        # Display
        timing.start("imshow")
        cv2.imshow("Tracking V17v5", frame)
        timing.stop("imshow")

        timing.stop("frame_total")

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
    print(f"YOLO recoveries: {yolo_recoveries}")
    print(f"Output: {out_path}")

    # Print timing summary
    timing.print_summary()


if __name__ == "__main__":
    main()
