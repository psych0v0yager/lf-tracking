"""
Robust Single Object Tracker - ONNX Accelerated Version

Neural tracker using MixFormerV2-S with ONNX Runtime acceleration.
Based on tracker_mixformer.py but uses ONNX for ~2x faster neural inference.

Expected Performance:
- PyTorch version: 83.5 FPS (7.4ms neural inference)
- ONNX version: ~150 FPS (~2ms neural inference)

Key Components:
- MixFormerV2-S ONNX: Transformer tracker with ONNX Runtime CUDA
- Camera motion compensation: Handles large camera pans
- Template verification: Fallback for drift detection
- Light pyramidal recovery: Fast re-detection

Usage:
    uv run python tracker_mixformer_onnx.py test_track_dron1.mp4 1314 623 73 46
    uv run python tracker_mixformer_onnx.py test_track_dron1.mp4 1314 623 73 46 --no-save --no-show
"""

import cv2
import numpy as np
import sys
import os
from pathlib import Path
import argparse
import time
from collections import defaultdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from trackers.mixformer_onnx import MixFormerONNXTracker


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


class CameraMotionEstimator:
    """
    Optimized Motion Estimator using Grid Flow on Downscaled Image.
    """
    def __init__(self):
        self.prev_gray_small = None
        self.target_width = 480
        self.grid_step = 25
        self.grid_points = None
        self.grid_shape = None

        self.lk_params = dict(
            winSize=(11, 11),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
        )

    def _get_grid(self, h, w):
        """Generate or retrieve cached grid points"""
        if self.grid_points is None or self.grid_shape != (h, w):
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
        fh, fw = gray.shape
        scale = self.target_width / float(fw)
        sh, sw = int(fh * scale), int(fw * scale)

        gray_small = cv2.resize(gray, (sw, sh), interpolation=cv2.INTER_LINEAR)

        if self.prev_gray_small is None:
            self.prev_gray_small = gray_small
            return None, 0

        all_pts = self._get_grid(sh, sw)
        if all_pts is None:
            self.prev_gray_small = gray_small
            return None, 0

        # Filter points inside the object bbox
        bx, by, bw, bh = [v * scale for v in bbox]
        margin = max(10, bw, bh)
        x1, y1 = bx - margin, by - margin
        x2, y2 = bx + bw + margin, by + bh + margin

        px = all_pts[:, 0, 0]
        py = all_pts[:, 0, 1]
        mask_keep = (px < x1) | (px > x2) | (py < y1) | (py > y2)
        p0 = all_pts[mask_keep]

        if len(p0) < 8:
            self.prev_gray_small = gray_small
            return None, 0

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

        H_small, inliers = cv2.findHomography(good_p0, good_p1, cv2.RANSAC, 3.0)

        H_full = None
        motion = 0

        if H_small is not None:
            H_full = H_small.copy()
            H_full[0, 2] /= scale
            H_full[1, 2] /= scale
            H_full[2, 0] *= scale
            H_full[2, 1] *= scale

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
    """Template-based verification and recovery with edge-based matching"""
    def __init__(self):
        self.reference_template = None
        self.original_template = None
        self.reference_edge = None
        self.original_edge = None
        self.template_size = None
        self.update_alpha = 0.05

    def _compute_edges(self, img):
        """Compute edge map using Canny with adaptive thresholds"""
        if img is None or img.size == 0:
            return None
        median_val = np.median(img)
        lower = int(max(0, 0.5 * median_val))
        upper = int(min(255, 1.5 * median_val))
        if upper - lower < 30:
            lower = max(0, median_val - 25)
            upper = min(255, median_val + 25)
        edges = cv2.Canny(img, lower, upper)
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        return edges

    def set_reference(self, gray, bbox, save_path=None):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w > 0 and h > 0:
            self.reference_template = gray[y:y+h, x:x+w].copy()
            self.original_template = self.reference_template.copy()
            self.template_size = (w, h)
            self.reference_edge = self._compute_edges(self.reference_template)
            self.original_edge = self._compute_edges(self.original_template)

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

        self.reference_template = cv2.addWeighted(
            self.reference_template, 1 - alpha,
            current, alpha, 0
        )
        self.reference_edge = self._compute_edges(self.reference_template)

    def _verify_edge(self, current):
        """Compute edge-based matching score"""
        if self.reference_edge is None:
            return 0.0
        try:
            current_edge = self._compute_edges(current)
            if current_edge is None:
                return 0.0
            if current_edge.shape != self.reference_edge.shape:
                current_edge = cv2.resize(current_edge, (self.template_size[0], self.template_size[1]))
            result = cv2.matchTemplate(current_edge, self.reference_edge, cv2.TM_CCOEFF_NORMED)
            return result[0, 0] if result.size > 0 else 0.0
        except:
            return 0.0

    def verify(self, gray, bbox, threshold=0.3):
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
            gray_score = result[0, 0] if result.size > 0 else 0.0
        except:
            gray_score = 0.0

        edge_score = self._verify_edge(current)

        if gray_score >= 0.5:
            combined_score = gray_score
        elif gray_score >= 0.15 and edge_score >= 0.45:
            combined_score = gray_score + (edge_score - 0.4) * 0.4
        else:
            combined_score = gray_score

        return combined_score >= threshold, combined_score

    def _search_with_template_fast(self, gray, search_region, template, threshold, scales=[0.9, 1.0, 1.1]):
        """Fast search using a specific template"""
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
        for scale in scales:
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

    def search_light(self, gray, search_region, threshold=0.25):
        """Light search: coarse only at 1/3 res, no refinement (~3x faster)"""
        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)

        if self.reference_template is None:
            return None

        down_scale = 0.33
        gray_small = cv2.resize(gray, None, fx=down_scale, fy=down_scale, interpolation=cv2.INTER_AREA)
        template_small = cv2.resize(self.reference_template, None, fx=down_scale, fy=down_scale, interpolation=cv2.INTER_AREA)

        search_small = (int(sx * down_scale), int(sy * down_scale),
                       int(sw * down_scale), int(sh * down_scale))

        match = self._search_with_template_fast(gray_small, search_small, template_small,
                                                threshold * 0.85, scales=[1.0])

        if match is None:
            orig_small = cv2.resize(self.original_template, None, fx=down_scale, fy=down_scale, interpolation=cv2.INTER_AREA)
            match = self._search_with_template_fast(gray_small, search_small, orig_small,
                                                    threshold * 0.85, scales=[1.0])

        if match is None:
            return None

        cx, cy, cw, ch, score = match
        th, tw = self.reference_template.shape[:2]
        return (int(cx / down_scale), int(cy / down_scale), tw, th, score)

    def search(self, gray, search_region, threshold=0.25):
        """Search using light approach (faster, slightly less accurate)"""
        return self.search_light(gray, search_region, threshold)


class RobustTrackerONNX:
    """
    MixFormerV2-S ONNX tracker with motion compensation.

    Uses ONNX Runtime for accelerated neural inference (~2ms vs ~7.4ms PyTorch).
    Otherwise identical to the PyTorch version in tracker_mixformer.py.
    """
    def __init__(self, motion_threshold=20, verify_threshold=0.5, score_threshold=0.5):
        self.motion_threshold = motion_threshold
        self.verify_threshold = verify_threshold
        self.score_threshold = score_threshold

        self.tracker = None
        self.camera_motion = CameraMotionEstimator()
        self.verifier = TemplateVerifier()

        self.bbox = None
        self.original_size = None
        self.lost_count = 0
        self.frame_count = 0
        self.recovery_method = None
        self.verify_score = 1.0
        self.mixformer_score = 1.0

        self.last_good_bbox = None
        self.motion_pred_bbox = None
        self.score_history = []
        self.score_history_len = 5
        self.last_cam_motion = 0

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])
        self.last_good_bbox = self.bbox

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.camera_motion.update(gray, self.bbox)
        self.verifier.set_reference(gray, self.bbox)

        # Create ONNX tracker with reduced search factor to prevent drift
        print("[ONNX] Initializing MixFormerV2-S ONNX tracker...")
        self.tracker = MixFormerONNXTracker(search_factor=3.0)  # Default 4.5, reduced to prevent drift

        try:
            success = self.tracker.init(frame, list(self.bbox))
            if success:
                print(f"[ONNX] Tracker initialized at {self.bbox}")
            return success
        except Exception as e:
            print(f"[ONNX] Tracker init failed: {e}")
            return False

    def update(self, frame, timing=None):
        self.frame_count += 1

        if timing:
            timing.start("cvt_gray")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if timing:
            timing.stop("cvt_gray")

        fh, fw = frame.shape[:2]
        self.recovery_method = None

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

        # For MixFormer, we DON'T reinitialize on camera motion
        if timing:
            timing.start("tracker_reinit")
        self.last_cam_motion = cam_motion
        self.motion_pred_bbox = motion_pred
        if timing:
            timing.stop("tracker_reinit")

        # Run ONNX tracker
        if timing:
            timing.start("onnx_update")
        try:
            tracker_success, box, self.mixformer_score = self.tracker.update(frame)
        except Exception as e:
            print(f"[ONNX] Update error: {e}")
            tracker_success = False
            box = self.bbox
            self.mixformer_score = 0.0
        if timing:
            timing.stop("onnx_update")

        actual_success = False

        if tracker_success and self.mixformer_score >= self.score_threshold:
            temp_bbox = tuple(int(v) for v in box)

            # Verify with template matching as backup check
            if timing:
                timing.start("verify")
            is_valid, self.verify_score = self.verifier.verify(
                gray, temp_bbox, self.verify_threshold
            )
            if timing:
                timing.stop("verify")

            # Check for sudden score drop
            if len(self.score_history) >= 3:
                recent_avg = np.mean(self.score_history[-3:])
                if self.verify_score < recent_avg - 0.25:
                    is_valid = False

            if is_valid:
                x, y, w, h = temp_bbox
                orig_w, orig_h = self.original_size
                # Stricter size constraints - tighter when confidence is low
                if self.mixformer_score >= 0.7:
                    min_scale, max_scale = 0.8, 1.2
                else:
                    min_scale, max_scale = 0.9, 1.1
                w = max(int(orig_w * min_scale), min(int(orig_w * max_scale), w))
                h = max(int(orig_h * min_scale), min(int(orig_h * max_scale), h))
                cx, cy = x + box[2]/2, y + box[3]/2
                self.bbox = clamp_bbox((cx - w/2, cy - h/2, w, h), frame.shape)

                self.lost_count = 0
                self.last_good_bbox = self.bbox
                actual_success = True

                # Adapt template when both MixFormer and template verify are confident
                if self.verify_score > 0.65 and self.mixformer_score > 0.7:
                    if timing:
                        timing.start("template_update")
                    self.verifier.update_template(gray, self.bbox, alpha=0.03)
                    if timing:
                        timing.stop("template_update")

                self.score_history.append(self.verify_score)
                if len(self.score_history) > self.score_history_len:
                    self.score_history.pop(0)
        else:
            self.verify_score = 0.0

        # Recovery if tracking failed
        if not actual_success:
            self.lost_count += 1
            orig_w, orig_h = self.original_size

            if timing:
                timing.start("recovery")

            # Calculate distance constraint from last good position
            last_cx = self.last_good_bbox[0] + self.last_good_bbox[2] / 2
            last_cy = self.last_good_bbox[1] + self.last_good_bbox[3] / 2
            max_dist = 150 + self.last_cam_motion * 5

            def is_valid_match(mx, my, mw, mh):
                """Check if match is within acceptable distance"""
                match_cx = mx + mw / 2
                match_cy = my + mh / 2
                dist = np.sqrt((match_cx - last_cx)**2 + (match_cy - last_cy)**2)
                return dist <= max_dist, dist

            # Full-frame search with high threshold
            match = self.verifier.search(gray, (0, 0, fw, fh), threshold=0.55)

            if match:
                mx, my, mw, mh, score = match
                valid, dist = is_valid_match(mx, my, mw, mh)
                if valid:
                    self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                    self.verify_score = score
                    self.tracker.init(frame, list(self.bbox))
                    self.recovery_method = f"fullsearch({score:.2f},d={dist:.0f})"
                    self.last_good_bbox = self.bbox
                    self.score_history = [score]
                    actual_success = True

            # Try lower threshold if high fails
            if not actual_success:
                match = self.verifier.search(gray, (0, 0, fw, fh), threshold=0.35)
                if match:
                    mx, my, mw, mh, score = match
                    valid, dist = is_valid_match(mx, my, mw, mh)
                    if valid:
                        self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                        self.verify_score = score
                        self.tracker.init(frame, list(self.bbox))
                        self.recovery_method = f"lowthresh({score:.2f},d={dist:.0f})"
                        self.last_good_bbox = self.bbox
                        self.score_history = [score]
                        actual_success = True

            if timing:
                timing.stop("recovery")

            # Fallback to last good bbox
            if not actual_success and self.last_good_bbox is not None:
                self.bbox = self.last_good_bbox

        return actual_success, self.bbox, self.verify_score, self.recovery_method


def run_tracker(video_path, initial_bbox, save_video=True, show_video=True):
    """Run the MixFormerV2 ONNX tracker on a video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {video_path}")
    print(f"Resolution: {width}x{height}, FPS: {fps:.1f}, Frames: {total_frames}")

    out = None
    if save_video:
        output_path = str(Path(video_path).stem) + "_onnx_tracked.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Output: {output_path}")

    tracker = RobustTrackerONNX(motion_threshold=20, verify_threshold=0.50, score_threshold=0.5)
    timing = TimingStats()

    frame_count = 0
    success_count = 0
    recovery_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timing.start("frame_total")

        if frame_count == 0:
            tracker.initialize(frame, initial_bbox)
            bbox = initial_bbox
            success = True
            recovery = None
        else:
            success, bbox, score, recovery = tracker.update(frame, timing)

        timing.stop("frame_total")

        if success:
            success_count += 1
        if recovery:
            recovery_count += 1
            print(f"Frame {frame_count}: Recovery - {recovery}")

        # Draw results
        x, y, w, h = [int(v) for v in bbox]
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Info text
        info = f"Frame {frame_count}/{total_frames} | MF:{tracker.mixformer_score:.2f} TV:{tracker.verify_score:.2f}"
        if recovery:
            info += f" | {recovery}"
        cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if save_video and out is not None:
            out.write(frame)

        if show_video:
            cv2.imshow("MixFormerV2 ONNX Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

        frame_count += 1

    cap.release()
    if out is not None:
        out.release()
    if show_video:
        cv2.destroyAllWindows()

    print(f"\n{'='*50}")
    print(f"Results: {success_count}/{frame_count} frames tracked ({100*success_count/max(1,frame_count):.1f}%)")
    print(f"Recovery events: {recovery_count}")
    timing.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MixFormerV2 ONNX Accelerated Tracker")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--no-save", action="store_true", help="Don't save output video")
    parser.add_argument("--no-show", action="store_true", help="Don't show video window")

    args = parser.parse_args()
    initial_bbox = (args.x, args.y, args.w, args.h)

    run_tracker(
        args.video,
        initial_bbox,
        save_video=not args.no_save,
        show_video=not args.no_show
    )
