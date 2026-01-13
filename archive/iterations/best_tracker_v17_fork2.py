"""
Robust Single Object Tracker - Version 16v9 (Full Optimization)

Changes from v16v8:
1. Centralized BGR->Gray conversion (done once per frame).
2. TemplateVerifier: Added "Early Exit" (stops if score > 0.8 at scale 1.0).
3. TemplateVerifier: Added "Downscale Verification" (runs matchTemplate at 0.5x scale).
4. YOLODetector: Added 'half=True' for GPU speedup (if available).
"""

import cv2
import numpy as np
import sys
from pathlib import Path
import argparse
import time
from collections import defaultdict


class TimingStats:
    """Track timing statistics for each pipeline step"""
    def __init__(self):
        self.times = defaultdict(list)
        self.current_frame_times = {}

    def start(self, name):
        self.current_frame_times[name] = time.perf_counter()

    def stop(self, name):
        if name in self.current_frame_times:
            elapsed = (time.perf_counter() - self.current_frame_times[name]) * 1000
            self.times[name].append(elapsed)
            del self.current_frame_times[name]
            return elapsed
        return 0

    def get_stats(self, name):
        if name not in self.times or len(self.times[name]) == 0:
            return 0, 0, 0, 0
        data = self.times[name]
        return min(data), np.mean(data), max(data), sum(data)

    def print_summary(self):
        print("\n" + "=" * 70)
        print("TIMING SUMMARY (all times in milliseconds)")
        print("=" * 70)
        print(f"{'Step':<25} {'Min':>8} {'Avg':>8} {'Max':>8} {'Total':>10} {'Count':>7}")
        print("-" * 70)
        total_avg = 0
        for name in sorted(self.times.keys()):
            min_t, avg_t, max_t, total_t = self.get_stats(name)
            print(f"{name:<25} {min_t:>8.2f} {avg_t:>8.2f} {max_t:>8.2f} {total_t:>10.1f} {len(self.times[name]):>7}")
            if name != "frame_total": total_avg += avg_t
        print("-" * 70)
        _, frame_avg, _, _ = self.get_stats("frame_total")
        if frame_avg > 0:
            print(f"{'FPS':<25} {1000/frame_avg:>8.1f}")
        print("=" * 70)


def create_tracker():
    return cv2.TrackerCSRT_create(), "CSRT"


class CameraMotionEstimator:
    def __init__(self):
        self.prev_gray_small = None
        self.target_width = 320
        self.grid_step = 25
        self.grid_points = None
        self.grid_shape = None
        self.lk_params = dict(
            winSize=(11, 11),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03)
        )

    def _get_grid(self, h, w):
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

        bx, by, bw, bh = [v * scale for v in bbox]
        margin = max(10, bw, bh)
        x1, y1, x2, y2 = bx - margin, by - margin, bx + bw + margin, by + bh + margin
        
        px = all_pts[:, 0, 0]
        py = all_pts[:, 0, 1]
        mask_keep = (px < x1) | (px > x2) | (py < y1) | (py > y2)
        p0 = all_pts[mask_keep]

        if len(p0) < 8:
            self.prev_gray_small = gray_small
            return None, 0

        p1, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray_small, gray_small, p0, None, **self.lk_params)
        
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
                    motion = np.median(np.linalg.norm(diff, axis=1)) / scale

        self.prev_gray_small = gray_small
        return H_full, motion


def transform_bbox(bbox, H, frame_shape):
    x, y, w, h = bbox
    corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32).reshape(-1, 1, 2)
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
    """
    Optimized Verifier:
    1. Supports downscaling for faster matching.
    2. Implements Early Exit if scale 1.0 is good enough.
    """
    def __init__(self):
        self.reference_template = None
        self.original_template = None
        self.template_size = None
        self.update_alpha = 0.05
        # Verification runs at 0.5 scale for speed
        self.match_scale = 0.5 

    def set_reference(self, gray, bbox):
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w > 0 and h > 0:
            self.reference_template = gray[y:y+h, x:x+w].copy()
            self.original_template = self.reference_template.copy()
            self.template_size = (w, h)

    def update_template(self, gray, bbox, alpha=None):
        if self.reference_template is None: return
        if alpha is None: alpha = self.update_alpha
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 0 or h <= 0: return

        current = gray[y:y+h, x:x+w]
        if current.shape != self.reference_template.shape:
            try:
                current = cv2.resize(current, (self.template_size[0], self.template_size[1]))
            except: return

        self.reference_template = cv2.addWeighted(
            self.reference_template, 1 - alpha, current, alpha, 0
        )

    def verify(self, gray, bbox, threshold=0.3):
        """
        Verify with optimization:
        - Downscale image and template by 50%
        """
        if self.reference_template is None: return True, 1.0
        x, y, w, h = [int(v) for v in bbox]
        fh, fw = gray.shape
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w <= 0 or h <= 0: return False, 0.0

        current = gray[y:y+h, x:x+w]
        if current.size == 0: return False, 0.0

        # Resize current crop to template size first
        if current.shape != self.reference_template.shape:
            try:
                current = cv2.resize(current, (self.template_size[0], self.template_size[1]))
            except: return False, 0.0

        # OPTIMIZATION: Downscale for matching
        if self.match_scale < 1.0:
            curr_small = cv2.resize(current, None, fx=self.match_scale, fy=self.match_scale)
            ref_small = cv2.resize(self.reference_template, None, fx=self.match_scale, fy=self.match_scale)
        else:
            curr_small = current
            ref_small = self.reference_template

        try:
            result = cv2.matchTemplate(curr_small, ref_small, cv2.TM_CCOEFF_NORMED)
            score = result[0, 0] if result.size > 0 else 0.0
        except:
            score = 0.0

        return score >= threshold, score

    def verify_bbox(self, gray, bbox):
        _, score = self.verify(gray, bbox, threshold=0.0)
        return score

    def _search_with_template(self, gray, search_region, template, threshold):
        if template is None: return None
        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = gray.shape
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)
        th, tw = template.shape[:2]
        if sw <= tw or sh <= th: return None
        
        region = gray[sy:sy+sh, sx:sx+sw]
        
        # 1. Check at Scale 1.0 first (Early Exit)
        try:
            res = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            
            # EARLY EXIT: If match is excellent, don't search other scales
            if max_val > 0.8:
                x = sx + max_loc[0]
                y = sy + max_loc[1]
                return (x, y, tw, th, max_val)
        except: pass

        # 2. Multi-scale search
        best_val = 0
        best_loc = None
        best_scale = 1.0
        
        # Reduced scales list (removed 0.8/1.2 to save time unless necessary)
        scales = [0.9, 1.0, 1.1] 
        
        for scale in scales:
            new_h, new_w = int(th * scale), int(tw * scale)
            if new_h >= region.shape[0] or new_w >= region.shape[1]: continue
            if new_h < 5 or new_w < 5: continue
            
            try:
                scaled = cv2.resize(template, (new_w, new_h))
                result = cv2.matchTemplate(region, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_scale = scale
            except: continue

        if best_val >= threshold and best_loc is not None:
            x = sx + best_loc[0]
            y = sy + best_loc[1]
            w = int(tw * best_scale)
            h = int(th * best_scale)
            return (x, y, w, h, best_val)
        return None

    def search(self, gray, search_region, threshold=0.25):
        # Try adapted first
        match1 = self._search_with_template(gray, search_region, self.reference_template, threshold)
        if match1 and match1[4] > 0.7: return match1  # Early exit if good match

        match2 = self._search_with_template(gray, search_region, self.original_template, threshold)
        
        if match1 is None: return match2
        if match2 is None: return match1
        return match1 if match1[4] >= match2[4] else match2


class YOLODetector:
    def __init__(self, model_name="yolo11n.pt", conf_threshold=0.3):
        self.model = None
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.last_request_time = 0
        self.cooldown = 0.1
        self._load_model()

    def _load_model(self):
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_name)
            # Optimize for inference
            # self.model.fuse() # Optional, sometimes helps
            print(f"YOLO model loaded: {self.model_name}")
        except Exception as e:
            print(f"Failed to load YOLO: {e}")
            self.model = None

    def detect_in_region(self, frame, search_region, target_size):
        if self.model is None: return []

        now = time.time()
        if now - self.last_request_time < self.cooldown: return []
        self.last_request_time = now

        sx, sy, sw, sh = [int(v) for v in search_region]
        fh, fw = frame.shape[:2]
        sx, sy = max(0, sx), max(0, sy)
        sw, sh = min(sw, fw - sx), min(sh, fh - sy)
        if sw < 50 or sh < 50: return []

        crop = frame[sy:sy+sh, sx:sx+sw]

        try:
            # Run YOLO (half=True for GPU speedup)
            results = self.model(crop, conf=self.conf_threshold, verbose=False, half=False) # half=True if GPU

            candidates = []
            target_w, target_h = target_size
            target_area = target_w * target_h

            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    if area < target_area * 0.3 or area > target_area * 5: continue
                    fx = sx + x1
                    fy = sy + y1
                    candidates.append((int(fx), int(fy), int(w), int(h), conf))
            return candidates
        except Exception as e:
            print(f"YOLO error: {e}")
            return []


class RobustTrackerV16:
    def __init__(self, motion_threshold=20, verify_threshold=0.5, use_yolo=True):
        self.motion_threshold = motion_threshold
        self.verify_threshold = verify_threshold
        self.use_yolo = use_yolo
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
        self.yolo_cooldown = 0
        self.score_history = []
        self.score_history_len = 5

    def initialize(self, frame, bbox):
        self.bbox = tuple(int(v) for v in bbox)
        self.original_size = (self.bbox[2], self.bbox[3])
        self.last_good_bbox = self.bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.camera_motion.update(gray, self.bbox)
        self.verifier.set_reference(gray, self.bbox)
        self.tracker, self.tracker_name = create_tracker()
        try:
            self.tracker.init(frame, self.bbox)
            return True
        except: return False

    def update(self, frame, timing=None):
        self.frame_count += 1
        
        # Centralized Gray Conversion
        if timing: timing.start("cvt_gray")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if timing: timing.stop("cvt_gray")

        fh, fw = frame.shape[:2]
        self.recovery_method = None
        if self.yolo_cooldown > 0: self.yolo_cooldown -= 1

        # Motion Est
        if timing: timing.start("motion_est")
        H, cam_motion = self.camera_motion.update(gray, self.bbox)
        if timing: timing.stop("motion_est")

        # Motion Comp
        if timing: timing.start("motion_comp")
        motion_pred = transform_bbox(self.bbox, H, frame.shape) if H is not None else self.bbox
        if timing: timing.stop("motion_comp")

        # Tracker Reinit
        if cam_motion > self.motion_threshold:
            self.tracker, _ = create_tracker()
            self.tracker.init(frame, motion_pred)

        # CSRT Update
        if timing: timing.start("csrt_update")
        try:
            tracker_success, box = self.tracker.update(frame)
        except:
            tracker_success = False
            box = self.bbox
        if timing: timing.stop("csrt_update")

        actual_success = False
        if tracker_success:
            temp_bbox = tuple(int(v) for v in box)
            if timing: timing.start("verify")
            is_valid, self.verify_score = self.verifier.verify(gray, temp_bbox, self.verify_threshold)
            if timing: timing.stop("verify")

            if len(self.score_history) >= 3:
                recent_avg = np.mean(self.score_history[-3:])
                if self.verify_score < recent_avg - 0.25: is_valid = False

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

                if self.verify_score > 0.65:
                    self.verifier.update_template(gray, self.bbox, alpha=0.03)

                self.score_history.append(self.verify_score)
                if len(self.score_history) > self.score_history_len: self.score_history.pop(0)
        else:
            self.verify_score = 0.0

        if not actual_success:
            self.lost_count += 1
            orig_w, orig_h = self.original_size
            if timing: timing.start("recovery")
            
            # Optimized Search: Try adapted first
            match = self.verifier.search(gray, (0, 0, fw, fh), threshold=0.55)
            
            if match:
                mx, my, mw, mh, score = match
                self.bbox = clamp_bbox((mx, my, orig_w, orig_h), frame.shape)
                self.verify_score = score
                self.tracker, _ = create_tracker()
                self.tracker.init(frame, self.bbox)
                self.recovery_method = f"search({score:.2f})"
                self.last_good_bbox = self.bbox
                self.score_history = [score]
                actual_success = True
            
            if not actual_success:
                self.bbox = self.last_good_bbox if self.last_good_bbox else motion_pred
                self.tracker, _ = create_tracker()
                self.tracker.init(frame, self.bbox)
                self.recovery_method = "hold"
            
            if timing: timing.stop("recovery")

        return actual_success, self.bbox, cam_motion, self.verify_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Video file path")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--no-yolo", action="store_true")
    parser.add_argument("--verify-threshold", type=float, default=0.25)
    parser.add_argument("--output", "-o")

    args = parser.parse_args()
    bbox = (args.x, args.y, args.w, args.h)
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened(): sys.exit(1)
    ret, frame = cap.read()
    if not ret: sys.exit(1)
    fh, fw = frame.shape[:2]

    tracker = RobustTrackerV16(verify_threshold=args.verify_threshold, use_yolo=not args.no_yolo)
    if not tracker.initialize(frame, bbox): sys.exit(1)

    out_path = args.output or (Path(args.video).stem + "_tracked_v16v9.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), (fw, fh))
    
    timing = TimingStats()
    frame_num = 0
    success_count = 0

    while True:
        timing.start("frame_total")
        timing.start("frame_read")
        ret, frame = cap.read()
        timing.stop("frame_read")
        if not ret: break
        
        frame_num += 1
        success, bbox, cam_motion, verify_score = tracker.update(frame, timing)
        if success: success_count += 1
        
        timing.start("drawing")
        x, y, w, h = bbox
        color = (0, 255, 0) if success else (0, 0, 255)
        if tracker.recovery_method: color = (0, 255, 255)
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 0, 0.7, (0, 255, 0), 2)
        timing.stop("drawing")
        
        writer.write(frame)
        cv2.imshow("Tracking v16v9", frame)
        timing.stop("frame_total")
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    timing.print_summary()

if __name__ == "__main__":
    main()
