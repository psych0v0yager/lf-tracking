#!/usr/bin/env python3
"""
Small Object Drone Tracker

Optimized for:
- Small targets (~50-100 pixels)
- Heavy camera motion
- No deep learning

Strategy:
1. Primary: Correlation filter (CSRT/KCF) for frame-to-frame tracking
2. Camera motion compensation: Predict bbox shift from global motion
3. Recovery: Motion-based detection when tracker confidence drops
4. Validation: Ensure tracked region matches initial appearance

Usage:
    python track.py --video test_track_dron1.mp4 --bbox 1314,623,73,46
"""

import cv2
import numpy as np
import argparse
import time
from typing import Tuple, Optional, List
from collections import deque


# ============================================================================
# UTILITIES
# ============================================================================

def detect_content_region(frame: np.ndarray, threshold: int = 10) -> Tuple[int, int, int, int]:
    """Detect actual content area, excluding black letterboxing."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    H, W = gray.shape
    
    row_means = np.mean(gray, axis=1)
    content_rows = np.where(row_means > threshold)[0]
    col_means = np.mean(gray, axis=0)
    content_cols = np.where(col_means > threshold)[0]
    
    if len(content_rows) == 0 or len(content_cols) == 0:
        return (0, 0, W, H)
    
    y1, y2 = content_rows[0], content_rows[-1]
    x1, x2 = content_cols[0], content_cols[-1]
    
    return (x1, y1, x2 - x1 + 1, y2 - y1 + 1)


def clamp_bbox(bbox: Tuple[int, int, int, int], 
               frame_size: Tuple[int, int],
               content: Optional[Tuple[int, int, int, int]] = None) -> Optional[Tuple[int, int, int, int]]:
    """Clamp bbox to frame/content bounds."""
    x, y, w, h = bbox
    H, W = frame_size
    
    if content:
        cx, cy, cw, ch = content
        x = max(cx, min(x, cx + cw - w))
        y = max(cy, min(y, cy + ch - h))
    else:
        x = max(0, min(x, W - w))
        y = max(0, min(y, H - h))
    
    if w <= 0 or h <= 0 or x < 0 or y < 0:
        return None
    
    return (int(x), int(y), int(w), int(h))


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x, y, w, h = bbox
    return (x + w/2, y + h/2)


def center_to_bbox(cx: float, cy: float, w: int, h: int) -> Tuple[int, int, int, int]:
    return (int(cx - w/2), int(cy - h/2), w, h)


# ============================================================================
# CAMERA MOTION ESTIMATION
# ============================================================================

class CameraMotionEstimator:
    """
    Fast camera motion estimation using sparse optical flow.
    Returns translation (dx, dy) and optionally scale.
    """
    
    def __init__(self, content: Tuple[int, int, int, int], max_corners: int = 200):
        self.content = content
        self.max_corners = max_corners
        self.prev_gray = None
        self.prev_pts = None
        
    def _get_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        mask = np.zeros(shape, dtype=np.uint8)
        cx, cy, cw, ch = self.content
        mask[cy:cy+ch, cx:cx+cw] = 255
        return mask
    
    def estimate(self, gray: np.ndarray, 
                 exclude_bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[float, float, float]:
        """
        Estimate camera motion.
        
        Args:
            gray: Current grayscale frame
            exclude_bbox: Exclude this region from motion estimation (the target)
            
        Returns:
            (dx, dy, scale) - translation and scale factor
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            mask = self._get_mask(gray.shape)
            self.prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=self.max_corners, qualityLevel=0.01,
                minDistance=10, mask=mask, blockSize=7
            )
            return (0.0, 0.0, 1.0)
        
        if self.prev_pts is None or len(self.prev_pts) < 20:
            mask = self._get_mask(gray.shape)
            self.prev_pts = cv2.goodFeaturesToTrack(
                self.prev_gray, maxCorners=self.max_corners, qualityLevel=0.01,
                minDistance=10, mask=mask, blockSize=7
            )
            if self.prev_pts is None or len(self.prev_pts) < 20:
                self.prev_gray = gray.copy()
                return (0.0, 0.0, 1.0)
        
        # Track points
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        status = status.ravel()
        prev_good = self.prev_pts[status == 1].reshape(-1, 2)
        curr_good = curr_pts[status == 1].reshape(-1, 2)
        
        # Exclude points inside target bbox
        if exclude_bbox is not None and len(prev_good) > 0:
            bx, by, bw, bh = exclude_bbox
            margin = 20
            mask = ~(
                (prev_good[:, 0] >= bx - margin) & (prev_good[:, 0] <= bx + bw + margin) &
                (prev_good[:, 1] >= by - margin) & (prev_good[:, 1] <= by + bh + margin)
            )
            prev_good = prev_good[mask]
            curr_good = curr_good[mask]
        
        dx, dy, scale = 0.0, 0.0, 1.0
        
        if len(prev_good) >= 10:
            # Estimate affine transform with RANSAC
            M, inliers = cv2.estimateAffinePartial2D(
                prev_good, curr_good, method=cv2.RANSAC, ransacReprojThreshold=3.0
            )
            
            if M is not None:
                dx = M[0, 2]
                dy = M[1, 2]
                scale = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
        
        # Update for next frame
        self.prev_gray = gray.copy()
        mask = self._get_mask(gray.shape)
        if exclude_bbox:
            bx, by, bw, bh = exclude_bbox
            margin = 10
            mask[max(0,by-margin):by+bh+margin, max(0,bx-margin):bx+bw+margin] = 0
        
        self.prev_pts = cv2.goodFeaturesToTrack(
            gray, maxCorners=self.max_corners, qualityLevel=0.01,
            minDistance=10, mask=mask, blockSize=7
        )
        
        return (dx, dy, scale)


# ============================================================================
# TEMPLATE MATCHER (for recovery)
# ============================================================================

class TemplateMatcher:
    """
    Multi-scale template matching for recovery.
    """
    
    def __init__(self, search_scales: List[float] = None):
        self.template = None
        self.template_gray = None
        self.search_scales = search_scales or [0.8, 0.9, 1.0, 1.1, 1.2]
        
    def set_template(self, gray: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Set the reference template."""
        x, y, w, h = bbox
        self.template = gray[y:y+h, x:x+w].copy()
        
    def update_template(self, gray: np.ndarray, bbox: Tuple[int, int, int, int], 
                        alpha: float = 0.1):
        """Slowly update template with new observation."""
        if self.template is None:
            self.set_template(gray, bbox)
            return
        
        x, y, w, h = bbox
        new_patch = gray[y:y+h, x:x+w]
        
        if new_patch.shape == self.template.shape:
            self.template = cv2.addWeighted(
                self.template, 1 - alpha, new_patch, alpha, 0
            )
    
    def match(self, gray: np.ndarray, search_region: Tuple[int, int, int, int],
              threshold: float = 0.4) -> Optional[Tuple[int, int, int, int]]:
        """
        Find template in search region.
        
        Returns:
            bbox if found, None otherwise
        """
        if self.template is None:
            return None
        
        sx, sy, sw, sh = search_region
        region = gray[sy:sy+sh, sx:sx+sw]
        
        if region.size == 0:
            return None
        
        best_val = 0
        best_loc = None
        best_scale = 1.0
        
        for scale in self.search_scales:
            th, tw = self.template.shape[:2]
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
            except:
                continue
        
        if best_val >= threshold and best_loc is not None:
            th, tw = self.template.shape[:2]
            w, h = int(tw * best_scale), int(th * best_scale)
            x = sx + best_loc[0]
            y = sy + best_loc[1]
            return (x, y, w, h)
        
        return None


# ============================================================================
# MOTION DETECTOR (for recovery)
# ============================================================================

class MotionDetector:
    """
    Detects independently moving objects after camera motion compensation.
    """
    
    def __init__(self, content: Tuple[int, int, int, int]):
        self.content = content
        self.prev_gray = None
        
    def detect(self, gray: np.ndarray, camera_motion: Tuple[float, float, float],
               min_area: int = 100, max_area: int = 10000) -> List[Tuple[int, int, int, int]]:
        """
        Detect independently moving blobs.
        
        Args:
            gray: Current frame
            camera_motion: (dx, dy, scale) from camera estimator
            
        Returns:
            List of bboxes
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return []
        
        dx, dy, scale = camera_motion
        
        # Create transformation matrix for camera motion
        H, W = gray.shape
        cx, cy = W / 2, H / 2
        
        # Affine matrix: scale around center + translate
        M = np.array([
            [scale, 0, dx + cx * (1 - scale)],
            [0, scale, dy + cy * (1 - scale)]
        ], dtype=np.float32)
        
        # Warp previous frame to align
        prev_warped = cv2.warpAffine(self.prev_gray, M, (W, H), 
                                      borderMode=cv2.BORDER_REPLICATE)
        
        # Difference
        diff = cv2.absdiff(prev_warped, gray)
        
        # Threshold
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        # Morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        results = []
        cont_x, cont_y, cont_w, cont_h = self.content
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                cx_blob = x + w / 2
                cy_blob = y + h / 2
                
                if cont_x <= cx_blob < cont_x + cont_w and cont_y <= cy_blob < cont_y + cont_h:
                    results.append((x, y, w, h))
        
        self.prev_gray = gray.copy()
        return results


# ============================================================================
# MAIN TRACKER
# ============================================================================

class DroneTracker:
    """
    Robust small-object tracker combining:
    1. Correlation filter (CSRT) for frame-to-frame
    2. Camera motion compensation
    3. Template matching recovery
    4. Motion detection recovery
    """
    
    def __init__(self, content: Tuple[int, int, int, int], tracker_type: str = "CSRT"):
        self.content = content
        self.tracker_type = tracker_type
        
        # Components
        self.tracker = None
        self.camera_est = CameraMotionEstimator(content)
        self.template_matcher = TemplateMatcher()
        self.motion_detector = MotionDetector(content)
        
        # State
        self.bbox = None
        self.initial_bbox = None
        self.velocity = (0.0, 0.0)
        self.lost_count = 0
        self.frame_count = 0
        
        # History for smoothing
        self.bbox_history = deque(maxlen=5)
        
    def _create_tracker(self):
        if self.tracker_type == "CSRT":
            return cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            return cv2.TrackerKCF_create()
        else:
            return cv2.legacy.TrackerMOSSE_create()
    
    def init(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Initialize with first frame and bbox."""
        self.bbox = bbox
        self.initial_bbox = bbox
        self.frame_count = 0
        self.lost_count = 0
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Init correlation tracker
        self.tracker = self._create_tracker()
        self.tracker.init(frame, bbox)
        
        # Init template
        self.template_matcher.set_template(gray, bbox)
        
        # Init camera estimator
        self.camera_est.estimate(gray, exclude_bbox=bbox)
        
        # Init motion detector
        self.motion_detector.prev_gray = gray.copy()
        
        self.bbox_history.append(bbox)
        
    def update(self, frame: np.ndarray) -> Tuple[bool, Optional[Tuple[int, int, int, int]], dict]:
        """
        Update tracker with new frame.
        
        Returns:
            (success, bbox, debug_info)
        """
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape
        
        debug = {
            "frame": self.frame_count,
            "method": "none",
            "lost_count": self.lost_count,
            "camera_motion": (0, 0)
        }
        
        # Estimate camera motion (exclude current bbox from estimation)
        dx, dy, scale = self.camera_est.estimate(gray, exclude_bbox=self.bbox)
        debug["camera_motion"] = (dx, dy)
        
        # Predict bbox position based on camera motion
        if self.bbox:
            old_cx, old_cy = bbox_center(self.bbox)
            pred_cx = old_cx + dx + self.velocity[0]
            pred_cy = old_cy + dy + self.velocity[1]
            w, h = self.bbox[2], self.bbox[3]
            
            # Scale adjustment
            if 0.9 < scale < 1.1:
                w = int(w * scale)
                h = int(h * scale)
            
            predicted_bbox = center_to_bbox(pred_cx, pred_cy, w, h)
            predicted_bbox = clamp_bbox(predicted_bbox, (H, W), self.content)
        else:
            predicted_bbox = None
        
        # Method 1: Try correlation tracker
        success, tracked_bbox = self.tracker.update(frame)
        
        if success:
            tracked_bbox = tuple(map(int, tracked_bbox))
            tracked_bbox = clamp_bbox(tracked_bbox, (H, W), self.content)
            
            if tracked_bbox:
                # Validate: check if tracked region looks similar to template
                x, y, w, h = tracked_bbox
                if w > 0 and h > 0:
                    patch = gray[y:y+h, x:x+w]
                    
                    if self.template_matcher.template is not None and patch.size > 0:
                        try:
                            tmpl_resized = cv2.resize(self.template_matcher.template, 
                                                      (patch.shape[1], patch.shape[0]))
                            match_score = cv2.matchTemplate(
                                patch, tmpl_resized, cv2.TM_CCOEFF_NORMED
                            )[0, 0]
                        except:
                            match_score = 0.5
                        
                        debug["match_score"] = match_score
                        
                        if match_score > 0.25:  # Reasonable match
                            self._accept_bbox(tracked_bbox, gray, "tracker")
                            debug["method"] = "tracker"
                            return True, self.bbox, debug
        
        # Method 2: Template matching in predicted region
        if predicted_bbox:
            px, py, pw, ph = predicted_bbox
            search_margin = max(50, pw, ph)
            search_region = (
                max(0, px - search_margin),
                max(0, py - search_margin),
                min(W, px + pw + search_margin) - max(0, px - search_margin),
                min(H, py + ph + search_margin) - max(0, py - search_margin)
            )
            
            found_bbox = self.template_matcher.match(gray, search_region, threshold=0.35)
            
            if found_bbox:
                found_bbox = clamp_bbox(found_bbox, (H, W), self.content)
                if found_bbox:
                    self._accept_bbox(found_bbox, gray, "template")
                    debug["method"] = "template"
                    return True, self.bbox, debug
        
        # Method 3: Motion detection recovery
        if self.lost_count >= 3:
            motion_bboxes = self.motion_detector.detect(
                gray, (dx, dy, scale),
                min_area=50, max_area=15000
            )
            
            if motion_bboxes and predicted_bbox:
                # Find closest to prediction
                pred_center = bbox_center(predicted_bbox)
                best_bbox = None
                best_dist = float('inf')
                
                for mb in motion_bboxes:
                    mc = bbox_center(mb)
                    dist = np.sqrt((mc[0] - pred_center[0])**2 + (mc[1] - pred_center[1])**2)
                    if dist < best_dist:
                        best_dist = dist
                        best_bbox = mb
                
                if best_bbox and best_dist < 150:
                    best_bbox = clamp_bbox(best_bbox, (H, W), self.content)
                    if best_bbox:
                        self._accept_bbox(best_bbox, gray, "motion")
                        debug["method"] = "motion"
                        return True, self.bbox, debug
        
        # Method 4: Use prediction from camera motion
        if predicted_bbox and self.lost_count < 15:
            self.bbox = predicted_bbox
            self.lost_count += 1
            debug["method"] = "prediction"
            debug["lost_count"] = self.lost_count
            return True, self.bbox, debug
        
        # Completely lost
        self.lost_count += 1
        debug["lost_count"] = self.lost_count
        return False, self.bbox, debug
    
    def _accept_bbox(self, bbox: Tuple[int, int, int, int], gray: np.ndarray, method: str):
        """Accept a new bbox and update state."""
        # Update velocity
        if self.bbox:
            old_cx, old_cy = bbox_center(self.bbox)
            new_cx, new_cy = bbox_center(bbox)
            alpha = 0.4
            self.velocity = (
                alpha * (new_cx - old_cx) + (1 - alpha) * self.velocity[0],
                alpha * (new_cy - old_cy) + (1 - alpha) * self.velocity[1]
            )
        
        self.bbox = bbox
        self.bbox_history.append(bbox)
        self.lost_count = 0
        
        # Reinit correlation tracker
        # Create a fresh frame for tracker init (need BGR)
        # We'll do this in update() instead
        
        # Update template slowly
        if method == "tracker":
            self.template_matcher.update_template(gray, bbox, alpha=0.05)
    
    def reinit_tracker(self, frame: np.ndarray):
        """Reinitialize the correlation tracker at current bbox."""
        if self.bbox:
            self.tracker = self._create_tracker()
            self.tracker.init(frame, self.bbox)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Small Object Drone Tracker")
    parser.add_argument("--video", required=True, help="Path to video")
    parser.add_argument("--bbox", required=True, help="Initial bbox as 'x,y,w,h'")
    parser.add_argument("--output", default="output_tracked.mp4", help="Output path")
    parser.add_argument("--tracker", default="CSRT", choices=["CSRT", "KCF", "MOSSE"])
    parser.add_argument("--show", action="store_true", help="Show preview")
    args = parser.parse_args()
    
    # Parse bbox
    try:
        bbox = tuple(map(int, args.bbox.split(',')))
        assert len(bbox) == 4
    except:
        print("Error: bbox must be 'x,y,w,h'")
        return
    
    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open {args.video}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video: {width}x{height} @ {fps:.2f} FPS, {total_frames} frames")
    print(f"Initial bbox: {bbox}")
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    # Detect content region
    content = detect_content_region(frame)
    print(f"Content region: {content}")
    
    # Initialize tracker
    tracker = DroneTracker(content, args.tracker)
    tracker.init(frame, bbox)
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Draw first frame
    x, y, w, h = bbox
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame, "Init", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    out.write(frame)
    
    # Process
    frame_times = []
    frame_count = 1
    reinit_interval = 30  # Reinit correlation tracker periodically
    
    print(f"\nTracking with {args.tracker}...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.perf_counter()
        success, bbox, debug = tracker.update(frame)
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        
        frame_count += 1
        
        # Periodically reinit correlation tracker to prevent drift
        if frame_count % reinit_interval == 0 and success:
            tracker.reinit_tracker(frame)
        
        # Draw
        if success and bbox:
            x, y, w, h = bbox
            
            method = debug.get("method", "")
            lost = debug.get("lost_count", 0)
            
            if lost == 0:
                color = (0, 255, 0)  # Green - active
            elif lost < 5:
                color = (0, 255, 255)  # Yellow - recent
            else:
                color = (0, 165, 255)  # Orange - predicted
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Method label
            label = method[:4].upper() if method else "?"
            cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Status
        ms = elapsed * 1000
        cam_dx, cam_dy = debug.get("camera_motion", (0, 0))
        status = f"{ms:.0f}ms | {debug.get('method', 'LOST')} | cam:({cam_dx:.0f},{cam_dy:.0f})"
        
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0) if success else (0, 0, 255), 2)
        
        out.write(frame)
        
        if args.show:
            cv2.imshow("Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 100 == 0:
            avg = sum(frame_times[-100:]) / len(frame_times[-100:]) * 1000
            print(f"Frame {frame_count}/{total_frames} - {avg:.1f}ms")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    avg_ms = sum(frame_times) / len(frame_times) * 1000
    print(f"\nDone! {frame_count} frames")
    print(f"Average: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
