#!/usr/bin/env python3
"""
Learning-Free Drone Detector + Tracker

The key insight: The drone moves INDEPENDENTLY of the camera.
When we compensate for camera motion, everything except the drone should be still.

Detection Pipeline:
1. Estimate camera motion (homography from sparse optical flow)
2. Warp previous frame to align with current
3. Compute difference - only independent motion remains
4. Find and cluster motion blobs
5. Track the most consistent blob

Usage:
    python track.py --video test_track_dron1.mp4
    python track.py --video test_track_dron1.mp4 --output tracked.mp4
"""

import cv2
import numpy as np
import argparse
import time
from typing import Tuple, Optional, List
from dataclasses import dataclass, field
from collections import deque


# ============================================================================
# UTILITY FUNCTIONS
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
               content: Tuple[int, int, int, int],
               min_size: int = 10) -> Optional[Tuple[int, int, int, int]]:
    """Clamp bbox to content region."""
    x, y, w, h = bbox
    cx, cy, cw, ch = content
    
    w = max(min_size, w)
    h = max(min_size, h)
    x = max(cx, min(x, cx + cw - w))
    y = max(cy, min(y, cy + ch - h))
    
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return None
    
    return (int(x), int(y), int(w), int(h))


def bbox_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    """Get center of bbox."""
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def bbox_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    """Calculate IoU between two bboxes."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    
    inter = (xi2 - xi1) * (yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    
    return inter / union if union > 0 else 0.0


# ============================================================================
# CAMERA MOTION ESTIMATION
# ============================================================================

class CameraMotionEstimator:
    """
    Estimates camera motion using sparse optical flow + RANSAC homography.
    """
    
    def __init__(self, content: Tuple[int, int, int, int]):
        self.content = content
        self.prev_gray = None
        self.prev_pts = None
        
        # Feature detection params
        self.feature_params = dict(
            maxCorners=300,
            qualityLevel=0.01,
            minDistance=10,
            blockSize=7
        )
        
        # Optical flow params
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
    
    def _get_content_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create mask for content region."""
        mask = np.zeros(shape, dtype=np.uint8)
        cx, cy, cw, ch = self.content
        mask[cy:cy+ch, cx:cx+cw] = 255
        return mask
    
    def estimate(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate camera motion from previous frame to current.
        Returns 3x3 homography matrix or None.
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            mask = self._get_content_mask(gray.shape)
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            return None
        
        if self.prev_pts is None or len(self.prev_pts) < 10:
            # Re-detect features
            mask = self._get_content_mask(gray.shape)
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, mask=mask, **self.feature_params)
            if self.prev_pts is None or len(self.prev_pts) < 10:
                self.prev_gray = gray.copy()
                return None
        
        # Track points
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )
        
        # Filter good points
        status = status.ravel()
        prev_good = self.prev_pts[status == 1].reshape(-1, 2)
        curr_good = curr_pts[status == 1].reshape(-1, 2)
        
        H = None
        if len(prev_good) >= 10:
            # Estimate homography with RANSAC
            H, inliers = cv2.findHomography(prev_good, curr_good, cv2.RANSAC, 3.0)
            
            if H is not None and inliers is not None:
                inlier_ratio = np.sum(inliers) / len(inliers)
                if inlier_ratio < 0.5:
                    H = None  # Not enough consensus
        
        # Update for next frame
        self.prev_gray = gray.copy()
        mask = self._get_content_mask(gray.shape)
        self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
        
        return H


# ============================================================================
# INDEPENDENT MOTION DETECTOR
# ============================================================================

@dataclass
class Detection:
    """A single detection."""
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    area: float
    frame: int
    confidence: float = 1.0


@dataclass 
class Track:
    """A tracked object across frames."""
    id: int
    detections: List[Detection] = field(default_factory=list)
    lost_frames: int = 0
    
    @property
    def last_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self.detections[-1].bbox if self.detections else None
    
    @property
    def last_center(self) -> Optional[Tuple[float, float]]:
        return self.detections[-1].center if self.detections else None
    
    @property
    def avg_size(self) -> Tuple[float, float]:
        if not self.detections:
            return (30, 20)
        ws = [d.bbox[2] for d in self.detections[-10:]]
        hs = [d.bbox[3] for d in self.detections[-10:]]
        return (np.median(ws), np.median(hs))
    
    @property
    def velocity(self) -> Tuple[float, float]:
        if len(self.detections) < 2:
            return (0, 0)
        c1 = self.detections[-2].center
        c2 = self.detections[-1].center
        return (c2[0] - c1[0], c2[1] - c1[1])
    
    def predict_position(self, frames_ahead: int = 1) -> Tuple[float, float]:
        if not self.detections:
            return (0, 0)
        cx, cy = self.last_center
        vx, vy = self.velocity
        return (cx + vx * frames_ahead, cy + vy * frames_ahead)


class IndependentMotionDetector:
    """
    Detects objects moving independently of camera motion.
    
    Algorithm:
    1. Warp previous frame using camera homography
    2. Compute absolute difference with current frame
    3. Threshold and find contours
    4. Filter by size and shape
    """
    
    def __init__(self, content: Tuple[int, int, int, int],
                 min_area: int = 30, max_area: int = 10000,
                 diff_threshold: int = 20):
        self.content = content
        self.min_area = min_area
        self.max_area = max_area
        self.diff_threshold = diff_threshold
        self.prev_gray = None
        
    def detect(self, gray: np.ndarray, H: Optional[np.ndarray], 
               frame_num: int) -> List[Detection]:
        """
        Detect independently moving objects.
        
        Args:
            gray: Current grayscale frame
            H: Camera motion homography (prev -> current)
            frame_num: Current frame number
            
        Returns:
            List of Detection objects
        """
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            return []
        
        # Warp previous frame to align with current
        if H is not None:
            prev_warped = cv2.warpPerspective(
                self.prev_gray, H, (gray.shape[1], gray.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )
        else:
            prev_warped = self.prev_gray
        
        # Compute difference
        diff = cv2.absdiff(prev_warped, gray)
        
        # Threshold
        _, thresh = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        cx, cy, cw, ch = self.content
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            if area < self.min_area or area > self.max_area:
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            center = (x + w / 2, y + h / 2)
            
            # Must be in content region
            if not (cx <= center[0] < cx + cw and cy <= center[1] < cy + ch):
                continue
            
            # Aspect ratio filter (drone-like shapes)
            aspect = max(w, h) / (min(w, h) + 1e-5)
            if aspect > 5:  # Too elongated
                continue
            
            detections.append(Detection(
                bbox=(x, y, w, h),
                center=center,
                area=area,
                frame=frame_num
            ))
        
        self.prev_gray = gray.copy()
        return detections


# ============================================================================
# MULTI-OBJECT TRACKER (Simple IoU-based)
# ============================================================================

class SimpleTracker:
    """
    Simple multi-object tracker using IoU matching.
    Maintains track continuity across frames.
    """
    
    def __init__(self, iou_threshold: float = 0.2, max_lost: int = 30):
        self.tracks: List[Track] = []
        self.next_id = 0
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
    
    def update(self, detections: List[Detection]) -> List[Track]:
        """
        Update tracks with new detections.
        Returns active tracks.
        """
        # Predict positions for lost tracks
        for track in self.tracks:
            if track.lost_frames > 0:
                # Use velocity to predict
                pred_cx, pred_cy = track.predict_position(track.lost_frames)
                w, h = track.avg_size
                # Update last detection with prediction
                if track.detections:
                    track.detections[-1] = Detection(
                        bbox=(int(pred_cx - w/2), int(pred_cy - h/2), int(w), int(h)),
                        center=(pred_cx, pred_cy),
                        area=w * h,
                        frame=track.detections[-1].frame
                    )
        
        # Match detections to existing tracks
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))
        
        if detections and self.tracks:
            # Compute IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            
            for i, track in enumerate(self.tracks):
                if track.last_bbox is None:
                    continue
                for j, det in enumerate(detections):
                    iou_matrix[i, j] = bbox_iou(track.last_bbox, det.bbox)
            
            # Greedy matching
            while True:
                if iou_matrix.size == 0:
                    break
                    
                max_iou = iou_matrix.max()
                if max_iou < self.iou_threshold:
                    break
                
                i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
                
                # Match track i with detection j
                self.tracks[i].detections.append(detections[j])
                self.tracks[i].lost_frames = 0
                
                if i in unmatched_tracks:
                    unmatched_tracks.remove(i)
                if j in unmatched_detections:
                    unmatched_detections.remove(j)
                
                # Remove matched row/col
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0
        
        # Increment lost count for unmatched tracks
        for i in unmatched_tracks:
            self.tracks[i].lost_frames += 1
        
        # Create new tracks for unmatched detections
        for j in unmatched_detections:
            new_track = Track(id=self.next_id, detections=[detections[j]])
            self.tracks.append(new_track)
            self.next_id += 1
        
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.lost_frames < self.max_lost]
        
        return self.tracks
    
    def get_best_track(self) -> Optional[Track]:
        """
        Get the most likely drone track.
        Prioritizes: longest track, most recent detections, consistent motion.
        """
        if not self.tracks:
            return None
        
        def score(track: Track) -> float:
            # Length score
            length_score = len(track.detections)
            
            # Recency score (penalize lost tracks)
            recency_score = max(0, 10 - track.lost_frames)
            
            # Consistency score (smooth motion)
            if len(track.detections) >= 3:
                velocities = []
                for i in range(len(track.detections) - 1):
                    c1 = track.detections[i].center
                    c2 = track.detections[i + 1].center
                    velocities.append((c2[0] - c1[0], c2[1] - c1[1]))
                
                # Low variance = consistent motion
                if velocities:
                    vx_var = np.var([v[0] for v in velocities])
                    vy_var = np.var([v[1] for v in velocities])
                    consistency_score = 10 / (1 + vx_var + vy_var)
                else:
                    consistency_score = 0
            else:
                consistency_score = 0
            
            return length_score * 2 + recency_score + consistency_score
        
        return max(self.tracks, key=score)


# ============================================================================
# MAIN TRACKING PIPELINE
# ============================================================================

class DroneTracker:
    """
    Complete drone tracking pipeline:
    1. Camera motion estimation
    2. Independent motion detection  
    3. Multi-object tracking
    4. Best track selection
    """
    
    def __init__(self, content: Tuple[int, int, int, int]):
        self.content = content
        self.camera_estimator = CameraMotionEstimator(content)
        self.detector = IndependentMotionDetector(content)
        self.tracker = SimpleTracker(iou_threshold=0.15, max_lost=30)
        self.frame_count = 0
        self.initialized = False
        self.warmup_frames = 5  # Frames to collect before tracking
        
    def update(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], dict]:
        """
        Process a frame and return the drone bbox.
        
        Returns:
            (bbox or None, debug_info dict)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.frame_count += 1
        
        debug = {
            "frame": self.frame_count,
            "num_detections": 0,
            "num_tracks": 0,
            "camera_motion": False
        }
        
        # Estimate camera motion
        H = self.camera_estimator.estimate(gray)
        debug["camera_motion"] = H is not None
        
        # Detect independent motion
        detections = self.detector.detect(gray, H, self.frame_count)
        debug["num_detections"] = len(detections)
        
        # Update tracker
        tracks = self.tracker.update(detections)
        debug["num_tracks"] = len(tracks)
        
        # Get best track
        best = self.tracker.get_best_track()
        
        if best and best.lost_frames < 15:
            bbox = best.last_bbox
            bbox = clamp_bbox(bbox, self.content)
            debug["track_id"] = best.id
            debug["track_length"] = len(best.detections)
            debug["lost_frames"] = best.lost_frames
            return bbox, debug
        
        return None, debug


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Learning-Free Drone Tracker with Detection")
    parser.add_argument("--video", required=True, help="Path to video")
    parser.add_argument("--output", default="output_tracked.mp4", help="Output path")
    parser.add_argument("--show", action="store_true", help="Show preview")
    parser.add_argument("--min-area", type=int, default=30, help="Min detection area")
    parser.add_argument("--max-area", type=int, default=8000, help="Max detection area")
    args = parser.parse_args()
    
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
    
    # Read first frame to detect content region
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return
    
    content = detect_content_region(frame)
    print(f"Content region: x={content[0]}, y={content[1]}, w={content[2]}, h={content[3]}")
    
    # Reset video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    # Initialize tracker
    tracker = DroneTracker(content)
    tracker.detector.min_area = args.min_area
    tracker.detector.max_area = args.max_area
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    # Process
    frame_times = []
    frame_count = 0
    
    print("\nProcessing...")
    print("(First few frames are warmup for motion estimation)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start = time.perf_counter()
        bbox, debug = tracker.update(frame)
        elapsed = time.perf_counter() - start
        frame_times.append(elapsed)
        
        frame_count += 1
        
        # Draw detections count
        cx, cy, cw, ch = content
        
        # Draw content boundary (dim)
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (50, 50, 50), 1)
        
        # Draw bbox if found
        if bbox:
            x, y, w, h = bbox
            
            # Color based on confidence
            lost = debug.get("lost_frames", 0)
            if lost == 0:
                color = (0, 255, 0)  # Green - active detection
            elif lost < 5:
                color = (0, 255, 255)  # Yellow - recent
            else:
                color = (0, 165, 255)  # Orange - predicted
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            # Label
            label = f"Drone (T{debug.get('track_id', '?')})"
            cv2.putText(frame, label, (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Status overlay
        ms = elapsed * 1000
        status = f"{ms:.1f}ms | Dets:{debug['num_detections']} Tracks:{debug['num_tracks']}"
        if debug.get("camera_motion"):
            status += " | CAM"
        
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        out.write(frame)
        
        if args.show:
            cv2.imshow("Drone Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if frame_count % 100 == 0:
            avg = sum(frame_times[-100:]) / len(frame_times[-100:]) * 1000
            print(f"Frame {frame_count}/{total_frames} - {avg:.1f}ms/frame")
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Stats
    avg_ms = sum(frame_times) / len(frame_times) * 1000
    print(f"\nDone! {frame_count} frames")
    print(f"Average: {avg_ms:.1f}ms ({1000/avg_ms:.0f} FPS)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
