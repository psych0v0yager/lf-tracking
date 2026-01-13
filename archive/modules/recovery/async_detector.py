"""
Asynchronous YOLO/SAHI Detection for Recovery

Runs object detection in a background thread to recover
the drone when tracking confidence drops.

Uses SAHI (Sliced Aided Hyper Inference) for small object detection.
"""

import cv2
import numpy as np
import threading
import time
from queue import Queue, Empty
from typing import Tuple, Optional, List
from pathlib import Path


class AsyncDetector:
    """
    Asynchronous object detector for tracking recovery.

    Features:
    - Runs detection in background thread
    - Non-blocking interface
    - SAHI support for small object detection
    - Cooldown to limit detection frequency
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 confidence_threshold: float = 0.25,
                 use_sahi: bool = True,
                 slice_size: int = 320,
                 overlap_ratio: float = 0.2,
                 cooldown_seconds: float = 0.1):
        """
        Args:
            model_path: Path to YOLO model (default: yolo11n.pt)
            confidence_threshold: Detection confidence threshold
            use_sahi: Whether to use sliced inference for small objects
            slice_size: Size of slices for SAHI
            overlap_ratio: Overlap between slices
            cooldown_seconds: Minimum time between detections
        """
        self.confidence_threshold = confidence_threshold
        self.use_sahi = use_sahi
        self.slice_size = slice_size
        self.overlap_ratio = overlap_ratio
        self.cooldown_seconds = cooldown_seconds

        self.model = None
        self.model_path = model_path
        self.sahi_model = None

        # Threading
        self.request_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=5)
        self.worker_thread = None
        self.running = False

        # State
        self.last_detection_time = 0
        self.detection_count = 0

    def start(self):
        """Start the background detection thread."""
        if self.running:
            return

        self._load_model()

        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        """Stop the background detection thread."""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=1.0)

    def _load_model(self):
        """Load YOLO model."""
        try:
            from ultralytics import YOLO

            if self.model_path is None:
                # Look for model in common locations
                possible_paths = [
                    Path("yolo11n.pt"),
                    Path("../yolo/yolo11n.pt"),
                    Path("/home/overseer/Desktop/9mothers/yolo/yolo11n.pt"),
                ]
                for p in possible_paths:
                    if p.exists():
                        self.model_path = str(p)
                        break

            if self.model_path is None:
                print("Warning: YOLO model not found, downloading yolo11n.pt")
                self.model_path = "yolo11n.pt"

            self.model = YOLO(self.model_path)
            print(f"Loaded YOLO model: {self.model_path}")

            if self.use_sahi:
                try:
                    from sahi import AutoDetectionModel
                    self.sahi_model = AutoDetectionModel.from_pretrained(
                        model_type='yolov8',
                        model_path=self.model_path,
                        confidence_threshold=self.confidence_threshold,
                        device='cuda:0'
                    )
                    print("SAHI model loaded for sliced inference")
                except ImportError:
                    print("SAHI not available, using standard inference")
                    self.use_sahi = False
                except Exception as e:
                    print(f"SAHI init failed: {e}, using standard inference")
                    self.use_sahi = False

        except ImportError:
            print("Warning: ultralytics not installed, detection disabled")
            self.model = None

    def request_detection(self,
                          frame: np.ndarray,
                          search_region: Optional[Tuple[int, int, int, int]] = None,
                          confidence: float = 1.0,
                          timestamp: float = None) -> bool:
        """
        Request a detection (non-blocking).

        Args:
            frame: BGR frame to detect in
            search_region: Optional region to search (x, y, w, h)
            confidence: Current tracker confidence (detection triggered when low)
            timestamp: Frame timestamp

        Returns:
            True if request was queued, False if skipped (cooldown/queue full)
        """
        if not self.running or self.model is None:
            return False

        if timestamp is None:
            timestamp = time.time()

        # Check cooldown
        if timestamp - self.last_detection_time < self.cooldown_seconds:
            return False

        # Only trigger if confidence is low
        if confidence > 0.5:
            return False

        try:
            self.request_queue.put_nowait({
                'frame': frame.copy(),
                'search_region': search_region,
                'timestamp': timestamp
            })
            return True
        except:
            return False

    def get_result(self) -> Optional[dict]:
        """
        Get detection result (non-blocking).

        Returns:
            Dict with 'detections', 'timestamp', 'best_match' or None
        """
        try:
            return self.result_queue.get_nowait()
        except Empty:
            return None

    def _worker_loop(self):
        """Background worker that processes detection requests."""
        while self.running:
            try:
                request = self.request_queue.get(timeout=0.1)
            except Empty:
                continue

            frame = request['frame']
            search_region = request['search_region']
            timestamp = request['timestamp']

            # Run detection
            detections = self._detect(frame, search_region)

            # Find best match (largest detection in search region)
            best_match = None
            if detections:
                best_match = max(detections, key=lambda d: d['area'])

            result = {
                'detections': detections,
                'timestamp': timestamp,
                'best_match': best_match,
                'search_region': search_region
            }

            try:
                self.result_queue.put_nowait(result)
            except:
                pass

            self.last_detection_time = timestamp
            self.detection_count += 1

    def _detect(self,
                frame: np.ndarray,
                search_region: Optional[Tuple[int, int, int, int]]) -> List[dict]:
        """
        Run detection on frame.

        Args:
            frame: BGR frame
            search_region: Optional (x, y, w, h) to crop

        Returns:
            List of detections with 'bbox', 'confidence', 'class'
        """
        if self.model is None:
            return []

        # Crop to search region if provided
        if search_region is not None:
            sx, sy, sw, sh = search_region
            crop = frame[sy:sy+sh, sx:sx+sw]
            offset = (sx, sy)
        else:
            crop = frame
            offset = (0, 0)

        if crop.size == 0:
            return []

        detections = []

        if self.use_sahi and self.sahi_model is not None:
            detections = self._detect_sahi(crop, offset)
        else:
            detections = self._detect_standard(crop, offset)

        return detections

    def _detect_standard(self,
                         image: np.ndarray,
                         offset: Tuple[int, int]) -> List[dict]:
        """Standard YOLO inference."""
        results = self.model(image, verbose=False, conf=self.confidence_threshold)

        detections = []
        for r in results:
            boxes = r.boxes
            if boxes is None:
                continue

            for i in range(len(boxes)):
                box = boxes[i]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # Adjust coordinates for offset
                x1 += offset[0]
                y1 += offset[1]
                x2 += offset[0]
                y2 += offset[1]

                w = x2 - x1
                h = y2 - y1

                detections.append({
                    'bbox': (int(x1), int(y1), int(w), int(h)),
                    'confidence': conf,
                    'class': cls,
                    'area': w * h
                })

        return detections

    def _detect_sahi(self,
                     image: np.ndarray,
                     offset: Tuple[int, int]) -> List[dict]:
        """Sliced inference using SAHI for small objects."""
        try:
            from sahi.predict import get_sliced_prediction

            result = get_sliced_prediction(
                image,
                self.sahi_model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap_ratio,
                overlap_width_ratio=self.overlap_ratio,
                perform_standard_pred=True,
                postprocess_type="NMS",
                postprocess_match_threshold=0.5,
                verbose=0
            )

            detections = []
            for obj in result.object_prediction_list:
                box = obj.bbox
                x1, y1 = box.minx + offset[0], box.miny + offset[1]
                w, h = box.maxx - box.minx, box.maxy - box.miny

                detections.append({
                    'bbox': (int(x1), int(y1), int(w), int(h)),
                    'confidence': obj.score.value,
                    'class': obj.category.id,
                    'area': w * h
                })

            return detections

        except Exception as e:
            print(f"SAHI detection error: {e}")
            return self._detect_standard(image, offset)

    def filter_by_size(self,
                       detections: List[dict],
                       expected_size: Tuple[int, int],
                       tolerance: float = 0.5) -> List[dict]:
        """
        Filter detections by expected size.

        Args:
            detections: List of detections
            expected_size: Expected (width, height)
            tolerance: Size tolerance (0.5 = 50% difference allowed)

        Returns:
            Filtered detections
        """
        ew, eh = expected_size
        filtered = []

        for det in detections:
            x, y, w, h = det['bbox']

            # Check size ratio
            w_ratio = w / ew if ew > 0 else float('inf')
            h_ratio = h / eh if eh > 0 else float('inf')

            if (1 - tolerance) <= w_ratio <= (1 + tolerance):
                if (1 - tolerance) <= h_ratio <= (1 + tolerance):
                    filtered.append(det)

        return filtered

    def find_nearest(self,
                     detections: List[dict],
                     center: Tuple[float, float]) -> Optional[dict]:
        """
        Find detection nearest to expected center.

        Args:
            detections: List of detections
            center: Expected (cx, cy)

        Returns:
            Nearest detection or None
        """
        if not detections:
            return None

        cx, cy = center
        best = None
        best_dist = float('inf')

        for det in detections:
            x, y, w, h = det['bbox']
            det_cx = x + w/2
            det_cy = y + h/2

            dist = np.sqrt((det_cx - cx)**2 + (det_cy - cy)**2)
            if dist < best_dist:
                best_dist = dist
                best = det

        return best
