"""
MixFormerV2 ONNX Tracker Wrapper

A wrapper around MixFormerV2-S ONNX model that provides an interface similar to
the PyTorch version for easy integration with existing tracking pipelines.

Usage:
    tracker = MixFormerONNXTracker()
    tracker.init(frame, bbox)  # bbox = [x, y, w, h]
    success, bbox, score = tracker.update(frame)
"""

import os
import math
import numpy as np
import cv2
from typing import Tuple, List, Optional

try:
    import onnxruntime
except ImportError:
    raise ImportError("onnxruntime-gpu is required. Install with: uv add onnxruntime-gpu")


# Project paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


class Preprocessor:
    """Preprocessor for ONNX model inputs.

    Converts BGR images to normalized float32 arrays ready for ONNX Runtime.
    Uses ImageNet normalization (same as PyTorch version).
    """

    def __init__(self):
        # ImageNet normalization values
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

    def process(self, img_arr: np.ndarray) -> np.ndarray:
        """Convert BGR image array to normalized NCHW float32 array.

        Args:
            img_arr: Input image (H, W, 3) BGR uint8

        Returns:
            Normalized array (1, 3, H, W) float32
        """
        # Convert HWC to NCHW
        img = img_arr.astype(np.float32)
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)  # CHW -> NCHW

        # Normalize
        img = (img / 255.0 - self.mean) / self.std

        # Ensure contiguous memory layout (critical for ONNX Runtime)
        return np.ascontiguousarray(img)


def sample_target(im: np.ndarray, target_bb: List[float], search_area_factor: float,
                  output_sz: Optional[int] = None) -> Tuple[np.ndarray, float, np.ndarray]:
    """Extract a square crop centered at target_bb box.

    Args:
        im: Input image (H, W, 3)
        target_bb: Target box [x, y, w, h]
        search_area_factor: Ratio of crop size to target size
        output_sz: Size to resize the crop to

    Returns:
        im_crop_padded: Cropped and padded image
        resize_factor: The resize factor applied
        att_mask: Attention mask (True for padding, False for valid pixels)
    """
    if not isinstance(target_bb, list):
        x, y, w, h = target_bb.tolist()
    else:
        x, y, w, h = target_bb

    # Crop size based on target size
    crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)

    if crop_sz < 1:
        raise Exception('Too small bounding box.')

    # Crop coordinates
    x1 = int(round(x + 0.5 * w - crop_sz * 0.5))
    x2 = int(x1 + crop_sz)
    y1 = int(round(y + 0.5 * h - crop_sz * 0.5))
    y2 = int(y1 + crop_sz)

    # Padding amounts
    x1_pad = int(max(0, -x1))
    x2_pad = int(max(x2 - im.shape[1] + 1, 0))
    y1_pad = int(max(0, -y1))
    y2_pad = int(max(y2 - im.shape[0] + 1, 0))

    # Crop target
    im_crop = im[y1 + y1_pad:y2 - y2_pad, x1 + x1_pad:x2 - x2_pad, :]

    # Pad
    im_crop_padded = cv2.copyMakeBorder(im_crop, y1_pad, y2_pad, x1_pad, x2_pad, cv2.BORDER_CONSTANT)

    # Create attention mask
    H, W, _ = im_crop_padded.shape
    att_mask = np.ones((H, W))
    end_x, end_y = -x2_pad, -y2_pad
    if y2_pad == 0:
        end_y = None
    if x2_pad == 0:
        end_x = None
    att_mask[y1_pad:end_y, x1_pad:end_x] = 0

    if output_sz is not None:
        resize_factor = output_sz / crop_sz
        im_crop_padded = cv2.resize(im_crop_padded, (output_sz, output_sz))
        att_mask = cv2.resize(att_mask, (output_sz, output_sz)).astype(np.bool_)
        return im_crop_padded, resize_factor, att_mask

    return im_crop_padded, 1.0, att_mask.astype(np.bool_)


def clip_box(box: List[float], H: int, W: int, margin: int = 0) -> List[float]:
    """Clip box to image boundaries."""
    x1, y1, w, h = box
    x2, y2 = x1 + w, y1 + h
    x1 = min(max(0, x1), W - margin)
    x2 = min(max(margin, x2), W)
    y1 = min(max(0, y1), H - margin)
    y2 = min(max(margin, y2), H)
    w = max(margin, x2 - x1)
    h = max(margin, y2 - y1)
    return [x1, y1, w, h]


class MixFormerONNXTracker:
    """MixFormerV2-S ONNX Tracker wrapper with OpenCV-like interface.

    This class wraps the MixFormerV2-S ONNX model to provide a simple tracking API.
    It handles template initialization, online template updating, and ONNX inference.

    Attributes:
        search_factor: Search region size factor (default 4.5)
        search_size: Search region size in pixels (default 224)
        template_factor: Template region size factor (default 2.0)
        template_size: Template size in pixels (default 112)
        update_interval: Frames between template updates (default 200)
    """

    def __init__(self,
                 model_path: Optional[str] = None,
                 device: str = 'cuda',
                 search_factor: float = 4.5,
                 search_size: int = 224,
                 template_factor: float = 2.0,
                 template_size: int = 112,
                 update_interval: int = 200):
        """Initialize the MixFormerV2-S ONNX tracker.

        Args:
            model_path: Path to ONNX model. If None, uses default location.
            device: Device to run on ('cuda' or 'cpu')
            search_factor: Search region size factor
            search_size: Search region size in pixels
            template_factor: Template region size factor
            template_size: Template size in pixels
            update_interval: Frames between automatic template updates
        """
        self.device = device
        self.search_factor = search_factor
        self.search_size = search_size
        self.template_factor = template_factor
        self.template_size = template_size
        self.update_interval = update_interval

        # Find model path
        if model_path is None:
            model_path = os.path.join(PROJECT_DIR, 'models', 'mixformerv2', 'onnx', 'mixformer_v2_s.onnx')

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {model_path}. "
                f"Run 'uv run python scripts/export_onnx.py' to create it."
            )

        self.model_path = model_path
        self.preprocessor = Preprocessor()

        # Load ONNX model
        self._load_model()

        # State
        self.template = None
        self.online_template = None
        self.online_max_template = None
        self.state = None
        self.frame_id = 0
        self.max_pred_score = -1.0
        self.max_score_decay = 1.0

    def _load_model(self):
        """Load the ONNX model with ONNX Runtime."""
        # Configure providers based on device
        if self.device == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            provider_options = [{'device_id': '0'}, {}]
        else:
            providers = ['CPUExecutionProvider']
            provider_options = [{}]

        # Create session options
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Create inference session
        self.session = onnxruntime.InferenceSession(
            self.model_path,
            sess_options=sess_options,
            providers=providers,
            provider_options=provider_options
        )

        # Get actual provider
        actual_provider = self.session.get_providers()[0]
        print(f"[MixFormerONNX] Loaded model from {self.model_path}")
        print(f"[MixFormerONNX] Using provider: {actual_provider}")

    def init(self, frame: np.ndarray, bbox: List[float]) -> bool:
        """Initialize tracking with the first frame and bounding box.

        Args:
            frame: First frame (H, W, 3) BGR image
            bbox: Initial bounding box [x, y, w, h]

        Returns:
            True if initialization succeeded
        """
        try:
            self.state = list(bbox)
            self.frame_id = 0

            # Extract template
            z_patch_arr, _, _ = sample_target(
                frame, self.state, self.template_factor, output_sz=self.template_size
            )
            template = self.preprocessor.process(z_patch_arr)

            # Initialize templates
            self.template = template
            self.online_template = template.copy()
            self.online_max_template = template.copy()
            self.max_pred_score = -1.0

            return True
        except Exception as e:
            print(f"[MixFormerONNX] Init failed: {e}")
            return False

    def update(self, frame: np.ndarray) -> Tuple[bool, List[float], float]:
        """Update tracker with new frame.

        Args:
            frame: New frame (H, W, 3) BGR image

        Returns:
            success: True if tracking succeeded
            bbox: Updated bounding box [x, y, w, h]
            score: Confidence score
        """
        if self.state is None:
            return False, [0, 0, 0, 0], 0.0

        H, W, _ = frame.shape
        self.frame_id += 1

        try:
            # Extract search region
            x_patch_arr, resize_factor, _ = sample_target(
                frame, self.state, self.search_factor, output_sz=self.search_size
            )
            search = self.preprocessor.process(x_patch_arr)

            # Run ONNX inference
            ort_inputs = {
                'img_t': self.template,
                'img_ot': self.online_template,
                'img_search': search
            }
            pred_boxes, pred_scores = self.session.run(None, ort_inputs)

            # Convert to proper shapes
            pred_boxes = np.array(pred_boxes)  # (1, 1, 4) or similar
            pred_scores = np.array(pred_scores)  # (1, 1) or similar

            # Flatten predictions
            if pred_boxes.ndim > 2:
                pred_boxes = pred_boxes.reshape(-1, 4)
            pred_box = (pred_boxes.mean(axis=0) * self.search_size / resize_factor).tolist()

            # Get confidence score (already sigmoid'd in ONNX model)
            pred_score = float(pred_scores.flatten()[0])

            # Map back to image coordinates
            self.state = clip_box(self._map_box_back(pred_box, resize_factor), H, W, margin=10)

            # Update score tracking
            self.max_pred_score = self.max_pred_score * self.max_score_decay

            # Update online template if score is good
            if pred_score > 0.5 and pred_score > self.max_pred_score:
                z_patch_arr, _, _ = sample_target(
                    frame, self.state, self.template_factor, output_sz=self.template_size
                )
                self.online_max_template = self.preprocessor.process(z_patch_arr)
                self.max_pred_score = pred_score

            # Periodic template update
            if self.frame_id % self.update_interval == 0:
                self.online_template = self.online_max_template.copy()
                self.max_pred_score = -1
                self.online_max_template = self.template.copy()

            return True, self.state, pred_score

        except Exception as e:
            print(f"[MixFormerONNX] Update failed: {e}")
            import traceback
            traceback.print_exc()
            return False, self.state if self.state else [0, 0, 0, 0], 0.0

    def _map_box_back(self, pred_box: List[float], resize_factor: float) -> List[float]:
        """Map predicted box back to original image coordinates."""
        cx_prev = self.state[0] + 0.5 * self.state[2]
        cy_prev = self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def get_state(self) -> List[float]:
        """Get current tracking state (bounding box)."""
        return self.state if self.state else [0, 0, 0, 0]


def create_tracker(model_path: Optional[str] = None, device: str = 'cuda') -> MixFormerONNXTracker:
    """Factory function to create a MixFormerONNXTracker instance.

    Args:
        model_path: Path to ONNX model
        device: Device to run on

    Returns:
        MixFormerONNXTracker instance
    """
    return MixFormerONNXTracker(model_path=model_path, device=device)


if __name__ == '__main__':
    # Simple test
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Path to video file')
    parser.add_argument('x', type=int, help='Initial x')
    parser.add_argument('y', type=int, help='Initial y')
    parser.add_argument('w', type=int, help='Initial width')
    parser.add_argument('h', type=int, help='Initial height')
    args = parser.parse_args()

    # Create tracker
    tracker = MixFormerONNXTracker()

    # Open video
    cap = cv2.VideoCapture(args.video)

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id == 0:
            # Initialize
            bbox = [args.x, args.y, args.w, args.h]
            tracker.init(frame, bbox)
        else:
            # Track
            success, bbox, score = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Score: {score:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('MixFormer ONNX Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
