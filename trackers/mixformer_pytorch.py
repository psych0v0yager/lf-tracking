"""
MixFormerV2 PyTorch Tracker Wrapper

A wrapper around MixFormerV2-S that provides an interface similar to OpenCV trackers
for easy integration with existing tracking pipelines.

Usage:
    tracker = MixFormerTracker()
    tracker.init(frame, bbox)  # bbox = [x, y, w, h]
    success, bbox, score = tracker.update(frame)
"""

import os
import sys
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Tuple, List, Optional

# Add MixFormerV2 to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MIXFORMER_DIR = os.path.join(PROJECT_DIR, 'MixFormerV2')

if MIXFORMER_DIR not in sys.path:
    sys.path.insert(0, MIXFORMER_DIR)


class Preprocessor:
    """Preprocessor for MixFormerV2 inputs."""
    def __init__(self, device='cuda'):
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view((1, 3, 1, 1)).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view((1, 3, 1, 1)).to(device)

    def process(self, img_arr: np.ndarray) -> torch.Tensor:
        """Convert image array to normalized tensor."""
        img_tensor = torch.tensor(img_arr).to(self.device).float().permute((2, 0, 1)).unsqueeze(dim=0)
        img_tensor_norm = ((img_tensor / 255.0) - self.mean) / self.std
        return img_tensor_norm


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


class MixFormerTracker:
    """MixFormerV2-S Tracker wrapper with OpenCV-like interface.

    This class wraps the MixFormerV2-S model to provide a simple tracking API.
    It handles template initialization, online template updating, and inference.

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
        """Initialize the MixFormerV2-S tracker.

        Args:
            model_path: Path to model checkpoint. If None, uses default location.
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
            # Prefer clean checkpoint (no training code dependencies)
            clean_path = os.path.join(PROJECT_DIR, 'models', 'mixformerv2', 'models', 'mixformerv2_small_clean.pth')
            original_path = os.path.join(PROJECT_DIR, 'models', 'mixformerv2', 'models', 'mixformerv2_small.pth.tar')
            model_path = clean_path if os.path.exists(clean_path) else original_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")

        self.model_path = model_path
        self.preprocessor = Preprocessor(device)

        # Load model
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
        """Load the MixFormerV2-S model."""
        # Import model builder
        from lib.models.mixformer2_vit import build_mixformer2_vit_online
        from easydict import EasyDict

        # Create config for MixFormerV2-S (224_depth4_mlp1_score)
        cfg = EasyDict()
        cfg.MODEL = EasyDict()
        cfg.MODEL.VIT_TYPE = 'base_patch16'
        cfg.MODEL.BACKBONE = EasyDict()
        cfg.MODEL.BACKBONE.DEPTH = 4
        cfg.MODEL.BACKBONE.MLP_RATIO = 1
        cfg.MODEL.BACKBONE.PRETRAINED = False
        cfg.MODEL.HEAD_TYPE = 'MLP'
        cfg.MODEL.HEAD_FREEZE_BN = True
        cfg.MODEL.HIDDEN_DIM = 768
        cfg.MODEL.FEAT_SZ = 96
        cfg.MODEL.PREDICT_MASK = False
        cfg.MODEL.PRETRAINED_STATIC = True
        cfg.MODEL.NUM_OBJECT_QUERIES = 1

        cfg.DATA = EasyDict()
        cfg.DATA.SEARCH = EasyDict()
        cfg.DATA.SEARCH.SIZE = self.search_size
        cfg.DATA.TEMPLATE = EasyDict()
        cfg.DATA.TEMPLATE.SIZE = self.template_size
        cfg.DATA.TEMPLATE.NUMBER = 2
        cfg.DATA.MAX_SAMPLE_INTERVAL = [200]

        cfg.TEST = EasyDict()
        cfg.TEST.SEARCH_FACTOR = self.search_factor
        cfg.TEST.SEARCH_SIZE = self.search_size
        cfg.TEST.TEMPLATE_FACTOR = self.template_factor
        cfg.TEST.TEMPLATE_SIZE = self.template_size
        cfg.TEST.UPDATE_INTERVALS = EasyDict()
        cfg.TEST.UPDATE_INTERVALS.DEFAULT = [self.update_interval]
        cfg.TEST.ONLINE_SIZES = EasyDict()
        cfg.TEST.ONLINE_SIZES.DEFAULT = [1]

        # Build model
        self.network = build_mixformer2_vit_online(cfg, train=False)

        # Load checkpoint (weights_only=False needed for older checkpoints with custom classes)
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)

        # Handle different checkpoint formats
        if 'net' in checkpoint:
            state_dict = checkpoint['net']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        # Load state dict
        missing_keys, unexpected_keys = self.network.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"[MixFormerTracker] Missing keys: {len(missing_keys)}")
        if unexpected_keys:
            print(f"[MixFormerTracker] Unexpected keys: {len(unexpected_keys)}")

        self.network = self.network.to(self.device)
        self.network.eval()
        print(f"[MixFormerTracker] Loaded model from {self.model_path}")

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
            self.online_template = template
            self.online_max_template = template
            self.max_pred_score = -1.0

            return True
        except Exception as e:
            print(f"[MixFormerTracker] Init failed: {e}")
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

            # Run inference
            with torch.no_grad():
                out_dict = self.network(
                    self.template, self.online_template, search, softmax=True
                )

            # Get predictions
            pred_boxes = out_dict['pred_boxes'].view(-1, 4)
            # Score is raw logit - apply sigmoid to get probability
            raw_score = out_dict.get('pred_scores', torch.tensor([0.8]))
            if 'pred_scores' in out_dict:
                pred_score = torch.sigmoid(raw_score).item()
            else:
                pred_score = 0.8

            # Average box predictions and map back
            pred_box = (pred_boxes.mean(dim=0) * self.search_size / resize_factor).tolist()
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
                self.online_template = self.online_max_template
                self.max_pred_score = -1
                self.online_max_template = self.template

            return True, self.state, pred_score

        except Exception as e:
            print(f"[MixFormerTracker] Update failed: {e}")
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


def create_tracker(model_path: Optional[str] = None, device: str = 'cuda') -> MixFormerTracker:
    """Factory function to create a MixFormerTracker instance.

    Args:
        model_path: Path to model checkpoint
        device: Device to run on

    Returns:
        MixFormerTracker instance
    """
    return MixFormerTracker(model_path=model_path, device=device)


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
    tracker = MixFormerTracker()

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

        cv2.imshow('MixFormer Tracker', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()
