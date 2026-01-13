#!/usr/bin/env python3
"""
Export MixFormerV2-S to ONNX format.

This script loads the PyTorch MixFormerV2-S model and exports it to ONNX
for accelerated inference with ONNX Runtime.

Usage:
    uv run python scripts/export_onnx.py

Output:
    models/mixformerv2/onnx/mixformer_v2_s.onnx
"""

import os
import sys
import torch
import torch.nn as nn

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MIXFORMER_DIR = os.path.join(PROJECT_DIR, 'MixFormerV2')

if MIXFORMER_DIR not in sys.path:
    sys.path.insert(0, MIXFORMER_DIR)


class MixFormerONNXWrapper(nn.Module):
    """Wrapper for MixFormerV2-S that has clean ONNX-friendly interface.

    The original model has a complex forward() signature with keyword args.
    This wrapper provides fixed positional inputs for ONNX export.
    """

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, template, online_template, search):
        """Forward pass with ONNX-friendly signature.

        Args:
            template: Static template (B, 3, 112, 112)
            online_template: Online/adaptive template (B, 3, 112, 112)
            search: Search region (B, 3, 224, 224)

        Returns:
            pred_boxes: Predicted bounding boxes (B, 1, 4)
            pred_scores: Confidence scores (B, 1)
        """
        # Call the network with softmax=True
        out_dict = self.network(template, online_template, search, softmax=True)

        # Get predictions
        pred_boxes = out_dict['pred_boxes']  # (B, 1, 4)

        # Handle score - may be per-token or averaged
        if 'pred_scores' in out_dict:
            # Apply sigmoid to get probability
            pred_scores = torch.sigmoid(out_dict['pred_scores'])
            # If multi-dimensional, take mean
            if pred_scores.dim() > 1:
                pred_scores = pred_scores.mean(dim=-1, keepdim=True)
        else:
            pred_scores = torch.tensor([[0.8]])

        return pred_boxes, pred_scores


def build_model(model_path: str, device: str = 'cuda'):
    """Build MixFormerV2-S model from checkpoint."""
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
    cfg.DATA.SEARCH.SIZE = 224
    cfg.DATA.TEMPLATE = EasyDict()
    cfg.DATA.TEMPLATE.SIZE = 112
    cfg.DATA.TEMPLATE.NUMBER = 2
    cfg.DATA.MAX_SAMPLE_INTERVAL = [200]

    cfg.TEST = EasyDict()
    cfg.TEST.SEARCH_FACTOR = 4.5
    cfg.TEST.SEARCH_SIZE = 224
    cfg.TEST.TEMPLATE_FACTOR = 2.0
    cfg.TEST.TEMPLATE_SIZE = 112
    cfg.TEST.UPDATE_INTERVALS = EasyDict()
    cfg.TEST.UPDATE_INTERVALS.DEFAULT = [200]
    cfg.TEST.ONLINE_SIZES = EasyDict()
    cfg.TEST.ONLINE_SIZES.DEFAULT = [1]

    # Build model
    network = build_mixformer2_vit_online(cfg, train=False)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

    # Handle different checkpoint formats
    if 'net' in checkpoint:
        state_dict = checkpoint['net']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Load state dict
    missing_keys, unexpected_keys = network.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"[Export] Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"[Export] Unexpected keys: {len(unexpected_keys)}")

    network = network.to(device)
    network.eval()

    return network


def export_to_onnx(model_path: str, output_path: str, device: str = 'cuda'):
    """Export MixFormerV2-S to ONNX format.

    Args:
        model_path: Path to PyTorch checkpoint
        output_path: Path for output ONNX file
        device: Device to run export on
    """
    print(f"[Export] Loading model from {model_path}")
    network = build_model(model_path, device)

    # Wrap for ONNX export
    wrapper = MixFormerONNXWrapper(network)
    wrapper.eval()

    # Create dummy inputs
    template = torch.randn(1, 3, 112, 112, device=device)
    online_template = torch.randn(1, 3, 112, 112, device=device)
    search = torch.randn(1, 3, 224, 224, device=device)

    # Test forward pass
    print("[Export] Testing forward pass...")
    with torch.no_grad():
        pred_boxes, pred_scores = wrapper(template, online_template, search)
        print(f"  pred_boxes shape: {pred_boxes.shape}")
        print(f"  pred_scores shape: {pred_scores.shape}")

    # Export to ONNX
    print(f"[Export] Exporting to {output_path}")

    torch.onnx.export(
        wrapper,
        (template, online_template, search),
        output_path,
        input_names=['img_t', 'img_ot', 'img_search'],
        output_names=['pred_boxes', 'pred_scores'],
        opset_version=18,  # Use 18 for better compatibility
        do_constant_folding=True,
        dynamic_axes=None,  # Fixed batch size of 1
        verbose=False,
        export_params=True  # Ensure weights are exported
    )

    print(f"[Export] Successfully exported to {output_path}")

    # Verify the exported model
    print("[Export] Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("[Export] ONNX model validation passed!")

    # Print model size
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[Export] Model size: {file_size_mb:.2f} MB")

    return output_path


def main():
    """Main export function."""
    # Paths
    model_path = os.path.join(PROJECT_DIR, 'models', 'mixformerv2', 'models', 'mixformerv2_small_clean.pth')
    output_path = os.path.join(PROJECT_DIR, 'models', 'mixformerv2', 'onnx', 'mixformer_v2_s.onnx')

    if not os.path.exists(model_path):
        print(f"[Export] Error: Model not found at {model_path}")
        sys.exit(1)

    # Check CUDA availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Export] Using device: {device}")

    # Export
    export_to_onnx(model_path, output_path, device)

    print("\n[Export] Done! You can now use the ONNX model with tracker_mixformer_onnx.py")


if __name__ == '__main__':
    main()
