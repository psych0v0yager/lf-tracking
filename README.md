# Drone Tracking System

Real-time single-object tracking for a tiny drone (~73x46 pixels) in video footage. Achieves **100% accuracy (717/717 frames)** at **125.8 FPS**.

## Results

| Tracker | Accuracy | FPS | Use Case |
|---------|----------|-----|----------|
| **MixFormerV2 ONNX** | 717/717 (100%) | 125.8 | Production - fastest (ONNX Runtime) |
| **MixFormerV2** | 717/717 (100%) | 83.5 | Development - PyTorch GPU |
| **CSRT** | 717/717 (100%) | 44 | Fallback - no GPU needed |

## Quick Start

```bash
# Clone with submodules
git clone --recursive https://github.com/psych0v0yager/lf-tracking.git
cd lf-tracking

# Install dependencies
uv sync

# Run MixFormerV2 tracker (recommended)
# Usage: python tracker_mixformer.py <video> <x> <y> <width> <height>
uv run python tracker_mixformer.py test_track_dron1.mp4 1314 623 73 46

# Run CSRT tracker (no GPU required)
# Usage: python tracker_csrt.py <video> <x> <y> <width> <height>
uv run python tracker_csrt.py test_track_dron1.mp4 1314 623 73 46

# Run ONNX-accelerated tracker (125.8 FPS - fastest)
# First export the ONNX model:
uv run python scripts/export_onnx.py
# Install cuDNN for GPU acceleration:
uv add nvidia-cudnn-cu12
# Set library path and run:
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__.replace('__init__.py', 'lib'))"):$LD_LIBRARY_PATH
uv run python tracker_mixformer_onnx.py test_track_dron1.mp4 1314 623 73 46

# Production mode (no display/output for max speed)
uv run python tracker_mixformer.py test_track_dron1.mp4 1314 623 73 46 --no-save --no-show
```

Arguments: `<video_path> <x> <y> <width> <height>` where (x, y, w, h) is the initial bounding box.

## Challenges Solved

The test video presents several challenges:

1. **Sharp camera pans** - Motion compensation handles camera movement
2. **Tiny target** - 73x46 pixels in 1920x1080 frame (0.2% of frame)
3. **Drone accelerating** - Adaptive template updating prevents drift
4. **Dark backgrounds** - Edge-based verification (MixFormer) maintains tracking

## Architecture

### MixFormerV2 Tracker (`tracker_mixformer.py`)

```
Frame -> Motion Compensation (1.3ms)
      -> MixFormerV2-S Neural Tracker (7.4ms)
      -> Edge-Enhanced Verification (0.44ms)
      -> Light Recovery Search (9ms when needed)
```

- Transformer-based tracker (NeurIPS 2023)
- Edge-based verification for dark backgrounds
- 83.5 FPS on RTX 3090 Ti

### MixFormerV2 ONNX Tracker (`tracker_mixformer_onnx.py`)

```
Frame -> Motion Compensation (1.5ms)
      -> MixFormerV2-S ONNX Runtime (4.6ms)
      -> Edge-Enhanced Verification (0.42ms)
      -> Light Recovery Search (6ms when needed)
```

- Same architecture as PyTorch version but ~1.5x faster
- ONNX Runtime with CUDA execution provider
- 125.8 FPS on RTX 3090 Ti (vs 83.5 FPS PyTorch)
- Requires cuDNN 9+ (install via `uv add nvidia-cudnn-cu12`)

### CSRT Tracker (`tracker_csrt.py`)

```
Frame -> Motion Compensation (1.7ms)
      -> CSRT Correlation Filter (17ms)
      -> Template Verification
      -> Pyramidal Recovery Search (37ms when needed)
```

- Classical correlation filter approach
- Adaptive template updating
- 44 FPS, no GPU required

## Project Structure

```
lf-tracking/
├── tracker_mixformer.py       # Main tracker (MixFormerV2 PyTorch)
├── tracker_mixformer_onnx.py  # ONNX-accelerated tracker
├── tracker_csrt.py            # Fallback tracker (CSRT)
├── scripts/
│   └── export_onnx.py         # PyTorch -> ONNX model export
├── trackers/
│   ├── mixformer_pytorch.py   # PyTorch MixFormer wrapper
│   └── mixformer_onnx.py      # ONNX Runtime wrapper
├── models/
│   └── mixformerv2/
│       ├── models/            # PyTorch weights (.pth)
│       └── onnx/              # ONNX models (.onnx)
├── MixFormerV2/               # Neural tracker library (submodule)
├── MixformerV2-onnx/          # ONNX reference (submodule)
├── CLAUDE.md                  # Detailed technical documentation
└── archive/                   # Development iterations (36 versions)
```

## Technical Details

See [CLAUDE.md](CLAUDE.md) for:
- Complete version evolution (v3 through v18v3)
- Performance analysis and timing breakdowns
- Parameter tuning details
- The "Frame 600 Problem" and how it was solved

## Hardware

- **Development**: Threadripper + RTX 3090 Ti
- **Target**: Jetson Thor (30 FPS target exceeded)

## Constraint

Models cannot be finetuned on drone detection - must use pretrained models or classical computer vision techniques.
