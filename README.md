# Drone Tracking System

Real-time single-object tracking for a tiny drone (~73x46 pixels) in video footage. Achieves **100% accuracy (717/717 frames)** at **83.5 FPS**.

## Results

| Tracker | Accuracy | FPS | Use Case |
|---------|----------|-----|----------|
| **MixFormerV2** | 717/717 (100%) | 83.5 | Production - fastest |
| **CSRT** | 717/717 (100%) | 44 | Fallback - no GPU needed |

## Quick Start

```bash
# Install dependencies
uv sync

# Run MixFormerV2 tracker (recommended)
# Usage: python tracker_mixformer.py <video> <x> <y> <width> <height>
uv run python tracker_mixformer.py test_track_dron1.mp4 1314 623 73 46

# Run CSRT tracker (no GPU required)
# Usage: python tracker_csrt.py <video> <x> <y> <width> <height>
uv run python tracker_csrt.py test_track_dron1.mp4 1314 623 73 46

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
├── tracker_mixformer.py     # Main tracker (MixFormerV2)
├── tracker_csrt.py          # Fallback tracker (CSRT)
├── core/                    # Motion compensation modules
├── trackers/                # MixFormer wrapper
├── models/                  # Model weights
├── MixFormerV2/             # Neural tracker library
├── test_track_dron1.mp4     # Test video
├── CLAUDE.md                # Detailed technical documentation
└── archive/                 # Development iterations (36 versions)
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
