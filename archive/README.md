# Archive - Iteration History

This directory contains the development iterations that led to the final production trackers.

## Why This Exists

Tracking a tiny drone (~73x46 pixels) through camera pans and fast acceleration required systematic experimentation. Each version represents a different approach to solving specific challenges.

## Directory Structure

```
archive/
├── iterations/          # 36 tracker versions (v3 through v18v2)
├── modules/             # Supporting modules not used in final trackers
│   ├── fusion/          # Multi-signal fusion experiments
│   └── recovery/        # Async YOLO detection recovery
└── experimental/        # Alternative pipeline approaches
    └── main_robust.py   # EKF + phase detection pipeline
```

## Key Milestones

| Version | Milestone | Result |
|---------|-----------|--------|
| v3 | Baseline motion-compensated CSRT | ~580/717 frames |
| v16 | **First 100% accuracy** - Adaptive template updating | 717/717, 8.6 FPS |
| v17v9 | **Optimized CSRT** - Pyramidal recovery, grid sampling | 717/717, 44 FPS |
| v18v3 | **Neural tracker** - MixFormerV2 + edge verification | 717/717, 83.5 FPS |

## The Frame 600 Problem

The critical challenge was frames ~550-650 where the drone accelerates beyond camera speed:
- Motion compensation overshoots (predicts background motion, not drone motion)
- CSRT drifts to background features
- Recovery fails if template hasn't adapted to appearance changes

**Solution (v16)**: Adaptive template updating - slowly blend tracker template when confident, allowing recovery to find the drone even after significant appearance changes.

## Production Trackers

The final trackers are in the parent directory:
- `tracker_mixformer.py` - MixFormerV2 neural tracker (83.5 FPS)
- `tracker_csrt.py` - CSRT classical tracker (44 FPS)

See `CLAUDE.md` in the parent directory for detailed technical documentation.
