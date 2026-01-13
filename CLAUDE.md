# Drone Tracking Project - Claude Documentation

## Project Overview

Real-time tracking system for a tiny black drone in video footage. The system must handle:
- Sharp camera pans
- Tiny object (~73x46 pixels in 1920x1080 frame)
- Stationary drone periods
- Drone accelerating faster than camera can follow

**Constraint**: Cannot finetune models on drone detection - must use pretrained models or classical CV.

---

## Test Data

**Video**: `test_track_dron1.mp4`
- Resolution: 1920x1080
- FPS: ~24
- Duration: ~30 seconds (717 frames)

**Initial Bounding Box**: `(1314, 623, 73, 46)` - (x, y, width, height)

---

## Key Files

### Primary Trackers (Use These)

#### `tracker_mixformer.py` - BEST PERFORMER (717/717 frames, 83.5 FPS) 🏆
MixFormerV2-S neural tracker with edge-based verification and light recovery. **Tracks the entire video at 83.5 FPS - nearly 2x faster than CSRT-based tracker.**

**Usage:**
```bash
uv run python tracker_mixformer.py test_track_dron1.mp4 1314 623 73 46

# Production mode (faster - no video output or display):
uv run python tracker_mixformer.py test_track_dron1.mp4 1314 623 73 46 --no-save --no-show
```

**Key Features:**
- **MixFormerV2-S neural tracker** - Transformer-based, 300+ FPS capable
- **Edge-based verification** - Robust to dark backgrounds using Canny edge matching
- **Light recovery search** - 9ms avg (4x faster than pyramidal)
- **Adaptive size constraints** - Tighter bbox limits when confidence is low
- Motion compensation from v17 series (handles camera pans)

**Why It Works:**
1. **MixFormerV2-S** - SOTA transformer tracker (NeurIPS 2023), handles appearance changes natively
2. **Edge-based verification** - When drone passes over dark backgrounds (low grayscale contrast), edge matching maintains confidence
3. **Light recovery** - Coarse-only search at 1/3 resolution, single scale. Simpler = more robust + faster
4. **Conservative edge combination** - Edge score only boosts borderline cases (gray 0.15-0.5), preventing false positives

**Architecture:**
```
Frame -> Motion Compensation (1.3ms) -> Camera motion estimate
      -> MixFormerV2-S (7.4ms) -> BBox + Confidence
      -> Edge-enhanced Verification (0.44ms) -> Combined score
      -> If valid: Adaptive size constraints -> Output
      -> If invalid: Light recovery search (9ms) -> Reinit tracker
```

**Performance Comparison:**
| Tracker | Accuracy | FPS | Recovery Time |
|---------|----------|-----|---------------|
| v17v9 (CSRT) | 717/717 | 44 | 37ms |
| v18 (MixFormer) | 709/717 | 83 | 43ms |
| v18v2 (+edge verify) | 715/717 | 81 | 37ms |
| **v18v3 (+light recovery)** | **717/717** | **83.5** | **9ms** |

---

#### `archive/iterations/best_tracker_v18v2.py` - Edge-Enhanced MixFormer (715/717 frames, 81 FPS)
MixFormerV2-S with edge-based verification. Improved dark background handling but slower recovery.

---

#### `archive/iterations/best_tracker_v18.py` - Base MixFormer (709/717 frames, 83 FPS)
Pure MixFormerV2-S integration. Fast but struggles with dark backgrounds.

---

#### `tracker_csrt.py` - Best CSRT Tracker (717/717 frames, 44 FPS)
Optimized motion-compensated CSRT tracker with tuned CSRT parameters, adaptive template updating, and fast pyramidal recovery. **Successfully tracks the entire video at 44 FPS.**

**Usage:**
```bash
uv run python tracker_csrt.py test_track_dron1.mp4 1314 623 73 46

# Production mode (faster - no video output or display):
uv run python tracker_csrt.py test_track_dron1.mp4 1314 623 73 46 --no-save --no-show
```

**Key Features:**
- **Optimized CSRT parameters** - 17ms vs 20.5ms default (17% faster)
- **Optimized motion compensation** - downscaled to 480px with grid sampling (1.7ms vs 60ms)
- Template verification with score drop detection
- **Adaptive template updating** - template evolves as drone appearance changes
- **Pyramidal recovery search** - 12x faster recovery (37ms vs 500ms)
- Dual-template search (adapted + original) for recovery

**CSRT Parameter Tuning (v17v9):**
```python
params.use_segmentation = True    # Essential for accuracy (v17v8 failed without it)
params.number_of_scales = 17      # Default 33 (saves ~1ms)
params.template_size = 150        # Default 200 (saves ~1ms)
params.padding = 2.5              # Default 3.0 (saves ~0.5ms)
params.admm_iterations = 3        # Default 4 (saves ~0.5ms)
params.histogram_bins = 8         # Default 16 (saves ~0.5ms)
```

**Why It Works:**
Four breakthroughs:
1. **Adaptive template updating** - As the drone's appearance changes (motion blur, angle changes), the template slowly adapts (alpha=0.03 blend when score > 0.65). This allows recovery to find the drone even after significant appearance change.
2. **Optimized motion estimation** - Grid sampling on downscaled frames (480px) eliminates `goodFeaturesToTrack` overhead, achieving 40x speedup with no accuracy loss.
3. **Pyramidal recovery** - Coarse search at 1/4 resolution, then refine at full resolution. Reduces recovery from 500ms to 37ms.
4. **CSRT parameter tuning** - Reducing scales, template size, padding, iterations saves 3.4ms per frame while maintaining accuracy.

**Architecture:**
```
Frame -> Motion Compensation -> Predicted Position
      -> CSRT Tracker (tuned params) -> Tracked Position
      -> Template Verification (threshold 0.50)
      -> Score Drop Detection (>0.25 drop triggers recovery)
      -> If valid & score > 0.65: Update template (slow blend)
      -> If invalid: Pyramidal dual-template search (coarse -> refine)
```

---

### Archived Versions (in `archive/iterations/`)

The following versions are preserved for reference in `archive/iterations/`:

- **v17v7** - Previous Best (717/717 frames, 38 FPS) - Same as CSRT tracker but with default parameters
- **v17v2** - Older Version (717/717 frames, 20 FPS) - Slower recovery
- **v16** - First 100% accurate version (717/717, 8.6 FPS) - Solved frame 600 problem
- **v3-v15** - Development iterations leading to v16
- **v17v3-v17v8** - CSRT optimization experiments

### Experimental Pipeline (in `archive/experimental/`)

#### `main_robust.py` - Modular Pipeline
More complex system with EKF, phase detection, and async YOLO support.

**Note**: Underperforms the simpler trackers due to bbox lagging issues.

**Supporting Modules** (in `archive/modules/`):
- `trackers/ekf_tracker.py` - Extended Kalman Filter with acceleration state
- `trackers/phase_detector.py` - Detects stationary/following/accelerating phases
- `fusion/fusion_module.py` - Multi-signal fusion
- `recovery/async_detector.py` - Async YOLO/SAHI detection

---

## Video Phases

The test video has three distinct phases:

1. **Stationary** (~frames 0-200): Drone on ground, minimal motion
2. **Camera Following** (~frames 200-550): Camera pans to track moving drone
3. **Drone Accelerating** (~frames 550-717): Drone moves faster than camera - **this is where tracking fails**

---

## Key Parameters (v16)

| Parameter | Value | Notes |
|-----------|-------|-------|
| Motion threshold | 20px | Triggers tracker reinitialization |
| Verify threshold | 0.50 | CSRT result must score above this |
| Score drop threshold | 0.25 | Drop > this triggers recovery |
| Template update alpha | 0.03 | Slow adaptation when score > 0.65 |
| Template scales | 0.8-1.2 | Multi-scale matching |
| Recovery threshold | 0.55 (high), 0.35 (low) | Full-frame search thresholds |
| Size constraint | 70-130% | Prevents bbox drift |

---

## How Motion Compensation Works

```python
# 1. Find features in background (excluding object region)
prev_pts = cv2.goodFeaturesToTrack(prev_gray, mask=exclude_object)

# 2. Track features with optical flow
curr_pts = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts)

# 3. Compute homography (camera motion model)
H = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC)

# 4. Transform bbox to predicted position
predicted_bbox = cv2.perspectiveTransform(bbox_corners, H)

# 5. If motion > threshold, reinitialize tracker at predicted position
if motion_magnitude > 20:
    tracker.init(frame, predicted_bbox)
```

---

## Known Issues & Future Work

### SOLVED: Frame 600 Problem
**v16 successfully tracks all 717 frames** using adaptive template updating.

### The Core Problem (Historical)
There were TWO failure modes that required OPPOSITE solutions:

1. **Initial Pan Down** (~frame 50): Camera pans, drone is stationary
   - CSRT alone cannot handle this (drifts to background)
   - Motion compensation REQUIRED to predict where drone moved

2. **Frame 600 Camera Catch-up**: Camera catches up to moving drone
   - Motion compensation OVERSHOOTS
   - Drone moves LEFT in world, camera pans LEFT to follow
   - Background moves RIGHT in frame, drone moves RIGHT but LESS
   - Motion comp predicts drone moved fully RIGHT → wrong position

### The Solution
**Adaptive template updating** solved both problems:
- Motion compensation handles the initial pan
- Template verification catches when CSRT drifts to wrong target
- Adaptive template (slow EMA blend when confident) allows recovery to find drone even after appearance changes from motion blur
- Score drop detection triggers recovery before false-locking on background

### Future Improvements
1. **Performance optimization**: Profile and optimize for Jetson Thor deployment
2. **Multi-object tracking**: Extend to track multiple drones
3. **Re-detection**: Add YOLO/detection model trained on drones for cold-start

---

## Experimental Versions (v5-v16)

All attempts to solve the frame 600 problem while keeping the initial pan working:

| Version | Approach | Initial Pan | Frame 600 | Notes |
|---------|----------|-------------|-----------|-------|
| **v3** | Motion comp + CSRT | PASS | FAIL ~580 | Original baseline |
| **v4** | v3 + template recovery | PASS | FAIL ~600 | Template matching didn't help |
| **v5** | Object optical flow | PASS | FAIL ~600 | Velocity arrow flipped direction |
| **v5+** | + Outlier rejection | PASS | FAIL ~600 | Frame jitter still caused issues |
| **v6** | Motion-based detection | PASS | FAIL ~600 | Detect by frame differencing |
| **v7** | Camera acceleration detection | PASS | FAIL ~600 | Ignore velocity during cam accel |
| **v8** | No motion reinit | **FAIL** | ? | CSRT alone can't handle pan |
| **v9** | Adaptive blend | **FAIL** | ? | Blending reduced motion comp |
| **v10** | World velocity correction | **FAIL** | ? | Bug fixed, still failed |
| **v11** | Conditional correction | **FAIL** | ? | Only correct when vel > threshold |
| **v12** | Template verification | PASS | FAIL ~600 | Verify CSRT against template |
| **v13** | v12 + world velocity | **FAIL** | ? | Combination broke it |
| **v14** | Pure CSRT + verify | **FAIL** | ? | No motion comp at all |
| **v15** | Dual-region search | PASS | FAIL ~600 | Search both positions on failure |
| **v16** | **Adaptive template** | **PASS** | **PASS** | **717/717 frames, 8.6 FPS** |
| v17 | Grid sampling + 320px + 2 levels | PASS | FAIL (last 5) | 19.9 FPS, too aggressive |
| v17v2 | Grid sampling + 480px + 3 levels | PASS | PASS | 717/717, 20 FPS |
| v17v3 | Half-resolution CSRT (0.5x) | FAIL | - | Lost drone on rocks |
| v17v4 | KCF tracker | FAIL | - | Slow with many recovery events |
| v17v5 | ViTTrack | PASS | PASS | Inaccurate, many recoveries |
| v17v6 | NanoTrack | PASS | PASS | 26.6 FPS, 16 recovery events |
| v17v7 | Pyramidal recovery + no YOLO | PASS | PASS | 717/717, 38 FPS |
| v17v8 | All CSRT opts (no segmentation) | PASS | FAIL (end) | 67 FPS but loses drone |
| v17v8v2 | v8 + stricter verify/recovery | PASS | FAIL (mid) | Locked on mound ~550-630 |
| v17v8v3 | v8 + strict recovery + distance | PASS | FAIL (end) | Stuck on bush |
| v17v8v4 | v8 + proactive drift detection | PASS | FAIL (end) | Still drifts |
| v17v10 | Aggressive params + segmentation | PASS | FAIL | Worse than v17v9 |
| **v17v9** | **Balanced CSRT opts** | **PASS** | **PASS** | **FINAL BEST - 717/717, 44 FPS** |

### Key Findings

1. **Motion compensation is essential** for handling camera motion (v8, v14 proved this)
2. **Motion comp overshoots at frame 600** because drone moves opposite to camera pan
3. **Template verification catches drift** (v12) but doesn't fix overshoot
4. **World velocity correction** helps in theory but breaks initial pan in practice
5. **The problem is fundamental**: motion comp assumes object moves WITH background
6. **ADAPTIVE TEMPLATE is the solution** (v16): slowly updating template allows recovery to match changed appearance
7. **Score drop detection** catches false matches before they lock on (0.78 → 0.35 = clear drift)
8. **Dual-template search** (adapted + original) provides robust recovery
9. **CSRT is the best tracker for tiny objects** (v17v3-v17v6): KCF, ViTTrack, and NanoTrack all struggle with 73x46px targets
10. **Pyramidal search** (v17v7): coarse-to-fine recovery is 12x faster than full-res search
11. **CSRT parameter tuning** (v17v8-v17v9): `use_segmentation` is essential for robustness; other params (scales, template_size, padding, iterations) can be reduced for 3.4ms savings
12. **No substitute for segmentation** (v17v8v2-v4): Tried stricter verification, faster score drop detection, distance checks, and proactive drift detection - none could compensate for disabling segmentation. The ~8ms segmentation cost is unavoidable for accuracy.
13. **Optimal CSRT params** (v17v9): `use_segmentation=True`, `number_of_scales=17`, `template_size=150`, `padding=2.5`, `admm_iterations=3`, `histogram_bins=8` - pushing harder (v17v10) hurts accuracy

### Frame 600 Analysis (from debug screenshots)

```
Frame 598: Tracking OK, drone on left side
Frame 600: Camera catches up (24.9px motion), velocity arrow points LEFT
Frame 601: Box jumps RIGHT (motion comp overshoot), drone is LEFT of box
Frame 602+: Box stuck on wrong position, drone lost
```

The motion compensation correctly predicts background motion, but the drone was moving LEFT while camera panned LEFT, so drone's frame displacement is LESS than background.

---

## File Organization

```
lf-tracking/
├── tracker_mixformer.py    # 🏆 BEST - MixFormer + edge verify + light recovery (717/717, 83.5 FPS)
├── tracker_csrt.py         # Best CSRT tracker (717/717, 44 FPS)
├── trackers/
│   └── mixformer_pytorch.py    # MixFormerV2-S wrapper class
├── models/
│   └── mixformerv2/
│       └── models/
│           └── mixformerv2_small_clean.pth  # Clean MixFormer weights
├── MixFormerV2/            # Official MixFormerV2 repository (cloned)
├── MixformerV2-onnx/       # ONNX conversion reference (for future work)
├── test_track_dron1.mp4    # Test video
├── CLAUDE.md               # This file
├── core/                   # Motion compensation modules
└── archive/                # Development iterations (36 versions, v3-v18v2)
    ├── iterations/         # All versioned tracker files
    ├── modules/            # Unused modules (fusion, recovery)
    └── experimental/       # Alternative approaches (main_robust.py)
```

---

## Quick Start

```bash
# BEST - MixFormerV2 neural tracker (717/717 frames, 83.5 FPS)
uv run python tracker_mixformer.py test_track_dron1.mp4 1314 623 73 46

# Production mode (no video output or display window)
uv run python tracker_mixformer.py test_track_dron1.mp4 1314 623 73 46 --no-save --no-show

# CSRT tracker - no GPU required (717/717 frames, 44 FPS)
uv run python tracker_csrt.py test_track_dron1.mp4 1314 623 73 46
```

---

## Target Hardware

- Development: Threadripper + RTX 3090 Ti
- Deployment: Jetson Thor
- Target FPS: 30fps - **ACHIEVED (83.5 FPS with MixFormerV2)**

---

## Performance Analysis

### Current Best: v17v9 (44 FPS)

Performance on development hardware (Threadripper + RTX 3090 Ti) with `--no-save --no-show`:

| Step | Time (ms) | % of Frame |
|------|-----------|------------|
| csrt_update | 17.1 | 75% |
| motion_est | 1.7 | 7% |
| recovery (when needed) | 36.5 | - |
| other | 4.0 | 18% |
| **Total** | **22.8** | **43.9 FPS** |

### Version Comparison

| Version | motion_est | csrt_update | recovery | Total | FPS | Accuracy |
|---------|------------|-------------|----------|-------|-----|----------|
| v16v7 | 60ms | 30ms | ~500ms | 116ms | 8.6 | 717/717 |
| v17 | 0.9ms | 31ms | ~500ms | 50ms | 19.9 | ~712/717 |
| v17v2 | 1.5ms | 31ms | ~500ms | 50ms | 20 | 717/717 |
| v17v7 | 1.2ms | 20.5ms | 39ms | 26ms | 38.3 | 717/717 |
| v17v8 | 1.7ms | 9.2ms | 33ms | 15ms | 67.5 | Loses end |
| **v17v9** | **1.7ms** | **17.1ms** | **36ms** | **23ms** | **43.9** | **717/717** |

### v17v9 Optimization Details

Three major optimizations:

#### 1. CSRT Parameter Tuning (20.5ms → 17.1ms)

```python
params = cv2.TrackerCSRT_Params()
params.use_segmentation = True    # Essential - v17v8 failed without it
params.number_of_scales = 17      # Default 33 (saves ~1ms)
params.template_size = 150        # Default 200 (saves ~1ms)
params.padding = 2.5              # Default 3.0 (saves ~0.5ms)
params.admm_iterations = 3        # Default 4 (saves ~0.5ms)
params.histogram_bins = 8         # Default 16 (saves ~0.5ms)
```

**v17v8 lesson:** Disabling `use_segmentation` gave 67 FPS but lost the drone at the end. Segmentation is essential for robustness during fast motion/appearance changes.

#### 2. Pyramidal Recovery Search (500ms → 37ms)

```python
def search_pyramidal(self, gray, search_region, threshold=0.25):
    # Step 1: Coarse search at 1/4 resolution (fast)
    gray_small = cv2.resize(gray, None, fx=0.25, fy=0.25)
    template_small = cv2.resize(template, None, fx=0.25, fy=0.25)
    coarse_match = self._search_with_template_fast(gray_small, ..., scales=[1.0])

    # Step 2: Refine at full resolution in small region around coarse match
    refine_region = expand_around(coarse_match, factor=2.0)
    final_match = self._search_with_template_fast(gray, refine_region, ...,
                                                   scales=[0.9, 1.0, 1.1])
```

- Fewer scales: `[0.9, 1.0, 1.1]` instead of `[0.8, 0.9, 1.0, 1.1, 1.2]`
- Coarse-to-fine reduces search area dramatically

#### 3. YOLO Removal

YOLO was loaded on startup but never used for detection. Removing it:
- Reduced startup time
- Eliminated unused code complexity
- No accuracy impact (template matching handles all recovery)

### Motion Estimation Optimization (v17v2)

| Parameter | v16v7 (slow) | v17v7 (fast) |
|-----------|--------------|--------------|
| Resolution | Full (1920px) | 480px width |
| Feature detection | `goodFeaturesToTrack` | Fixed grid sampling |
| LK window | 21x21 | 11x11 |
| Pyramid levels | 4 | 3 |
| **Time** | **60ms** | **1.2ms** |

### Tracker Experiments (v17v3-v17v6)

| Tracker | FPS | Recovery Events | Result |
|---------|-----|-----------------|--------|
| CSRT (baseline) | 20 | 5 | BEST accuracy |
| CSRT 0.5x scale | - | - | FAILED - fixated on rocks |
| CSRT 0.75x scale | 18 | 5 | Slower than full res |
| KCF | slow | many | FAILED - inaccurate |
| ViTTrack | slow | many | FAILED - distracted by details |
| NanoTrack | 26.6 | 16 | Faster but less accurate |

**Conclusion:** CSRT remains the best tracker for tiny objects. Alternative trackers (KCF, ViTTrack, NanoTrack) all struggled with the small drone.

### Segmentation-Free CSRT Experiments (v17v8 series)

Disabling `use_segmentation` in CSRT gave 67 FPS (vs 44 FPS), but lost the drone at the end. Multiple attempts to compensate:

| Version | Strategy | Result |
|---------|----------|--------|
| v17v8 | All CSRT opts, no segmentation | 67 FPS, loses drone at end |
| v17v8v2 | Stricter verify (0.55), faster score drop (0.15), tighter bbox | Locked on mound mid-video |
| v17v8v3 | Strict recovery thresholds (0.60), max 300px distance check | Stuck on bush at end |
| v17v8v4 | Proactive drift detection (reject if >60px from prediction) | Still drifts at end |
| v17v10 | Keep segmentation, push other params harder | Worse than v17v9 |

**Why nothing worked:** Without segmentation, CSRT treats all pixels in the bbox equally. During fast motion at the end:
1. Drone appearance changes (motion blur)
2. Background pixels start dominating the correlation filter
3. Tracker gradually drifts to background features
4. By the time drift is detected, template has adapted to include background
5. Recovery finds the background feature as a "good" match

**Conclusion:** The ~8ms segmentation cost is unavoidable. It's what allows CSRT to distinguish foreground (drone) from background pixels. No amount of verification, recovery tuning, or drift detection can substitute for this.

### Production Flags

```bash
--no-save    # Disable video output (saves ~9ms/frame)
--no-show    # Disable display window (saves ~2ms/frame)
```

With both flags: 43.9 FPS effective (v17v9).

### Path Forward

**v17v9 is the final CSRT-optimized version.** 30 FPS target exceeded (44 FPS)!

Current bottleneck breakdown:
1. **csrt_update**: 17.1ms (75%) - tuned to practical limit
2. **motion_est**: 1.7ms (7%) - highly optimized
3. **other**: 4.0ms (18%) - overhead

The ~8ms segmentation cost cannot be removed without losing accuracy (v17v8 series proved this).

---

## MixFormerV2 Neural Tracker (v18 Series) ✅ COMPLETED

**Goal:** 50+ FPS on Jetson Thor, 200+ FPS on desktop
**Result:** 83.5 FPS on RTX 3090 Ti, 717/717 frames (100% accuracy)

### Why MixFormerV2?
- SOTA transformer-based tracker (NeurIPS 2023)
- 300+ FPS on GPU (MixFormerV2-S variant)
- Handles appearance changes better than correlation filters
- ONNX/TensorRT ready for edge deployment

### Final Architecture (v18v3)
```
Frame → Motion Compensation (1.3ms) → Camera motion estimate
      → MixFormerV2-S (7.4ms) → BBox + Confidence (search_factor=3.0)
      → Edge-Enhanced Verification (0.44ms) → Gray + Edge combined score
      → If valid: Adaptive size constraints → Output
      → If invalid: Light recovery (9ms) → Coarse search at 1/3 res
```

### Key Innovations

1. **Reduced search_factor (4.5 → 3.0)** - Prevents MixFormer from locking onto nearby bushes/rocks
2. **Edge-based verification** - Canny edge matching helps when drone passes over dark backgrounds
3. **Conservative edge combination** - Only boost borderline gray scores (0.15-0.5) with good edge scores
4. **Light recovery search** - Coarse-only at 1/3 resolution, 4x faster than full pyramidal
5. **Adaptive size constraints** - Tighter bbox limits (0.9-1.1x) when confidence is low

### v18 Evolution

| Version | Change | Accuracy | FPS | Recovery |
|---------|--------|----------|-----|----------|
| v18 | Base MixFormer | 709/717 | 83 | 43ms |
| v18v2 | +Edge verification | 715/717 | 81 | 37ms |
| **v18v3** | **+Light recovery** | **717/717** | **83.5** | **9ms** |

### Key Files
- `tracker_mixformer.py` - **BEST** - Full tracker with all optimizations
- `trackers/mixformer_pytorch.py` - MixFormerV2-S wrapper class
- `models/mixformerv2/models/mixformerv2_small_clean.pth` - Clean model weights
- `archive/iterations/best_tracker_v18v2.py` - Edge verification, full pyramidal recovery
- `archive/iterations/best_tracker_v18.py` - Base MixFormer integration

### Dark Background Challenge

The drone passing over dark rocks caused confidence to drop to ~0.11. Solutions:
- **Edge-based verification** - Drone silhouette visible even when luminance contrast is zero
- **Light recovery** - Fast enough (9ms) to not cause lag during difficult sections
- **Size constraints** - Prevent bbox from growing when tracker is uncertain

### References
- Official: https://github.com/MCG-NJU/MixFormerV2
- ONNX/TRT: https://github.com/maliangzhibi/MixformerV2-onnx

### Future Work
- ONNX conversion for platform independence
- TensorRT optimization for Jetson Thor deployment
- INT8 quantization for maximum speed
