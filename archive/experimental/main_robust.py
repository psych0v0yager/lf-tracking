#!/usr/bin/env python3
"""
Robust Drone Tracking System

Main entry point for the enhanced drone tracking pipeline.

Features:
- Extended Kalman Filter with acceleration modeling
- Phase detection (stationary/following/accelerating)
- Adaptive search regions
- Optional async YOLO/SAHI detection recovery
- Multi-signal fusion

Usage:
    python main_robust.py test_track_dron1.mp4 1314 623 73 46
    python main_robust.py test_track_dron1.mp4 1314 623 73 46 --tracker vit --no-detection
"""

import cv2
import numpy as np
import sys
import argparse
import time
from pathlib import Path
from collections import deque

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from core.pipeline import DroneTrackingPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Robust Drone Tracking System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with CSRT
    python main_robust.py test_track_dron1.mp4 1314 623 73 46

    # Use ViTTrack (if available)
    python main_robust.py test_track_dron1.mp4 1314 623 73 46 --tracker vit

    # Disable detection recovery (faster but less robust)
    python main_robust.py test_track_dron1.mp4 1314 623 73 46 --no-detection

    # Custom output path
    python main_robust.py test_track_dron1.mp4 1314 623 73 46 -o output.mp4
        """
    )

    parser.add_argument("video", help="Path to video file")
    parser.add_argument("x", type=int, help="Initial bbox x")
    parser.add_argument("y", type=int, help="Initial bbox y")
    parser.add_argument("w", type=int, help="Initial bbox width")
    parser.add_argument("h", type=int, help="Initial bbox height")
    parser.add_argument("--tracker", "-t", choices=["csrt", "kcf", "vit"],
                        default="csrt", help="Primary tracker type (default: csrt)")
    parser.add_argument("--no-detection", action="store_true",
                        help="Disable YOLO detection recovery")
    parser.add_argument("--detection-model", help="Path to YOLO model")
    parser.add_argument("--motion-threshold", type=float, default=15.0,
                        help="Camera motion threshold for reinit (default: 15)")
    parser.add_argument("--output", "-o", help="Output video path")
    parser.add_argument("--no-display", action="store_true",
                        help="Don't show live preview")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose debug output")

    args = parser.parse_args()

    # Initial bbox
    bbox = (args.x, args.y, args.w, args.h)

    print("=" * 60)
    print("Robust Drone Tracking System")
    print("=" * 60)
    print(f"OpenCV version: {cv2.__version__}")
    print(f"Video: {args.video}")
    print(f"Initial bbox: {bbox}")
    print(f"Tracker: {args.tracker.upper()}")
    print(f"Detection: {'Enabled' if not args.no_detection else 'Disabled'}")
    print()

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open '{args.video}'")
        sys.exit(1)

    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        sys.exit(1)

    fh, fw = frame.shape[:2]
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 24
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution: {fw}x{fh}")
    print(f"FPS: {video_fps:.2f}")
    print(f"Total frames: {total_frames}")
    print()

    # Validate bbox
    if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > fw or bbox[1] + bbox[3] > fh:
        print(f"Warning: bbox {bbox} may be outside frame {fw}x{fh}")

    # Initialize pipeline
    pipeline = DroneTrackingPipeline(
        tracker_type=args.tracker,
        motion_threshold=args.motion_threshold,
        use_detection=not args.no_detection,
        detection_model=args.detection_model
    )

    if not pipeline.initialize(frame, bbox):
        print("Error: Failed to initialize pipeline")
        sys.exit(1)

    print("Pipeline initialized successfully!")
    print("Press 'q' to quit, SPACE to pause")
    print()

    # Output video
    out_path = args.output or (Path(args.video).stem + "_robust.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(out_path, fourcc, video_fps, (fw, fh))

    # Statistics
    frame_num = 0
    success_count = 0
    phase_counts = {}
    start_time = time.time()

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Process frame
        result = pipeline.update(frame)

        if result['success']:
            success_count += 1

        # Count phases
        phase = result['phase']
        phase_counts[phase] = phase_counts.get(phase, 0) + 1

        # Draw results
        x, y, w, h = result['bbox']

        # Color based on confidence
        conf = result['confidence']
        if conf > 0.7:
            color = (0, 255, 0)  # Green
        elif conf > 0.4:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange

        # Draw bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.circle(frame, (x + w//2, y + h//2), 4, color, -1)

        # Draw EKF prediction (blue)
        ekf_x, ekf_y, ekf_w, ekf_h = result['ekf_prediction']
        cv2.rectangle(frame, (ekf_x, ekf_y), (ekf_x + ekf_w, ekf_y + ekf_h), (255, 0, 0), 1)

        # Info overlay
        status = "TRACK" if result['tracker_success'] else f"LOST({result['lost_count']})"
        det_str = " +DET" if result['detection_available'] else ""

        cv2.putText(frame,
                    f"Frame {frame_num}/{total_frames} | {status}{det_str} | {result['avg_fps']:.0f} fps",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame,
                    f"Phase: {phase} | Conf: {conf:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        vx, vy = result['velocity']
        ax, ay = result['acceleration']
        cv2.putText(frame,
                    f"Vel: ({vx:.1f}, {vy:.1f}) | Acc: ({ax:.1f}, {ay:.1f}) | CamMot: {result['camera_motion']:.1f}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw velocity vector
        cx, cy = x + w//2, y + h//2
        scale = 3  # Scale factor for visualization
        vx_draw, vy_draw = int(vx * scale), int(vy * scale)
        cv2.arrowedLine(frame, (cx, cy), (cx + vx_draw, cy + vy_draw), (0, 255, 255), 2)

        # Write frame
        writer.write(frame)

        # Display
        if not args.no_display:
            cv2.imshow("Robust Drone Tracker", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

        # Verbose output
        if args.verbose and frame_num % 30 == 0:
            print(f"Frame {frame_num}: phase={phase}, conf={conf:.2f}, "
                  f"vel=({vx:.1f},{vy:.1f}), acc=({ax:.1f},{ay:.1f})")

    # Cleanup
    elapsed = time.time() - start_time
    pipeline.shutdown()
    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    # Print summary
    print()
    print("=" * 60)
    print("Results Summary")
    print("=" * 60)
    print(f"Frames processed: {frame_num}")
    print(f"Track success: {success_count}/{frame_num} ({100*success_count/frame_num:.1f}%)")
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average FPS: {frame_num/elapsed:.1f}")
    print()
    print("Phase distribution:")
    for phase, count in sorted(phase_counts.items()):
        pct = 100 * count / frame_num
        print(f"  {phase}: {count} frames ({pct:.1f}%)")
    print()
    print(f"Output saved: {out_path}")


if __name__ == "__main__":
    main()
