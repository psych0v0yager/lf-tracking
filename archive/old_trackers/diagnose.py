#!/usr/bin/env python3
"""
Video Diagnostic Tool

Analyzes a video to help find the correct initial bounding box for tracking.

Usage:
    python diagnose.py --video test.mp4
    python diagnose.py --video test.mp4 --frame 0 --save-frame
"""

import cv2
import numpy as np
import argparse


def detect_content_region(frame: np.ndarray, threshold: int = 10):
    """Detect actual content area, excluding letterboxing."""
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


def find_motion_candidates(cap, num_frames=30, min_area=50, max_area=5000):
    """Find moving objects by frame differencing."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    ret, frame = cap.read()
    if not ret:
        return []
    
    content = detect_content_region(frame)
    cx, cy, cw, ch = content
    
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    all_candidates = []
    
    for i in range(num_frames - 1):
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Frame difference
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area < area < max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Check if in content region
                if cx <= x + w//2 < cx + cw and cy <= y + h//2 < cy + ch:
                    all_candidates.append({
                        'frame': i + 1,
                        'bbox': (x, y, w, h),
                        'center': (x + w//2, y + h//2),
                        'area': area
                    })
        
        prev_gray = gray
    
    return all_candidates, content


def cluster_candidates(candidates, grid_size=50):
    """Group candidates by location to find consistent motion."""
    if not candidates:
        return []
    
    centers = np.array([c['center'] for c in candidates])
    
    # Find bounds
    min_x, min_y = centers.min(axis=0)
    max_x, max_y = centers.max(axis=0)
    
    # Grid-based clustering
    clusters = {}
    
    for c in candidates:
        cx, cy = c['center']
        gx = int((cx - min_x) // grid_size)
        gy = int((cy - min_y) // grid_size)
        key = (gx, gy)
        
        if key not in clusters:
            clusters[key] = []
        clusters[key].append(c)
    
    # Sort by count
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    
    return sorted_clusters


def main():
    parser = argparse.ArgumentParser(description="Video Diagnostic Tool")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame number to analyze")
    parser.add_argument("--save-frame", action="store_true", help="Save the analyzed frame")
    parser.add_argument("--analyze-motion", action="store_true", help="Analyze motion to find candidates")
    args = parser.parse_args()
    
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Cannot open video {args.video}")
        return
    
    # Video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0
    
    print("=" * 60)
    print("VIDEO PROPERTIES")
    print("=" * 60)
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Total frames: {total_frames}")
    print(f"Duration: {duration:.2f} seconds")
    print()
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ret, frame = cap.read()
    
    if not ret:
        print(f"Error: Cannot read frame {args.frame}")
        return
    
    # Detect content region
    content = detect_content_region(frame)
    cx, cy, cw, ch = content
    
    print("=" * 60)
    print("CONTENT REGION (excluding letterbox)")
    print("=" * 60)
    print(f"Content starts at: ({cx}, {cy})")
    print(f"Content size: {cw} x {ch}")
    print(f"Content ends at: ({cx + cw}, {cy + ch})")
    print()
    
    # Check provided bbox
    print("=" * 60)
    print("BBOX VALIDATION")
    print("=" * 60)
    test_bbox = (120, 80, 60, 40)  # The original bbox from the task
    bx, by, bw, bh = test_bbox
    bbox_center = (bx + bw//2, by + bh//2)
    
    print(f"Original task bbox: {test_bbox}")
    print(f"Bbox center: {bbox_center}")
    
    if bbox_center[0] < cx or bbox_center[0] >= cx + cw or \
       bbox_center[1] < cy or bbox_center[1] >= cy + ch:
        print("❌ PROBLEM: Bbox center is OUTSIDE the content region!")
        print(f"   The bbox is pointing at the letterbox/black area, not the video content.")
        print(f"   Valid X range: [{cx}, {cx + cw}]")
        print(f"   Valid Y range: [{cy}, {cy + ch}]")
    else:
        print("✓ Bbox is within content region")
    print()
    
    # Analyze motion
    if args.analyze_motion:
        print("=" * 60)
        print("MOTION ANALYSIS")
        print("=" * 60)
        print("Analyzing first 30 frames for moving objects...")
        
        candidates, _ = find_motion_candidates(cap)
        clusters = cluster_candidates(candidates)
        
        print(f"Found {len(candidates)} motion detections")
        print(f"Clustered into {len(clusters)} regions")
        print()
        
        if clusters:
            print("Top motion clusters (potential drone locations):")
            for i, (key, items) in enumerate(clusters[:5]):
                centers = [c['center'] for c in items]
                avg_center = np.mean(centers, axis=0)
                sizes = [(c['bbox'][2], c['bbox'][3]) for c in items]
                avg_size = np.mean(sizes, axis=0)
                
                # Calculate suggested bbox
                suggested_x = int(avg_center[0] - avg_size[0] / 2)
                suggested_y = int(avg_center[1] - avg_size[1] / 2)
                suggested_w = max(20, int(avg_size[0]))
                suggested_h = max(15, int(avg_size[1]))
                
                print(f"\n  Cluster {i+1}: {len(items)} detections")
                print(f"    Average center: ({avg_center[0]:.0f}, {avg_center[1]:.0f})")
                print(f"    Average size: {avg_size[0]:.0f} x {avg_size[1]:.0f}")
                print(f"    SUGGESTED BBOX: --bbox {suggested_x},{suggested_y},{suggested_w},{suggested_h}")
    
    # Save frame with annotations
    if args.save_frame:
        # Draw content region
        cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 2)
        cv2.putText(frame, "Content Region", (cx + 5, cy + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Draw original bbox (if in frame)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 0, 255), 2)
        cv2.putText(frame, "Original bbox (WRONG)", (bx, by - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw grid to help locate drone
        grid_spacing = 100
        for x in range(cx, cx + cw, grid_spacing):
            cv2.line(frame, (x, cy), (x, cy + ch), (50, 50, 50), 1)
            cv2.putText(frame, str(x), (x, cy + ch + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        for y in range(cy, cy + ch, grid_spacing):
            cv2.line(frame, (cx, y), (cx + cw, y), (50, 50, 50), 1)
            cv2.putText(frame, str(y), (cx - 30, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        output_path = f"frame_{args.frame}_annotated.png"
        cv2.imwrite(output_path, frame)
        print()
        print(f"Saved annotated frame to: {output_path}")
    
    cap.release()
    
    print()
    print("=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. Use --analyze-motion to auto-detect drone candidates")
    print("2. Or use interactive mode: python track.py --video VIDEO --interactive")
    print("3. Ensure bbox is within content region:")
    print(f"   X must be in [{cx}, {cx + cw - 1}]")
    print(f"   Y must be in [{cy}, {cy + ch - 1}]")


if __name__ == "__main__":
    main()
