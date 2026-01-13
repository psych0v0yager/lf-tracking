#!/usr/bin/env python3
"""Real-time drone tracker using correlation filters."""

import cv2
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--bbox', required=True, help='x,y,w,h')
    parser.add_argument('--tracker', default='KCF', choices=['KCF', 'CSRT', 'MOSSE'])
    args = parser.parse_args()

    # Parse initial bounding box
    x, y, w, h = map(int, args.bbox.split(','))
    bbox = (x, y, w, h)

    # Initialize video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create tracker
    trackers = {
        'KCF': cv2.TrackerKCF_create,
        'CSRT': cv2.TrackerCSRT_create,
        'MOSSE': cv2.legacy.TrackerMOSSE_create
    }
    tracker = trackers[args.tracker]()

    # Read first frame and initialize
    ret, frame = cap.read()
    tracker.init(frame, bbox)

    # Output video
    out = cv2.VideoWriter(
        'output_tracked.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (width, height)
    )

    frame_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start = time.perf_counter()
        success, bbox = tracker.update(frame)
        frame_times.append(time.perf_counter() - start)

        if success:
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Lost", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        out.write(frame)

    avg_ms = sum(frame_times) / len(frame_times) * 1000
    print(f"Average: {avg_ms:.1f}ms/frame ({1000/avg_ms:.1f} FPS)")

    cap.release()
    out.release()

if __name__ == '__main__':
    main()
