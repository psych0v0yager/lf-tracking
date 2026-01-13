import cv2
import argparse
import sys
import time

def parse_bbox(bbox_str):
    """Parses a string 'x,y,w,h' into a tuple of integers."""
    try:
        return tuple(map(int, bbox_str.split(',')))
    except ValueError:
        print("Error: BBox must be in format 'x,y,w,h'")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Real-Time Learning-Free Drone Tracker")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--bbox", type=str, required=True, help="Initial bbox as 'x,y,w,h'")
    parser.add_argument("--output", type=str, default="output.mp4", help="Path to output video file")
    
    args = parser.parse_args()

    # 1. Initialize Video Capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Error: Could not open video {args.video}")
        sys.exit(1)

    # Read the first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame")
        sys.exit(1)

    # 2. Initialize the CSRT Tracker
    # CSRT is accurate and handles scale/rotation well, fitting the "Learning-Free" constraint
    try:
        # For newer OpenCV versions
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
        # Fallback for older or specific contrib builds
        print("Error: TrackerCSRT not found. Ensure opencv-contrib-python is installed if using older versions.")
        sys.exit(1)

    initial_bbox = parse_bbox(args.bbox)
    tracker.init(frame, initial_bbox)

    # 3. Setup Video Writer
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # MacOS/Linux friendly
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0

    print(f"Tracking started... (Input: {width}x{height} @ {fps} FPS)")

    while True:
        # Start timer for performance monitoring
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        # 4. Update Tracker
        # This is the core step: predicting the new box based on previous visual info
        success, box = tracker.update(frame)
        
        # Calculate processing time
        process_time = (time.time() - start_time) * 1000 # ms
        total_time += process_time
        frame_count += 1

        # Draw bounding box
        if success:
            x, y, w, h = [int(v) for v in box]
            # Draw green rectangle for tracked object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Drone", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            # Tracking failure alert
            cv2.putText(frame, "Tracking Failure", (20, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Performance overlay (MS per frame)
        cv2.putText(frame, f"Inference: {int(process_time)}ms", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        out.write(frame)

        # Optional: Display output (comment out for headless server run)
        # cv2.imshow("Drone Tracker", frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    avg_time = total_time / frame_count if frame_count > 0 else 0
    print(f"Done. Processed {frame_count} frames.")
    print(f"Average time per frame: {avg_time:.2f}ms ({1000/avg_time:.2f} FPS)")
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
