import cv2
import sys

# Global variables for mouse drawing
drawing = False
ix, iy = -1, -1
bbox = None
frame_copy = None

def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, bbox, frame_copy, frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Calculate bbox as (x, y, w, h)
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        w, h = x2 - x1, y2 - y1
        bbox = (x1, y1, w, h)
        
        frame_copy = frame.copy()
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        print(f"\nBounding box: --bbox {x1},{y1},{w},{h}")
        print(f"  (x, y, w, h) = {bbox}")
        print(f"  (x1, y1, x2, y2) = ({x1}, {y1}, {x2}, {y2})")

# Video path
video_path = sys.argv[1] if len(sys.argv) > 1 else "test_track_dron1.mp4"

# Capture first frame
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print(f"Error: Could not read video '{video_path}'")
    sys.exit(1)

frame_copy = frame.copy()

# Setup window and mouse callback
cv2.namedWindow("Draw Bounding Box")
cv2.setMouseCallback("Draw Bounding Box", draw_rectangle)

print("=" * 50)
print("INSTRUCTIONS:")
print("  - Click and drag to draw bounding box around drone")
print("  - Press 'r' to reset")
print("  - Press 'q' or Enter to quit and save")
print("=" * 50)

while True:
    cv2.imshow("Draw Bounding Box", frame_copy)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('r'):
        # Reset
        frame_copy = frame.copy()
        bbox = None
        print("Reset - draw again")
        
    elif key == ord('q') or key == 13:  # q or Enter
        break

cv2.destroyAllWindows()

if bbox:
    print("\n" + "=" * 50)
    print("FINAL BOUNDING BOX:")
    print(f"  --bbox {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
    print("=" * 50)
else:
    print("No bounding box selected")