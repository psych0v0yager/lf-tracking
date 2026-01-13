import cv2

# Initialize tracker (CSRT is good for accuracy, KCF for speed)
tracker = cv2.TrackerCSRT_create()
# or: tracker = cv2.TrackerKCF_create()

cap = cv2.VideoCapture("test_track_dron1.mp4")
ret, frame = cap.read()

# Your manually drawn bbox (x, y, w, h)
bbox = (1314,623,73,46)  # replace with your values
tracker.init(frame, bbox)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    success, bbox = tracker.update(frame)
    
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()