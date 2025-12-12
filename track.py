import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort  # make sure sort.py is in the same folder

# Initialize YOLOv11
model = YOLO("yolo11n.pt")

# Initialize SORT tracker
tracker = Sort(max_age=5, min_hits=1, iou_threshold=0.3)

# Open input video
cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.0.190:554/main")
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Prepare output video
out = cv2.VideoWriter("output_tracked.mp4",
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv11 detection
    results = model(frame)
    detections = []

    # Convert YOLO results to SORT format [x1, y1, x2, y2, score]
    for result in results:
        for box in result.boxes:
            conf = box.conf[0].item()
            if conf < 0.3:
                continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append([x1, y1, x2, y2, conf])

    detections = np.array(detections)

    # Update SORT tracker
    tracked_objects = tracker.update(detections)

    # Draw tracked objects
    for x1, y1, x2, y2, track_id in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    out.write(frame)
    cv2.imshow("YOLOv11 + SORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
