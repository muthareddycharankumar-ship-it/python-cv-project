from ultralytics import YOLO
import cv2
import time

# Load ONNX model (CPU only)
model = YOLO("yolov8n.onnx")

cap = cv2.VideoCapture("rtsp://admin:admin123@192.168.0.190:554/main")
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame")
        break

    results = model.predict(frame, imgsz=256, conf=0.3)

    person_count = 0
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0:  # person
                person_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time

    cv2.putText(frame, f"Persons: {person_count}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Person Detection CPU", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
