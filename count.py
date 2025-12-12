import cv2
import torch
from ultralytics import YOLO
from sort import Sort

# --- Configuration ---
VIDEO_PATH = "rtsp://admin:admin123@192.168.0.190:554/main"
OUTPUT_PATH = "output_video.mp4"
YOLO_MODEL = "yolo11n.pt"
IMG_SIZE = 416
CONF_THRESHOLD = 0.3

# --- Load YOLO ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model = YOLO(YOLO_MODEL).to(device)

# --- SORT Tracker ---
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# --- Video Setup ---
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
line_y = int(height * 0.8)  # middle horizontal line

# --- Video Writer ---
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# --- Tracking and Counting ---
track_history = {}     # store previous Y positions
counted_ids = set()    # store IDs already counted
total_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- YOLO Detection ---
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, device=device, verbose=False)[0]

    detections = []
    for box, conf, cls in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls):
        if int(cls) != 0:  # Only 'person' class
            continue
        x1, y1, x2, y2 = map(float, box)
        detections.append([x1, y1, x2, y2, float(conf)])

    # --- SORT Tracking ---
    track_bbs_ids = tracker.update(torch.tensor(detections)) if len(detections) else []

    # --- Draw the counting line ---
    cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)

    for x1, y1, x2, y2, track_id in track_bbs_ids:
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Draw person bounding box and center
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # --- Track history ---
        previous_y = track_history.get(track_id, None)
        track_history[track_id] = cy

        # --- One-way count logic (Top → Bottom only) ---
        if previous_y is not None:
            if previous_y < line_y and cy >= line_y and track_id not in counted_ids:
                total_count += 1
                counted_ids.add(track_id)

    # --- Display total count ---
    cv2.putText(frame, f"Count: {total_count}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow("One-Way Person Counting", frame)
    # out.write(frame)  # uncomment if you want to save output

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Output saved to {OUTPUT_PATH}")
