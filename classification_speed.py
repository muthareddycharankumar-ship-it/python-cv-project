import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import torch
import os

# ================= CONFIG =================
VIDEO_SOURCE = "rtsp://admin:admin@123@137.97.110.166:8554/main"   # CHANGE THIS
MODEL_PATH = "yolov8s.pt"

CONF_THRES = 0.35
REAL_DISTANCE_M = 5.0        # meters between lines
MAX_SPEED = 180              # km/h sanity limit

LINE_Y1 = 250
LINE_Y2 = 700

DISPLAY_W = 1280
DISPLAY_H = 720

SAVE_DIR = "/home/safepro/Desktop/opencv/outputs"

VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ================= GPU =================
assert torch.cuda.is_available(), " CUDA NOT AVAILABLE"
device = "cuda"
print(" Using GPU")

# ================= MODEL (SILENT) =================
model = YOLO(MODEL_PATH)
model.to(device)
model.overrides["verbose"] = False   #  disable YOLO speed logs

# ================= TRACKER =================
tracker = sv.ByteTrack()

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(" Cannot open video source")
    exit()

FPS = cap.get(cv2.CAP_PROP_FPS)
print(f" FPS: {FPS}")

# ================= MEMORY =================
prev_y = {}            # previous Y center
line1_frame = {}       # frame index at line 1
speed_done = set()     # ensure ONE speed per vehicle
speeds = {}

os.makedirs(SAVE_DIR, exist_ok=True)

frame_count = 0

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    original = frame.copy()

    #  ALWAYS DRAW LINES (NEVER DISAPPEAR)
    cv2.line(original, (0, LINE_Y1),
             (original.shape[1], LINE_Y1), (0, 255, 255), 2)
    cv2.line(original, (0, LINE_Y2),
             (original.shape[1], LINE_Y2), (0, 255, 255), 2)

    # -------- YOLO DETECTION --------
    results = model(frame, conf=CONF_THRES, device=device)[0]
    detections = sv.Detections.from_ultralytics(results)

    if len(detections) > 0:
        # Filter vehicles only
        mask = np.array([cls in VEHICLE_CLASSES for cls in detections.class_id])
        detections = detections[mask]

        # Tracking
        detections = tracker.update_with_detections(detections)

        for i in range(len(detections)):
            x1, y1, x2, y2 = map(int, detections.xyxy[i])
            track_id = int(detections.tracker_id[i])
            class_id = int(detections.class_id[i])
            class_name = VEHICLE_CLASSES[class_id]

            cy = int((y1 + y2) / 2)

            if track_id not in prev_y:
                prev_y[track_id] = cy
                continue

            # ---- LINE 1 CROSS ----
            if track_id not in speed_done:
                if prev_y[track_id] < LINE_Y1 and cy >= LINE_Y1:
                    line1_frame[track_id] = frame_count

            # ---- LINE 2 CROSS (ONLY ONCE) ----
            if track_id in line1_frame and track_id not in speed_done:
                if prev_y[track_id] < LINE_Y2 and cy >= LINE_Y2:
                    f1 = line1_frame[track_id]
                    f2 = frame_count

                    time_sec = (f2 - f1) / FPS
                    if time_sec > 0:
                        speed = (REAL_DISTANCE_M / time_sec) * 3.6
                        if 1 < speed < MAX_SPEED:
                            speed = round(speed, 2)
                            speeds[track_id] = speed
                            speed_done.add(track_id)

                            #  PRINT ONLY ONCE
                            print(f"[SPEED] {class_name} | ID {track_id} | {speed} km/h")

                            # SAVE IMAGE ONCE
                            class_dir = os.path.join(SAVE_DIR, class_name)
                            os.makedirs(class_dir, exist_ok=True)

                            crop = original[y1:y2, x1:x2]
                            if crop.size > 0:
                                path = os.path.join(
                                    class_dir,
                                    f"{class_name}_ID{track_id}_{speed}kmh.jpg"
                                )
                                cv2.imwrite(path, crop)

                    del line1_frame[track_id]

            prev_y[track_id] = cy

            # ---- DRAW BOX & LABEL ----
            label = f"{class_name} | ID {track_id}"
            if track_id in speeds:
                label += f" | {speeds[track_id]} km/h"

            cv2.rectangle(original, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                original,
                label,
                (x1, y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

    # -------- DISPLAY --------
    display = cv2.resize(original, (DISPLAY_W, DISPLAY_H))
    cv2.imshow("FINAL VEHICLE SPEED SYSTEM", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
