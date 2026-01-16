import os
import cv2
import time
import math
import torch
from ultralytics import YOLO

# ================= CONFIG =================
VIDEO_PATH = "rtsp://admin:admin@123@137.97.110.166:8554/main"   # replace with your video path or RTSP stream

SAVE_DIR = "/home/safepro/Desktop/final_output"
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = "yolov8s.pt"
CONF_THRES = 0.3

MATCH_DIST = 120
MAX_MISSED = 8

# -------- SPEED LINE SETTINGS (FINAL) --------
LINE_Y1 = 600          # moved DOWN
LINE_Y2 = 1200          # moved DOWN & farther
REAL_DISTANCE_M = 8.0  # meters between lines (MEASURE ON ROAD)

KMH = 3.6
MAX_SPEED = 120

DISPLAY_W = 1280
DISPLAY_H = 720

# ================= LOAD MODEL =================
print("Using device:", "GPU" if torch.cuda.is_available() else "CPU")
model = YOLO(MODEL_PATH)

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

# ================= TRACK STORAGE =================
tracks = {}
next_id = 1

# ================= HELPERS =================
def center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# -------- STRICT YOLO CLASS MAP (NO FALLBACK) --------
YOLO_CLASS_MAP = {
    1: "2W",   # bicycle
    3: "2W",   # motorcycle
    2: "4W",   # car
    5: "6W",   # bus
    7: "6W"    # truck
}

# ================= MAIN LOOP =================
print("Running FINAL SPEED + CLASSIFICATION")
print("Saving to:", SAVE_DIR)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=CONF_THRES, verbose=False)[0]
    detections = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls = YOLO_CLASS_MAP.get(cls_id)

        # Ignore unsupported classes (person, etc.)
        if cls is None:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        detections.append((x1, y1, x2, y2, cls))

    updated = set()

    for det in detections:
        x1, y1, x2, y2, cls = det
        cx, cy = center((x1, y1, x2, y2))

        matched_id = None
        min_d = 1e9

        for tid, t in tracks.items():
            d = dist((cx, cy), t["center"])
            if d < MATCH_DIST and d < min_d:
                matched_id = tid
                min_d = d

        # -------- NEW TRACK --------
        if matched_id is None:
            tracks[next_id] = {
                "center": (cx, cy),
                "last_y": cy,
                "t1": None,
                "t2": None,
                "speed": None,
                "saved": False,
                "missed": 0,
                "cls": cls
            }
            updated.add(next_id)
            next_id += 1
            continue

        # -------- UPDATE TRACK --------
        t = tracks[matched_id]
        t["center"] = (cx, cy)
        t["missed"] = 0
        updated.add(matched_id)

        # -------- LINE CROSSING --------
        if t["last_y"] < LINE_Y1 <= cy and t["t1"] is None:
            t["t1"] = time.time()
        if t["last_y"] < LINE_Y2 <= cy and t["t2"] is None:
            t["t2"] = time.time()
        t["last_y"] = cy

        if t["t1"] and t["t2"] and t["speed"] is None:
            dt = t["t2"] - t["t1"]
            if dt > 0:
                sp = (REAL_DISTANCE_M / dt) * KMH
                if sp < MAX_SPEED:
                    t["speed"] = int(sp)

        # -------- DRAW BBOX + SPEED --------
        label = f"{t['cls']} ID{matched_id}"
        if t["speed"]:
            label += f" {t['speed']} km/h"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # -------- SAVE IMAGE WITH TEXT (ONCE) --------
        if t["speed"] and not t["saved"]:
            folder = os.path.join(SAVE_DIR, t["cls"])
            os.makedirs(folder, exist_ok=True)

            path = os.path.join(
                folder,
                f"{t['cls']}_ID{matched_id}_{t['speed']}kmh.jpg"
            )

            cv2.imwrite(path, frame)
            print("[SAVED]", path)
            t["saved"] = True

    # -------- CLEAN OLD TRACKS --------
    for tid in list(tracks.keys()):
        if tid not in updated:
            tracks[tid]["missed"] += 1
            if tracks[tid]["missed"] > MAX_MISSED:
                del tracks[tid]

    # -------- DRAW SPEED LINES --------
    cv2.line(frame, (0, LINE_Y1), (frame.shape[1], LINE_Y1), (255, 0, 0), 2)
    cv2.line(frame, (0, LINE_Y2), (frame.shape[1], LINE_Y2), (255, 0, 0), 2)

    # -------- DISPLAY --------
    disp = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    cv2.imshow("FINAL SPEED + CLASSIFICATION", disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
