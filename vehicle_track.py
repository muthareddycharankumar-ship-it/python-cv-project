import cv2
import os
import numpy as np
from ultralytics import YOLO
from sort import Sort
from collections import defaultdict, deque
from statistics import median

# ================= CONFIG =================
MODEL_PATH = "yolov8s.pt"
VIDEO_PATH = ""   # video or RTSP
CONF_THRES = 0.4

DISPLAY_W, DISPLAY_H = 960, 540

SAVE_ROOT = "output"
MIN_FRAMES = 10
MIN_DET_AREA = 2500
# ========================================

CATEGORY_MAP = {
    "2W": "2W_Bicycle",
    "3W": "3W_Auto_Rickshaw",
    "4W": "4W_Car",
    "6W": "6W_Truck"
}

os.makedirs(SAVE_ROOT, exist_ok=True)
for v in CATEGORY_MAP.values():
    os.makedirs(os.path.join(SAVE_ROOT, v), exist_ok=True)

# Load YOLO + SORT
model = YOLO(MODEL_PATH)
tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.5)

cap = cv2.VideoCapture(VIDEO_PATH)

# Track history
area_hist   = defaultdict(list)
aspect_hist = defaultdict(list)
width_hist  = defaultdict(list)
bbox_hist   = defaultdict(lambda: deque(maxlen=5))
yolo_hist   = defaultdict(list)
saved_ids = set()

# ================= IOU =================
def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / (areaA + areaB - inter + 1e-6)

# ================= GEOMETRY =================
def classify_geometry(areas, aspects, widths, frame_area, frame_w):
    A = median(areas)
    R = median(aspects)   # h/w
    W = median(widths)

    na = A / frame_area
    nw = W / frame_w

    # 6W = big + tall
    if na > 0.11 and R > 1.15:
        return "6W"

    # 4W front (must not be small)
    if na > 0.025 and R < 0.85 and nw > 0.20:
        return "4W"

    # 2W = small + narrow + tall
    if na < 0.018 and nw < 0.17 and R > 1.05:
        return "2W"

    # 3W = medium + tall + wide
    if 0.018 <= na < 0.045 and R > 1.2 and nw >= 0.17:
        return "3W"

    return "4W"

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    fh, fw = frame.shape[:2]
    frame_area = fh * fw

    raw = []
    results = model(frame, conf=CONF_THRES, verbose=False)

    # -------- DETECTION --------
    for r in results:
        for box in r.boxes:
            cname = model.names[int(box.cls[0])]
            if cname == "person":
                continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            w,h = x2-x1, y2-y1
            area = w*h
            if area < MIN_DET_AREA:
                continue

            raw.append([x1,y1,x2,y2,float(box.conf[0]),cname])

    # -------- HARD NMS --------
    filtered = []
    for d in sorted(raw, key=lambda x: x[4], reverse=True):
        if all(iou(d, f) < 0.6 for f in filtered):
            filtered.append(d)

    dets = np.array([d[:5] for d in filtered]) if filtered else np.empty((0,5))
    tracks = tracker.update(dets)

    # -------- TRACKING --------
    for t in tracks:
        x1,y1,x2,y2,tid = map(int,t)
        w,h = x2-x1, y2-y1
        area = w*h
        aspect = h/(w+1e-6)

        area_hist[tid].append(area)
        aspect_hist[tid].append(aspect)
        width_hist[tid].append(w)
        bbox_hist[tid].append((x1,y1,x2,y2))

        # YOLO semantic hint (ONLY for 4W vs 6W)
        for d in filtered:
            if iou([x1,y1,x2,y2], d[:4]) > 0.5:
                if d[5] in ["truck","bus"]:
                    yolo_hist[tid].append("6W")
                elif d[5] == "car":
                    yolo_hist[tid].append("4W")

        if len(area_hist[tid]) < MIN_FRAMES:
            continue

        geom = classify_geometry(
            area_hist[tid],
            aspect_hist[tid],
            width_hist[tid],
            frame_area,
            fw
        )

        if yolo_hist[tid].count("6W") >= 3:
            final = "6W"
        elif yolo_hist[tid].count("4W") >= 3:
            final = "4W"
        else:
            final = geom

        folder = CATEGORY_MAP[final]

        bx = bbox_hist[tid]
        sx1 = int(sum(b[0] for b in bx)/len(bx))
        sy1 = int(sum(b[1] for b in bx)/len(bx))
        sx2 = int(sum(b[2] for b in bx)/len(bx))
        sy2 = int(sum(b[3] for b in bx)/len(bx))

        if tid not in saved_ids:
            crop = frame[sy1:sy2, sx1:sx2]
            if crop.size:
                cv2.imwrite(f"{SAVE_ROOT}/{folder}/{folder}_ID{tid}.jpg", crop)
                saved_ids.add(tid)
                print(f"[SAVED] ID {tid} → {folder}")

        cv2.rectangle(frame,(sx1,sy1),(sx2,sy2),(0,255,0),2)
        cv2.putText(frame,f"ID {tid} | {folder}",
                    (sx1,sy1-8),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

    cv2.imshow("Vehicle Classification (ABSOLUTE FINAL)",
               cv2.resize(frame,(DISPLAY_W,DISPLAY_H)))
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
