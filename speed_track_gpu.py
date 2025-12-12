import cv2
from ultralytics import YOLO
import time, math, os, torch

# CONFIG 
MODEL_PATH = "yolov8n.pt"
CONF_THRES = 0.25

REAL_DISTANCE_M = 5.0      # 👈 Tune this only for accuracy
KMH_CONVERSION = 3.6

DT_MIN = 0.03
DT_MAX = 8.0
MAX_SPEED_KMH = 100
MAX_MISSED = 12

RTSP_URL = "rtsp://admin:123456@192.168.0.73:554/stream1"
SAVE_DIR = "speed_captures"
os.makedirs(SAVE_DIR, exist_ok=True)

# GPU
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

model = YOLO(MODEL_PATH)

# TRACK STORAGE
tracks = {}
next_track_id = 1
next_uid = 1

# CAMERA
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FPS, 20)

if not cap.isOpened():
    print(" Camera failed")
    exit()

print(" Camera connected")

# MAIN LOOP
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    h, w = frame.shape[:2]

    #  AUTOMATIC YELLOW LINES
    line_top = int(h * 0.45)
    line_bot = int(h * 0.75)

    disp = frame.copy()

    cv2.line(disp, (0, line_top), (w, line_top), (0,255,255), 3)
    cv2.line(disp, (0, line_bot), (w, line_bot), (0,255,255), 3)

    # YOLO 
    results = model(disp, imgsz=640, device=DEVICE, verbose=False)[0]
    detections = []

    if results.boxes:
        dets = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()

        for i, box in enumerate(dets):
            if confs[i] < CONF_THRES:
                continue

            if int(classes[i]) not in [2,3,5,7]:
                continue

            x1,y1,x2,y2 = map(int, box)
            cx = (x1+x2)//2
            cy = (y1+y2)//2
            detections.append((cx,cy,[x1,y1,x2,y2]))

    # TRACKING 
    updated = set()

    for cx,cy,bbox in detections:
        matched = False
        for tid,t in tracks.items():
            d = abs(cx - t["cx"]) + abs(cy - t["cy"])
            if d < 120:
                t["last_cy"] = t["cy"]
                t["cx"], t["cy"] = cx, cy
                t["bbox"] = bbox
                t["missed"] = 0
                updated.add(tid)
                matched = True
                break

        if not matched:
            tracks[next_track_id] = {
                "cx":cx,"cy":cy,"last_cy":cy,"bbox":bbox,
                "tA":None,"tB":None,"direction":None,
                "speed":None,"uid":next_uid,"missed":0
            }
            next_uid += 1
            next_track_id += 1

    for tid in list(tracks.keys()):
        if tid not in updated:
            tracks[tid]["missed"] += 1
            if tracks[tid]["missed"] > MAX_MISSED:
                del tracks[tid]

    # SPEED LOGIC 
    now = time.time()

    for tid,t in tracks.items():
        cx,cy = t["cx"], t["cy"]
        x1,y1,x2,y2 = t["bbox"]

        if abs(cy - t["last_cy"]) < 2:
            continue

        uid = t["uid"]

        cv2.rectangle(disp,(x1,y1),(x2,y2),(255,0,0),2)
        cv2.putText(disp,f"UID {uid}",(x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        #  ENTRY
        if t["tA"] is None:
            if t["last_cy"] < line_top and cy >= line_top:
                t["tA"] = now
                t["direction"] = "down"
            elif t["last_cy"] > line_bot and cy <= line_bot:
                t["tA"] = now
                t["direction"] = "up"

        #  EXIT
        if t["tA"] is not None and t["tB"] is None:
            if t["direction"] == "down" and t["last_cy"] < line_bot and cy >= line_bot:
                t["tB"] = now
            elif t["direction"] == "up" and t["last_cy"] > line_top and cy <= line_top:
                t["tB"] = now

        #  SPEED
        if t["tA"] and t["tB"] and t["speed"] is None:
            dt = t["tB"] - t["tA"]
            if DT_MIN < dt < DT_MAX:
                sp = (REAL_DISTANCE_M / dt) * KMH_CONVERSION
                if sp < MAX_SPEED_KMH:
                    t["speed"] = sp
                    fname = f"UID{uid}_{int(sp)}kmh_{int(now)}.jpg"
                    cv2.imwrite(os.path.join(SAVE_DIR,fname), disp)
                    print(" Saved:", fname)

        if t["speed"]:
            cv2.putText(disp,f"{t['speed']:.1f} km/h",
                        (x1,y1-25),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        t["last_cy"] = cy

    cv2.imshow("AUTOMATIC YELLOW LINE SPEED SYSTEM", disp)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
