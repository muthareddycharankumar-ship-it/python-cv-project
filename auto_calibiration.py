# speed_track_sort.py
import cv2
import numpy as np
import time, math, os, torch
from ultralytics import YOLO

# ---------------- CONFIG ----------------
MODEL_PATH = "yolov8s.pt"         # or yolov8n.pt if you want faster but less accurate
CONF_THRES = 0.30

REAL_DISTANCE_M = 5.0             # set measured distance between lines (meters)
KMH = 3.6

DT_MIN = 0.03
DT_MAX = 10.0
MAX_SPEED = 200

# SORT / tracking params
MAX_AGE = 12      # frames to keep track without updates
MIN_HITS = 1      # frames before confirming track
IOU_MATCH = 0.3   # IOU threshold for match

# detection / display
SAVE_DIR = "speed_captures"
os.makedirs(SAVE_DIR, exist_ok=True)

RTSP_URL = "/home/safepro/Downloads/vlc-record-2025-12-11-13h26m20s-rtsp___192.168.0.73_554_stream0-.mp4"
DISPLAY_W = 1280
DISPLAY_H = 720

# wheel bottom offset
WHEEL_BOTTOM_OFFSET = 5  # y2 - offset

# device
DEVICE = 0 if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# load model
model = YOLO(MODEL_PATH)

# ---------------- utility functions ----------------
def iou(bb_test, bb_gt):
    """Compute IoU between two boxes [x1,y1,x2,y2]"""
    xx1 = np.maximum(bb_test[0], bb_gt[0])
    yy1 = np.maximum(bb_test[1], bb_gt[1])
    xx2 = np.minimum(bb_test[2], bb_gt[2])
    yy2 = np.minimum(bb_test[3], bb_gt[3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    inter = w * h
    area1 = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
    area2 = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
    union = area1 + area2 - inter
    return inter / (union + 1e-9)

def hard_nms(boxes, scores, iou_thr=0.4):
    """Return indices of boxes kept after hard NMS (descending score)"""
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes)
    scores = np.array(scores)
    idxs = scores.argsort()[::-1]
    keep = []
    while len(idxs) > 0:
        curr = idxs[0]
        keep.append(curr)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        x1 = np.maximum(boxes[curr, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[curr, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[curr, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[curr, 3], boxes[rest, 3])
        w = np.maximum(0., x2 - x1)
        h = np.maximum(0., y2 - y1)
        inter = w * h
        area_curr = (boxes[curr,2] - boxes[curr,0])*(boxes[curr,3] - boxes[curr,1])
        area_rest = (boxes[rest,2] - boxes[rest,0])*(boxes[rest,3] - boxes[rest,1])
        union = area_curr + area_rest - inter
        ious = inter / (union + 1e-9)
        idxs = rest[ious < iou_thr]
    return keep

# ---------------- SORT classes ----------------
class KalmanBoxTracker:
    """
    A single-track Kalman filter based tracker for a bbox (cx,cy,w,h).
    State vector: [cx, cy, s, r, vx, vy, vs] where s=scale(width*height), r=aspect
    We use a simple 7D filter similar to SORT paper.
    """
    count = 0
    def __init__(self, bbox):
        # bbox: [x1,y1,x2,y2]
        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2.
        cy = y1 + h/2.
        s = w * h
        r = w / float(h + 1e-6)

        # we'll implement a Kalman filter with OpenCV for simplicity
        # state: [cx, cy, s, r, vx, vy, vs]
        self.kf = cv2.KalmanFilter(7, 4)  # 7 state, 4 measurement
        # transition matrix
        self.kf.transitionMatrix = np.eye(7, dtype=np.float32)
        for i in range(3):
            self.kf.transitionMatrix[i, i+4] = 1.0  # position <- velocity
        # measurement matrix (we measure cx,cy,s,r)
        self.kf.measurementMatrix = np.zeros((4,7), dtype=np.float32)
        self.kf.measurementMatrix[0,0] = 1.0
        self.kf.measurementMatrix[1,1] = 1.0
        self.kf.measurementMatrix[2,2] = 1.0
        self.kf.measurementMatrix[3,3] = 1.0

        # process & measurement noise
        self.kf.processNoiseCov = np.eye(7, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1

        self.kf.statePre = np.zeros((7,1), dtype=np.float32)
        self.kf.statePre[0,0] = cx
        self.kf.statePre[1,0] = cy
        self.kf.statePre[2,0] = s
        self.kf.statePre[3,0] = r
        self.kf.statePre[4,0] = 0
        self.kf.statePre[5,0] = 0
        self.kf.statePre[6,0] = 0

        self.kf.statePost = self.kf.statePre.copy()
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

        self.history = []
        self.bbox = bbox  # last bbox

    def update(self, bbox):
        """Correct with observed bbox"""
        x1,y1,x2,y2 = bbox
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2.
        cy = y1 + h/2.
        s = w * h
        r = w / float(h + 1e-6)
        meas = np.array([[np.float32(cx)], [np.float32(cy)], [np.float32(s)], [np.float32(r)]])
        self.kf.correct(meas)
        self.bbox = bbox
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

    def predict(self):
        """Advance state and return predicted bbox"""
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        state = self.kf.statePost.flatten()
        cx, cy, s, r = state[0], state[1], state[2], state[3]
        w = math.sqrt(abs(s*r)) if s>0 and r>0 else 0
        h = s / (w + 1e-6) if w>0 else 0
        x1 = cx - w/2.
        y1 = cy - h/2.
        x2 = cx + w/2.
        y2 = cy + h/2.
        pred_box = [int(x1), int(y1), int(x2), int(y2)]
        self.history.append(pred_box)
        return pred_box

    def get_state(self):
        return self.bbox

# ---------------- SORT manager ----------------
class SortManager:
    def __init__(self, max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_MATCH):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []

    def update(self, detections):
        """
        detections: list of bbox [x1,y1,x2,y2]
        returns: list of (bbox, track_id)
        """
        # Predict step for all trackers
        trks = []
        to_del = []
        for t in self.trackers:
            pos = t.predict()
            trks.append(pos)
        if len(self.trackers) == 0:
            trks = np.empty((0,4))
        else:
            trks = np.array(trks)

        dets = np.array(detections)
        N = len(trks)
        M = len(dets)

        matched, unmatched_trks, unmatched_dets = [], list(range(N)), list(range(M))

        if N>0 and M>0:
            iou_matrix = np.zeros((N,M), dtype=np.float32)
            for i in range(N):
                for j in range(M):
                    iou_matrix[i,j] = iou(trks[i], dets[j])
            # Hungarian style greedy matching based on largest IoU
            for i in range(N):
                j = np.argmax(iou_matrix[i])
                if iou_matrix[i,j] >= self.iou_threshold:
                    matched.append((i,j))
            matched_trk_idx = [m[0] for m in matched]
            matched_det_idx = [m[1] for m in matched]
            unmatched_trks = [i for i in range(N) if i not in matched_trk_idx]
            unmatched_dets = [j for j in range(M) if j not in matched_det_idx]
        else:
            unmatched_trks = list(range(N))
            unmatched_dets = list(range(M))

        # Update matched trackers with assigned detections
        for (i,j) in matched:
            self.trackers[i].update(dets[j].tolist())

        # Create and initialise new trackers for unmatched detections
        for j in unmatched_dets:
            trk = KalmanBoxTracker(dets[j].tolist())
            trk.update(dets[j].tolist())
            self.trackers.append(trk)

        # Remove dead trackers
        ret = []
        new_trackers = []
        for trk in self.trackers:
            if trk.time_since_update < 1 and (trk.hits >= self.min_hits or trk.age <= self.min_hits):
                ret.append((trk.get_state(), trk.id))
            if trk.time_since_update <= self.max_age:
                new_trackers.append(trk)
        self.trackers = new_trackers
        return ret

# instantiate sorter
sorter = SortManager()

# ---------------- main loop (detection + tracking + speed) ----------------
cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    print("ERROR: cannot open video")
    raise SystemExit

print("Stream opened, running... (ESC to quit)")

# track state for speed logic per SORT ID (not Kalman.id)
speed_tracks = {}  # track_id -> {tA, tB, dir, speed, last_foot, missed}
now_time = lambda: time.time()

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        time.sleep(0.01)
        continue

    frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
    h, w = frame.shape[:2]
    L1 = int(h * 0.45)
    L2 = int(h * 0.75)

    # draw lines
    disp = frame.copy()
    cv2.line(disp, (0, L1), (w, L1), (0,255,255), 3)
    cv2.line(disp, (0, L2), (w, L2), (0,255,255), 3)

    # YOLO detect on resized frame to keep coords consistent
    try:
        yres = model(frame, imgsz=640, device=DEVICE, verbose=False)[0]
    except Exception as e:
        print("YOLO error:", e)
        continue

    raw_boxes = []
    raw_scores = []
    raw_classes = []

    if yres.boxes is not None:
        xyxy = yres.boxes.xyxy.cpu().numpy()
        confs = yres.boxes.conf.cpu().numpy()
        classes = yres.boxes.cls.cpu().numpy()
        for i, b in enumerate(xyxy):
            if confs[i] < CONF_THRES:
                continue
            cls = int(classes[i])
            if cls not in [2,3,5,7]:
                continue
            x1,y1,x2,y2 = map(int, b)
            if (x2-x1) < 30 or (y2-y1) < 30:
                continue
            raw_boxes.append([x1,y1,x2,y2])
            raw_scores.append(float(confs[i]))
            raw_classes.append(int(cls))

    # NMS to remove duplicate boxes (hard)
    keep_idx = hard_nms(np.array(raw_boxes) if len(raw_boxes) else np.array([]),
                        np.array(raw_scores) if len(raw_scores) else np.array([]),
                        iou_thr=0.45)
    boxes_kept = [raw_boxes[i] for i in keep_idx] if len(raw_boxes) else []

    # Feed detections to SORT
    tracked = sorter.update(boxes_kept)  # returns list of (bbox, kalman_id)

    # tracked list contains only confirmed trackers; we need unique ids for speed_tracks
    # We'll use kalman tracker.id as stable id.
    # Ensure speed_tracks entry exists per id
    for bbox, k_id in tracked:
        x1,y1,x2,y2 = map(int, bbox)
        # get wheel bottom point (use bottom - offset)
        foot_x = (x1 + x2)//2
        foot_y = max(0, y2 - WHEEL_BOTTOM_OFFSET)

        if k_id not in speed_tracks:
            speed_tracks[k_id] = {
                "tA": None, "tB": None, "dir": None, "speed": None, "last_foot": foot_y, "missed": 0
            }

        st = speed_tracks[k_id]
        # draw box and id
        cv2.rectangle(disp, (x1,y1), (x2,y2), (255,0,0), 2)
        cv2.putText(disp, f"ID {k_id}", (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.circle(disp, (foot_x, foot_y), 4, (0,255,0), -1)

        # skip tiny jitter
        if abs(foot_y - st["last_foot"]) < 2:
            st["last_foot"] = foot_y
            continue

        now = now_time()
        # ENTRY detection
        if st["tA"] is None:
            if st["last_foot"] < L1 and foot_y >= L1:
                st["tA"] = now
                st["dir"] = "down"
            elif st["last_foot"] > L2 and foot_y <= L2:
                st["tA"] = now
                st["dir"] = "up"
            elif L1 < foot_y < L2:
                st["tA"] = now
                st["dir"] = "down" if foot_y > st["last_foot"] else "up"

        # EXIT detection
        if st["tA"] is not None and st["tB"] is None:
            if st["dir"] == "down" and st["last_foot"] < L2 and foot_y >= L2:
                st["tB"] = now
            elif st["dir"] == "up" and st["last_foot"] > L1 and foot_y <= L1:
                st["tB"] = now

        # speed compute and save only when speed found
        if st["tA"] and st["tB"] and st["speed"] is None:
            dt = st["tB"] - st["tA"]
            if DT_MIN < dt < DT_MAX:
                speed_kmh = (REAL_DISTANCE_M / dt) * KMH
                if speed_kmh < MAX_SPEED:
                    st["speed"] = speed_kmh
                    # draw speed before save
                    cv2.putText(disp, f"{speed_kmh:.1f} km/h", (x1, y1 - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    fname = f"ID{k_id}_{int(speed_kmh)}kmh_{int(now)}.jpg"
                    cv2.imwrite(os.path.join(SAVE_DIR, fname), disp)
                    print("Saved:", fname)

        # show speed if present
        if st["speed"] is not None:
            cv2.putText(disp, f"{st['speed']:.1f} km/h", (x1, y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        st["last_foot"] = foot_y
        st["missed"] = 0

    # age speed_tracks: increment missed for tracks not present this frame
    present_ids = {k for _,k in tracked}
    for k in list(speed_tracks.keys()):
        if k not in present_ids:
            speed_tracks[k]["missed"] += 1
            if speed_tracks[k]["missed"] > MAX_AGE:
                del speed_tracks[k]

    # display medium window
    disp_small = cv2.resize(disp, (int(DISPLAY_W*0.9), int(DISPLAY_H*0.9)))
    cv2.imshow("SPEED SORT", disp_small)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
