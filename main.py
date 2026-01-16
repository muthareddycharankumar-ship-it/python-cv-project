from ultralytics import YOLO
import cv2
import sys
import cvzone
import numpy as np
from sort import *
import os
import csv
from datetime import datetime
from classifier import classify_cup


class ObjectDetection:
    def __init__(self, capture, result):
        self.capture = capture
        self.result = result
        self.model = self.load_model()

        # COCO bottle class → cup
        self.CUP_CLASS_ID = 39

        # Cup types
        self.CUP_TYPES = ["Glass_cups", "Paper_cups", "Porcelain_cups"]

        #  USE YOUR CLICKED LINE COORDINATES HERE
        self.center_line = [(15, 400), (1261, 400)]

        # Counts
        self.counts = {c: {"filled": 0, "empty": 0} for c in self.CUP_TYPES}

        self.already_counted = set()
        self.previous_positions = {}

        os.makedirs("cropped_images", exist_ok=True)
        os.makedirs(self.result, exist_ok=True)

        self.csv_file = os.path.join(self.result, "counts_log.csv")
        self.init_csv()

    # ================= MODEL =================
    def load_model(self):
        model = YOLO("runs/detect/train/weights/best.pt")
        model.fuse()
        return model

    # ================= CSV =================
    def init_csv(self):
        if not os.path.exists(self.csv_file):
            with open(self.csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Cup_Type", "Status", "Timestamp"])

    def log_csv(self, cup_type, status):
        with open(self.csv_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                cup_type,
                status,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ])

    # ================= YOLO =================
    def predict(self, img):
        return self.model(img, stream=True)

    # ================= DETECTION =================
    def plot_boxes(self, results, img, detections):
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                # Only bottle → cup
                if cls != self.CUP_CLASS_ID or conf < 0.6:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                detections = np.vstack((
                    detections,
                    np.array([x1, y1, x2, y2, conf])
                ))

                cvzone.cornerRect(
                    img, (x1, y1, x2 - x1, y2 - y1),
                    l=8, rt=2, colorR=(0, 255, 0)
                )
                cvzone.putTextRect(
                    img, "Cup",
                    (x1, y1 - 10),
                    scale=1, thickness=1, colorR=(255, 0, 0)
                )

        return detections, img

    # ================= TRACK + COUNT =================
    def track_detect(self, detections, tracker, img):
        tracks = tracker.update(detections)

        for x1, y1, x2, y2, obj_id in tracks:
            x1, y1, x2, y2, obj_id = map(int, (x1, y1, x2, y2, obj_id))
            crop = img[y1:y2, x1:x2]

            if self.crossed_line((x1, y1, x2, y2), obj_id):

                result = classify_cup(crop).lower()

                if "glass" in result:
                    cup_type = "Glass_cups"
                elif "paper" in result:
                    cup_type = "Paper_cups"
                elif "porcelain" in result:
                    cup_type = "Porcelain_cups"
                else:
                    continue

                status = "filled" if "filled" in result else "empty"

                self.counts[cup_type][status] += 1
                self.log_csv(cup_type, status)

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(
                    f"cropped_images/{cup_type}_{status}_{obj_id}_{ts}.jpg",
                    crop
                )

        # Display counts
        y = 30
        for c in self.CUP_TYPES:
            text = f"{c} | Filled: {self.counts[c]['filled']}  Empty: {self.counts[c]['empty']}"
            cv2.putText(
                img, text, (20, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
            y += 30

        return img

    # ================= ROBUST LINE CROSS =================
    def crossed_line(self, box, obj_id):
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        (xA, yA), (xB, yB) = self.center_line
        slope = (yB - yA) / (xB - xA)
        y_line = slope * cx + (yA - slope * xA)

        THRESH = 8  # tolerance in pixels

        if obj_id in self.already_counted:
            return False

        dist = cy - y_line

        if obj_id not in self.previous_positions:
            self.previous_positions[obj_id] = dist
            return False

        prev_dist = self.previous_positions[obj_id]

        if (prev_dist < -THRESH and dist > THRESH) or (prev_dist > THRESH and dist < -THRESH):
            self.already_counted.add(obj_id)
            print(f"[COUNTED] ID {obj_id}")
            return True

        self.previous_positions[obj_id] = dist
        return False

    # ================= MAIN =================
    def __call__(self):
        cap = cv2.VideoCapture(self.capture)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 30

        # 🔑 LESS STRICT SORT (IMPORTANT)
        tracker = Sort(max_age=15, min_hits=1, iou_threshold=0.3)

        while True:
            ret, img = cap.read()
            if not ret:
                break

            detections = np.empty((0, 5))
            results = self.predict(img)
            detections, img = self.plot_boxes(results, img, detections)
            img = self.track_detect(detections, tracker, img)

            cv2.line(img, self.center_line[0], self.center_line[1], (0, 255, 0), 2)
            cv2.imshow("Cup Counter", img)

            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


# ================= ENTRY =================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <video_path>")
        sys.exit(1)

    detector = ObjectDetection(sys.argv[1], "result")
    detector()
