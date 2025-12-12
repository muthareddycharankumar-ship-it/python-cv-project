import cv2
from ultralytics import YOLO

# -----------------------------
# Load YOLOv11 model (GPU auto)
# -----------------------------
model = YOLO("yolo11n.pt")    # change to best.pt if needed

# -----------------------------------
# Enable ByteTrack instead of default
# -----------------------------------
tracker_config = "bytetrack.yaml"   # already provided by Ultralytics

# -----------------------------------
# RTSP Camera Stream
# -----------------------------------
video_path = "rtsp://admin:admin123@192.168.0.190:554/main"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("❌ ERROR: Cannot open RTSP stream")
    exit()

# -----------------------------------
# Main Loop
# -----------------------------------
while True:
    success, frame = cap.read()
    if not success:
        print("Stream ended or lost connection.")
        break

    # -----------------------------------
    # YOLOv11 + ByteTrack Tracking
    # -----------------------------------
    results = model.track(
        frame,
        persist=True,         # keep track IDs across frames
        tracker=tracker_config,  # use ByteTrack
        verbose=False
    )

    # -----------------------------------
    # Visualization
    # -----------------------------------
    annotated = results[0].plot()

    cv2.imshow("YOLOv11 + ByteTrack", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
