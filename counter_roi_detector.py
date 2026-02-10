import cv2
import numpy as np
import os
import time

# ================= CONFIG =================
VIDEO_SOURCE = "/home/safepro/ccd2-1.mp4"
SAVE_DIR = "/home/safepro/Desktop/opencv/event_out"

# Pre-defined polygon ROI (counter area)
USE_PREDEFINED_ROI = True  # Set to False to draw manually
PREDEFINED_POLYGON = [
    (362, 490),
    (523, 670),
    (797, 438),
    (599, 308),
    (365, 487)
]

STATIC_TIMEOUT = 2.0
MIN_MOTION_PIXELS = 800  # Lower threshold
BG_WARMUP_FRAMES = 40

os.makedirs(SAVE_DIR, exist_ok=True)

# ================= GLOBALS =================
drawing_points = []
drawing = False
roi_finalized = False
roi_armed = False
warmup_counter = 0
bg = None

# ================= MOUSE CALLBACK =================
def mouse_draw(event, x, y, flags, param):
    global drawing_points, drawing, roi_finalized, roi_armed, warmup_counter, bg
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        drawing_points = [(x, y)]
        roi_finalized = False
        roi_armed = False
        warmup_counter = 0
        print("🖊️  Drawing ROI... (hold and drag)")
    
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        drawing_points.append((x, y))
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if len(drawing_points) > 10:
            roi_finalized = True
            warmup_counter = 0
            bg = cv2.createBackgroundSubtractorMOG2(
                history=400, varThreshold=16, detectShadows=False
            )
            print(f"✅ ROI finalized — warming up ({len(drawing_points)} points)")
        else:
            print("❌ ROI too small, draw again")
            drawing_points = []

def create_mask(frame_shape, points):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if len(points) > 2:
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

# ================= VIDEO =================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print("❌ Cannot open video")
    exit()

fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {W}x{H} @ {fps} fps")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
recording = False
last_motion_time = 0
out = None
current_filename = ""
polygon_mask = None

cv2.namedWindow("Counter Monitor", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Counter Monitor", mouse_draw)

# Initialize with predefined polygon if enabled
if USE_PREDEFINED_ROI:
    drawing_points = PREDEFINED_POLYGON.copy()
    roi_finalized = True
    roi_armed = False
    warmup_counter = 0
    bg = cv2.createBackgroundSubtractorMOG2(
        history=400, varThreshold=16, detectShadows=False
    )
    print(f"✅ Using pre-defined ROI with {len(drawing_points)} points")
    print("📌 Starting warmup...\n")
else:
    print("\n📌 INSTRUCTIONS:")
    print("1. CLICK & DRAG to draw around COUNTER AREA ONLY")
    print("2. Release mouse to finalize")
    print("3. Press 'c' to clear and redraw")
    print("4. Press 'q' to quit\n")

frame_count = 0

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
        continue

    frame_count += 1
    display = frame.copy()
    motion_detected = False

    # Draw ROI
    if len(drawing_points) > 1:
        pts = np.array(drawing_points, dtype=np.int32)
        
        if roi_finalized:
            overlay = display.copy()
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            cv2.addWeighted(overlay, 0.3, display, 0.7, 0, display)
            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
        else:
            cv2.polylines(display, [pts], False, (0, 255, 255), 2)

    # Motion detection
    if roi_finalized and bg is not None:
        if polygon_mask is None:
            polygon_mask = create_mask(frame.shape, drawing_points)
        
        fg = bg.apply(frame)
        fg_masked = cv2.bitwise_and(fg, fg, mask=polygon_mask)
        motion_pixels = cv2.countNonZero(fg_masked)
        
        if not roi_armed:
            warmup_counter += 1
            if warmup_counter >= BG_WARMUP_FRAMES:
                roi_armed = True
                print("🟢 ROI ARMED — waiting for events")
        else:
            motion_detected = motion_pixels > MIN_MOTION_PIXELS
        
        # Show motion count
        cv2.putText(display, f"Motion: {motion_pixels}px", (10, H-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    now = time.time()

    # ================= RECORDING =================
    if roi_armed and motion_detected:
        if not recording:
            current_filename = time.strftime("event_%Y%m%d_%H%M%S.mp4")
            filepath = os.path.join(SAVE_DIR, current_filename)
            out = cv2.VideoWriter(filepath, fourcc, fps, (W, H))
            recording = True
            print(f"🎬 RECORDING → {current_filename}")

        last_motion_time = now
        out.write(frame)

    elif recording and (now - last_motion_time > STATIC_TIMEOUT):
        out.release()
        out = None
        recording = False
        print(f"⛔ STOPPED → {current_filename}")
        current_filename = ""

    # ================= STATUS DISPLAY =================
    if recording:
        blink = int(time.time() * 2) % 2
        if blink:
            cv2.circle(display, (30, 30), 15, (0, 0, 255), -1)
        cv2.putText(display, "🔴 RECORDING", (60, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        cv2.putText(display, current_filename, (60, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    else:
        if roi_armed:
            status = "WAITING FOR EVENT"
            color = (0, 255, 0)
        elif roi_finalized:
            status = f"WARMING UP {warmup_counter}/{BG_WARMUP_FRAMES}"
            color = (0, 255, 255)
        elif drawing:
            status = "DRAWING..."
            color = (0, 255, 255)
        else:
            status = "DRAW COUNTER AREA"
            color = (255, 255, 255)
        
        cv2.putText(display, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Counter Monitor", display)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and not recording:
        drawing_points = []
        roi_finalized = False
        roi_armed = False
        polygon_mask = None
        warmup_counter = 0
        bg = None
        print("🔄 Cleared — draw new ROI")

# ================= CLEANUP =================
if out:
    out.release()
cap.release()
cv2.destroyAllWindows()
print("✅ Done")
