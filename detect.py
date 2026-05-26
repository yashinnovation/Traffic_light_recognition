import cv2
import numpy as np
from ultralytics import YOLO
import pygame
import time

# ==============================
# SETTINGS
# ==============================
MODEL_PATH = "models/best.pt"
CONF_THRESHOLD = 0.5
ENABLE_SOUND = True
ENABLE_ROI = True
SKIP_FRAMES = 3 

# ==============================
# Initialize sound
# ==============================
if ENABLE_SOUND:
    pygame.mixer.init()
    try:
        beep = pygame.mixer.Sound("beep.wav")
    except Exception as e:
        print(f"⚠ Warning: Could not load beep.wav ({e}). Sound disabled.")
        beep = None

# ==============================
# Load model
# ==============================
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"❌ ERROR: Could not load model at {MODEL_PATH}.")
    exit()

CLASS_NAMES = {0: "Red", 1: "Yellow", 2: "Green"}
COLORS = {"Red": (0, 0, 255), "Yellow": (0, 255, 255), "Green": (0, 255, 0)}

# ==============================
# Webcam initialization
# ==============================
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# --- AUTO EXPOSURE FIX ---
# Resetting to Auto so it's not too dark
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # 0.75 usually enables Auto on Windows
cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # Enable Auto-focus

# Set resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ ERROR: Camera could not be opened.")
    exit()

prev_time = 0
frame_count = 0
last_results = [] 

print("🚦 Traffic Light Detection Started... Press 'Q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w = frame.shape[:2]

    # 1. ROI Logic
    if ENABLE_ROI:
        y_start, y_end = 0, int(h * 0.6)
        x_start, x_end = int(w * 0.25), int(w * 0.75)
        roi = frame[y_start:y_end, x_start:x_end]
    else:
        roi = frame
        x_start, y_start = 0, 0
    
    # --- COLOR ENHANCEMENT FOR AI ---
    # Convert ROI to HSV to boost color saturation
    # This helps the AI see "Red" even if the camera washes it out
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = cv2.convertScaleAbs(hsv[:, :, 1], alpha=1.5, beta=10) # Boost saturation
    processed_roi = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. FRAME SKIPPING
    if frame_count % (SKIP_FRAMES + 1) == 0:
        # Resize the processed (color-boosted) ROI for the model
        small_roi = cv2.resize(processed_roi, (320, 320))
        
        last_results = model(small_roi, conf=CONF_THRESHOLD, verbose=False)
        
        for r in last_results:
            for box in r.boxes:
                label = CLASS_NAMES.get(int(box.cls[0]), "Unknown")
                if label == "Red":
                    print("🛑 STOP")
                    if ENABLE_SOUND and beep: beep.play()
                elif label == "Yellow": print("⚠ READY")
                elif label == "Green": print("✅ GO")

    # 3. DRAWING
    if last_results:
        for r in last_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = CLASS_NAMES.get(cls_id, "Unknown")
                color = COLORS.get(label, (255, 255, 255))

                rx1, ry1, rx2, ry2 = box.xyxy[0]
                x_scale = roi.shape[1] / 320
                y_scale = roi.shape[0] / 320
                
                x1, y1 = int(rx1 * x_scale) + x_start, int(ry1 * y_scale) + y_start
                x2, y2 = int(rx2 * x_scale) + x_start, int(ry2 * y_scale) + y_start

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    if ENABLE_ROI:
        cv2.rectangle(frame, (x_start, y_start), (int(w * 0.75), int(h * 0.6)), (255, 0, 0), 1)

    # 4. FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # 5. REFRESH WINDOW
    cv2.imshow("Traffic Light Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()