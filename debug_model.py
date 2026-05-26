import cv2
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"   # same path as your app.py

print("Loading model...")
model = YOLO(MODEL_PATH)

# ── THIS IS THE KEY ──────────────────────────────────────────────────────────
# Print the class names YOUR model actually has, not what we assumed.
print("\n========================================")
print("  MODEL CLASS NAMES (model.names):")
for idx, name in model.names.items():
    print(f"    class {idx}  ->  '{name}'")
print("========================================\n")

# Open camera and print every detection raw so you can verify in terminal
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Camera started. Hold a traffic light in front of it.")
print("Press Q in the window to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, verbose=False)
    r = results[0]

    if len(r.boxes) > 0:
        for box in r.boxes:
            cls_id     = int(box.cls[0])
            conf       = float(box.conf[0])
            # Use model.names directly - NOT our hardcoded dict
            class_name = model.names[cls_id]
            print(f"  Detected  class_id={cls_id}  name='{class_name}'  conf={conf:.2f}")

    annotated = r.plot()
    cv2.imshow("Debug - Check class names in terminal", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()