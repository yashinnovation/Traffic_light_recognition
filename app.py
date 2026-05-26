
import cv2
import numpy as np
import base64
import threading
import time
import json
from flask import Flask, render_template, Response, request, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# ── Model ───────────────────────────────────────────────────────────────────
MODEL_PATH = "models/best.pt"
try:
    model = YOLO(MODEL_PATH)
    # Use model's OWN class names - never hardcode, order may differ
    CLASS_NAMES = {k: v.strip().title() for k, v in model.names.items()}
    print(f"✅ Model loaded: {MODEL_PATH}")
    print(f"   Class map: {CLASS_NAMES}")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    model = None
    CLASS_NAMES = {}

# ── Shared buffer (written by capture thread, read by stream) ────────────────
_lock   = threading.Lock()
_buffer = {
    "jpg":        None,
    "label":      "None",
    "confidence": 0.0,
    "active":     True,    # set False by /camera/off to pause inference
}

# ── Capture + inference thread ───────────────────────────────────────────────
def capture_loop():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    if not cap.isOpened():
        print("❌ Camera not available"); return

    print("📷 Capture thread started")

    frame_n     = 0
    EVERY       = 2        # infer every N frames  (lower = more accurate, higher = faster)
    last_label  = "None"
    last_conf   = 0.0
    last_ann    = None

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05); continue

        frame_n += 1

        if model is not None and _buffer.get("active", True) and frame_n % EVERY == 0:
            results    = model(frame, stream=False, conf=0.5, verbose=False)
            r          = results[0]
            last_ann   = r.plot()

            best_label = "None"
            best_conf  = 0.0
            for box in r.boxes:
                c = float(box.conf[0])
                if c > best_conf:
                    best_conf  = c
                    best_label = CLASS_NAMES.get(int(box.cls[0]), "None")

            last_label = best_label
            last_conf  = round(best_conf, 4)

        display = last_ann if last_ann is not None else frame

        ok, buf = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, 78])
        if not ok:
            continue

        # Single atomic write — jpg + label always updated together
        with _lock:
            _buffer["jpg"]        = buf.tobytes()
            _buffer["label"]      = last_label
            _buffer["confidence"] = last_conf

    cap.release()


threading.Thread(target=capture_loop, daemon=True).start()


# ── MJPEG generator — embeds X-Detection header in every part ───────────────
def gen_frames():
    """
    Each multipart chunk looks like:

        --frame\r\n
        Content-Type: image/jpeg\r\n
        X-Detection: {"label": "Green", "confidence": 0.91}\r\n
        \r\n
        <jpeg bytes>
        \r\n

    The JS frontend reads the raw stream, parses this header, and updates
    the dashboard with data that came from the EXACT same bytes as the image.
    """
    while True:
        with _lock:
            jpg   = _buffer["jpg"]
            label = _buffer["label"]
            conf  = _buffer["confidence"]

        if jpg:
            meta = json.dumps({"label": label, "confidence": conf})
            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n'
                + f'X-Detection: {meta}\r\n'.encode()
                + b'\r\n'
                + jpg
                + b'\r\n'
            )
        time.sleep(0.033)


# ── Routes ───────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/camera/on',  methods=['POST'])
def camera_on():
    _buffer['active'] = True
    return ('', 204)

@app.route('/camera/off', methods=['POST'])
def camera_off():
    _buffer['active'] = False
    return ('', 204)

@app.route('/state')
def state():
    with _lock:
        return jsonify({"label": _buffer["label"], "confidence": _buffer["confidence"]})


@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file      = request.files['file']
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img       = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    results       = model(img, stream=False, verbose=False)
    annotated_img = results[0].plot()

    best_label = "None"
    best_conf  = 0.0
    for box in results[0].boxes:
        c = float(box.conf[0])
        if c > best_conf:
            best_conf  = c
            best_label = CLASS_NAMES.get(int(box.cls[0]), "None")

    _, buffer   = cv2.imencode('.jpg', annotated_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({
        'image':      encoded_img,
        'label':      best_label,
        'confidence': round(best_conf, 4),
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False,
            threaded=True, use_reloader=False)