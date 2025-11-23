import platform
import numpy as np
import cv2
import os

# These imports must not execute Picamera2 during pytest
try:
    from picamera2 import Picamera2
except Exception:
    Picamera2 = None

from vit_embedder import ViTEmbedder
from anomaly_detector import OnlineAnomalyDetector
from buffered_clip_writer import BufferedClipWriter
from event_manager import EventManager

MODEL_PATH = "data/models/model_latest.pkl"

# Overridable for tests
PERIODIC_LEARNING_INTERVAL = 30
FORCE_FAKE_CAMERA = False


def is_raspberry_pi():
    if FORCE_FAKE_CAMERA:     # <-- Tests set this True
        return False

    uname = platform.uname()
    machine = uname.machine.lower()
    node = getattr(uname, "node", "").lower()
    return (
        machine.startswith("arm")
        or machine.startswith("aarch64")
        or "raspberrypi" in node
        or os.path.exists("/usr/bin/rpicam-hello")
    )


# ----------------------------------------------------------
#   Raspberry Pi Camera Capture via Picamera2
# ----------------------------------------------------------
def get_picamera2_generator(width=640, height=480, fps=30):
    if Picamera2 is None:
        raise RuntimeError("Picamera2 not available")

    picam2 = Picamera2()

    config = picam2.create_preview_configuration(
        main={"size": (width, height), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()

    try:
        while True:
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield frame
    finally:
        picam2.stop()
        picam2.close()


# ----------------------------------------------------------
#   OpenCV webcam for laptops/desktops
# ----------------------------------------------------------
def get_opencv_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam.")
        return None
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        yield frame


# ----------------------------------------------------------
#   MAIN
# ----------------------------------------------------------
def main(warmup_target=50, max_frames=None, frame_skip=5):
    # CAMERA SELECTION
    if is_raspberry_pi() and Picamera2 is not None:
        print("[INFO] Raspberry Pi detected — using Picamera2")
        frame_stream = get_picamera2_generator()
    else:
        print("[INFO] Non-Pi system detected — using OpenCV webcam")
        frame_stream = get_opencv_camera()
        if frame_stream is None:
            return

    embedder = ViTEmbedder()
    detector = OnlineAnomalyDetector(n_clusters=8)
    clip_writer = BufferedClipWriter(buffer_seconds=3, fps=5)
    events = EventManager(threshold=2.0)

    # Guarantee threshold usable even when mocked
    try:
        events.threshold = float(events.threshold)
    except Exception:
        events.threshold = 2.0

    frame_count = 0
    warmup_data = []

    if detector.load(MODEL_PATH):
        print("[Resume learning] Using previous model")
    else:
        print("[New session] Starting warm-up")

    print("Monitoring… press 'q' to quit")

    # ============================ MAIN LOOP ============================
    for frame in frame_stream:
        frame_count += 1

        if max_frames is not None and frame_count > max_frames:
            break

        clip_writer.add_frame(frame)

        if frame_count % frame_skip != 0:
            continue

        emb = embedder.get_embedding(frame)

        # ---------- Warmup ----------
        if not detector.trained:
            warmup_data.append(emb)
            cv2.putText(frame, "Warming up…", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            if len(warmup_data) >= warmup_target:
                detector.update(np.array(warmup_data))
                detector.trained = True
                detector.save(MODEL_PATH)
                print(f"[Warmup complete] {len(warmup_data)} samples → model saved")
            continue

        # ---------- Detection ----------
        anomaly_raw = detector.score(emb)
        try:
            anomaly = float(anomaly_raw)
        except Exception:
            anomaly = 0.0

        color = (0, 0, 255) if anomaly > events.threshold else (0, 255, 0)

        cv2.putText(frame, f"A: {float(anomaly):.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        events.update(anomaly, frame, clip_writer)

        # ---------- Periodic Learning ----------
        if anomaly < 1.0 and (frame_count % PERIODIC_LEARNING_INTERVAL == 0):
            detector.update(np.array([emb]))

            if frame_count % 1000 == 0:
                detector.save_checkpoint()
                detector.save(MODEL_PATH)

        # ---------- Display ----------
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.save(MODEL_PATH)
    cv2.destroyAllWindows()
    print("Stopped — model saved for next run.")


if __name__ == "__main__":
    main()
