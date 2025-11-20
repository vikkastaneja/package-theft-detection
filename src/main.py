import cv2
import numpy as np
import time
import os

from vit_embedder import ViTEmbedder
from anomaly_detector import OnlineAnomalyDetector
from buffered_clip_writer import BufferedClipWriter
from event_manager import EventManager


MODEL_PATH = "data/models/model_latest.pkl"


def main(warmup_target=50, max_frames=None, frame_skip=5):
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    embedder = ViTEmbedder()
    detector = OnlineAnomalyDetector(n_clusters=8)
    clip_writer = BufferedClipWriter(buffer_seconds=3, fps=5)
    events = EventManager(threshold=2.0)

    frame_count = 0
    # frame_skip = 5
    warmup_data = []

    # Load previous model
    if detector.load(MODEL_PATH):
        print("[Resume learning] Using previous model")
    else:
        print("[New session] Starting warm-up")

    print("Monitoring… press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # ====================== TEST EXIT CONDITION ======================
        if max_frames is not None and frame_count > max_frames:
            break

        clip_writer.add_frame(frame)

        if frame_count % frame_skip != 0:
            continue

        emb = embedder.get_embedding(frame)

        # ====================== WARM-UP ======================
        if not detector.trained:
            warmup_data.append(emb)
            cv2.putText(frame, "Warming up…", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)

            if len(warmup_data) >= warmup_target:
                detector.update(np.array(warmup_data))
                detector.trained = True
                detector.save(MODEL_PATH)
                print(f"[Warmup complete] {len(warmup_data)} samples → model saved")
            continue

        # ==================== DETECTION PHASE ====================
        anomaly = detector.score(emb)
        color = (0, 0, 255) if anomaly > events.threshold else (0, 255, 0)
        cv2.putText(frame, f"A: {anomaly:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        events.update(anomaly, frame, clip_writer)

        # Periodic learning
        if anomaly < 1.0 and (frame_count % 30 == 0):
            detector.update(np.array([emb]))
            if frame_count % 1000 == 0:
                detector.save_checkpoint()
                detector.save(MODEL_PATH)

        # Show frame
        cv2.imshow("camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    detector.save(MODEL_PATH)
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped — model saved for next run.")


if __name__ == "__main__":
    main()