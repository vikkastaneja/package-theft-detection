import numpy as np
from unittest.mock import MagicMock
import main
import pytest


# Ultra-light fake camera that won't kill Raspberry Pi memory
def fake_opencv_stream(n
):
    for _ in range(n):
        yield np.zeros((64, 64, 3), dtype=np.uint8)


class FakeCapture:
    def __init__(self, frames=10):
        self.frames = frames
        self.count = 0
    def isOpened(self):
        return True
    def read(self):
        if self.count < self.frames:
            self.count += 1
            # Tiny frame to avoid RAM blowout
            return True, np.zeros((32, 32, 3), dtype=np.uint8)
        return False, None


# ==========================================================================================
#   TEST 1: basic execution path
# ==========================================================================================
def test_main_executes_and_calls_detectors(monkeypatch):
    monkeypatch.setattr(main, "is_raspberry_pi", lambda: False)
    monkeypatch.setattr(main, "FORCE_FAKE_CAMERA", True)

    monkeypatch.setattr(main.cv2, "VideoCapture",
                        lambda *_: FakeCapture(frames=10))
    monkeypatch.setattr(main.cv2, "imshow", lambda *a, **kw: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *_: ord(" "))

    fake_embedder = MagicMock()
    fake_embedder.get_embedding.return_value = np.zeros(16)
    monkeypatch.setattr(main, "ViTEmbedder", lambda: fake_embedder)

    fake_detector = MagicMock()
    fake_detector.trained = False
    fake_detector.load.return_value = False
    monkeypatch.setattr(main, "OnlineAnomalyDetector",
                        lambda *a, **kw: fake_detector)

    fake_events = MagicMock()
    fake_events.threshold = 2.0
    monkeypatch.setattr(main, "EventManager", lambda *a, **kw: fake_events)

    fake_writer = MagicMock()
    monkeypatch.setattr(main, "BufferedClipWriter",
                        lambda *a, **kw: fake_writer)

    # Critical: trigger learning fast
    monkeypatch.setattr(main, "PERIODIC_LEARNING_INTERVAL", 10)

    main.main(warmup_target=2, max_frames=5, frame_skip=1)

    assert fake_detector.update.call_count >= 1


# ==========================================================================================
#   TEST 2: periodic learning (frame_count % INTERVAL == 0)
# ==========================================================================================
def test_periodic_learning_triggered(monkeypatch):
    monkeypatch.setattr(main, "is_raspberry_pi", lambda: False)
    monkeypatch.setattr(main, "FORCE_FAKE_CAMERA", True)

    # Only 12 tiny frames → safe for Pi
    
    monkeypatch.setattr(main.cv2, "imshow", lambda *a, **kw: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *_: ord(" "))

    fake_embedder = MagicMock()
    fake_embedder.get_embedding.return_value = np.zeros(16)
    monkeypatch.setattr(main, "ViTEmbedder", lambda: fake_embedder)

    fake_detector = MagicMock()
    fake_detector.trained = True
    fake_detector.load.return_value = True
    fake_detector.score.return_value = 0.5  # < 1.0 triggers periodic learn
    monkeypatch.setattr(main, "OnlineAnomalyDetector",
                        lambda *a, **kw: fake_detector)

    fake_events = MagicMock()
    fake_events.threshold = 2.0
    monkeypatch.setattr(main, "EventManager", lambda *a, **kw: fake_events)

    fake_writer = MagicMock()
    monkeypatch.setattr(main, "BufferedClipWriter",
                        lambda *a, **kw: fake_writer)

    # Make periodic trigger hit at frame 10
    monkeypatch.setattr(main, "PERIODIC_LEARNING_INTERVAL", 10)
    monkeypatch.setattr(main, "get_opencv_camera", lambda *_: fake_opencv_stream(12))

    main.main(warmup_target=0, max_frames=12, frame_skip=1)

    fake_detector.update.assert_called()
