import numpy as np
import pytest
from unittest.mock import MagicMock

import main


class FakeCapture:
    def __init__(self, frames=10):
        self.frames = frames

    def isOpened(self):
        return True

    def read(self):
        if self.frames > 0:
            self.frames -= 1
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return True, frame
        return False, None

    def release(self):
        pass


@pytest.fixture
def patch_main_dependencies(monkeypatch):
    # Fake camera: return 10 frames so warmup progresses
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda *_: FakeCapture(frames=10))
    monkeypatch.setattr(main.cv2, "imshow", lambda *a, **kw: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *_: ord(" "))

    # Fake embedder
    fake_embedder = MagicMock()
    fake_embedder.get_embedding.return_value = np.zeros(16)
    monkeypatch.setattr(main, "ViTEmbedder", lambda: fake_embedder)

    # Fake detector
    fake_detector = MagicMock()
    fake_detector.trained = False
    def fake_update(_):
        fake_detector.trained = True
    fake_detector.update.side_effect = fake_update
    fake_detector.score.return_value = 0.4
    fake_detector.load.return_value = False
    monkeypatch.setattr(main, "OnlineAnomalyDetector", lambda *a, **kw: fake_detector)

    # Fake event manager
    fake_events = MagicMock()
    fake_events.threshold = 2.0  # 👈 ADD THIS
    monkeypatch.setattr(main, "EventManager", lambda *a, **kw: fake_events)  # 👈 AND THIS

    # Fake buffered writer
    fake_writer = MagicMock()
    monkeypatch.setattr(main, "BufferedClipWriter", lambda *a, **kw: fake_writer)

    return {
        "detector": fake_detector,
        "writer": fake_writer,
        "events": fake_events,
        "embedder": fake_embedder
    }

def test_main_executes_and_calls_detectors(patch_main_dependencies):
    deps = patch_main_dependencies

    main.main(warmup_target=2, max_frames=5, frame_skip=1)

    # Verify model warmup and score path executed
    assert deps["detector"].update.call_count >= 1
    assert deps["detector"].score.call_count >= 1
    assert deps["events"].update.call_count >= 1
    assert deps["writer"].add_frame.call_count >= 1

def test_periodic_learning_triggered(monkeypatch):
    """
    Ensures periodic learning executes when anomaly < 1.0 and frame_count % 30 == 0.
    """
    # Fake camera: ensure enough frames to hit frame_count % 30 == 0
    monkeypatch.setattr(main.cv2, "VideoCapture", lambda *_: FakeCapture(frames=40))
    monkeypatch.setattr(main.cv2, "imshow", lambda *a, **kw: None)
    monkeypatch.setattr(main.cv2, "destroyAllWindows", lambda: None)
    monkeypatch.setattr(main.cv2, "waitKey", lambda *_: ord(" "))

    # Fake embedder always returns a simple embedding
    fake_embedder = MagicMock()
    fake_embedder.get_embedding.return_value = np.zeros(16)
    monkeypatch.setattr(main, "ViTEmbedder", lambda: fake_embedder)

    # Fake detector: start as trained to skip warmup
    fake_detector = MagicMock()
    fake_detector.trained = True
    fake_detector.score.return_value = 0.5  # < 1.0 => triggers periodic update
    fake_detector.load.return_value = True  # Skip warmup path
    monkeypatch.setattr(main, "OnlineAnomalyDetector", lambda *a, **kw: fake_detector)

    fake_events = MagicMock()
    fake_events.threshold = 2.0
    monkeypatch.setattr(main, "EventManager", lambda *a, **kw: fake_events)

    fake_writer = MagicMock()
    monkeypatch.setattr(main, "BufferedClipWriter", lambda *a, **kw: fake_writer)

    # Run detection loop with enough frames
    main.main(warmup_target=0,   # No warmup
              max_frames=40,     # Ensure 30th frame hit
              frame_skip=1)      # Process every frame

    # ASSERT: periodic learning update() must be called at least once
    assert fake_detector.update.call_count >= 1