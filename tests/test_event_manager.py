# tests/test_event_manager.py
import json
import numpy as np
from pathlib import Path

from event_manager import EventManager


class FakeClipWriter:
    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        self.last_event_dir = None

    def save_event(self, event_id, future_frames):
        event_dir = self.base_dir / f"event_{event_id:03d}"
        event_dir.mkdir(parents=True, exist_ok=True)
        # We don't actually write video here; EventManager will write metadata.
        self.last_event_dir = event_dir
        return str(event_dir)


def test_event_lifecycle(tmp_events_dir, dummy_frame):
    manager = EventManager(threshold=2.0)
    writer = FakeClipWriter(tmp_events_dir)

    # Initially inactive
    assert manager.active is False
    assert manager.event_id == 0

    # Trigger start with high anomaly
    manager.update(anomaly_score=3.0, frame=dummy_frame, clip_writer=writer)
    assert manager.active is True
    assert len(manager.future_frames) == 1

    # Now send a sequence of "cool-down" frames below threshold
    for _ in range(manager.frames_after_end):
        manager.update(anomaly_score=1.0, frame=dummy_frame, clip_writer=writer)

    # After enough low-anomaly frames, event should close
    assert manager.active is False
    assert manager.event_id == 1
    assert writer.last_event_dir is not None

    # Metadata file should exist
    metadata_path = writer.last_event_dir / "metadata.json"
    assert metadata_path.exists()

    with open(metadata_path) as f:
        meta = json.load(f)
    assert meta["event_id"] == 1
    assert "timestamp" in meta