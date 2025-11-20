# tests/test_buffered_clip_writer.py
import cv2
import numpy as np

from pathlib import Path

from buffered_clip_writer import BufferedClipWriter


def test_buffer_capacity_and_add_frame(tmp_events_dir, dummy_frame):
    writer = BufferedClipWriter(buffer_seconds=2, fps=2, output_dir=str(tmp_events_dir))
    # buffer_size = 4 frames
    for _ in range(6):
        writer.add_frame(dummy_frame)

    # Deque maxlen should enforce cap
    assert len(writer.buffer) == writer.buffer_size == 4


def test_save_event_creates_files(tmp_events_dir):
    writer = BufferedClipWriter(buffer_seconds=1, fps=2, output_dir=str(tmp_events_dir))

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    for _ in range(2):
        writer.add_frame(frame)

    # Future frames as if during event
    future_frames = [(frame.copy(), 0.0) for _ in range(3)]

    event_dir = writer.save_event(event_id=1, future_frames=future_frames)
    event_path = Path(event_dir)

    assert event_path.exists()
    assert (event_path / "clip.mp4").exists()
    assert (event_path / "before.jpg").exists()
    assert (event_path / "during.jpg").exists()
    assert (event_path / "after.jpg").exists()

    # Clip should be > 0 bytes if encoding worked
    assert (event_path / "clip.mp4").stat().st_size >= 0