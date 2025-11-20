import collections
import cv2
import time
import os

class BufferedClipWriter:
    def __init__(self, buffer_seconds=3, fps=5, output_dir="../data/events"):
        self.buffer_size = buffer_seconds * fps
        self.buffer = collections.deque(maxlen=self.buffer_size)
        self.fps = fps
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def add_frame(self, frame):
        timestamp = time.time()
        self.buffer.append((frame.copy(), timestamp))

    def save_event(self, event_id, future_frames):
        event_dir = f"{self.output_dir}/event_{event_id:03d}"
        os.makedirs(event_dir, exist_ok=True)

        # Save video clip
        clip_path = f"{event_dir}/clip.mp4"

        h, w, _ = self.buffer[0][0].shape
        out = cv2.VideoWriter(clip_path,
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              self.fps,
                              (w, h))

        for f, _ in list(self.buffer) + future_frames:
            out.write(f)
        out.release()

        # Save thumbnails: before / during / after
        cv2.imwrite(f"{event_dir}/before.jpg", self.buffer[0][0])
        cv2.imwrite(f"{event_dir}/during.jpg", future_frames[len(future_frames)//2][0])
        cv2.imwrite(f"{event_dir}/after.jpg", future_frames[-1][0])

        print(f"[EVENT SAVED] {clip_path}")
        return event_dir