import json
import os
import time

class EventManager:
    def __init__(self, threshold=2.0):
        self.threshold = threshold
        self.active = False
        self.event_id = 0
        self.future_frames = []
        self.frames_after_end = 10

    def update(self, anomaly_score, frame, clip_writer):
        if not self.active and anomaly_score > self.threshold:
            self.active = True
            self.future_frames = []
            print(f"EVENT START: score={anomaly_score:.2f}")

        if self.active:
            self.future_frames.append((frame.copy(), time.time()))

            if anomaly_score <= self.threshold:
                self.frames_after_end -= 1

            if self.frames_after_end <= 0:
                self._close_event(clip_writer)
                self.frames_after_end = 10
                self.active = False

    def _close_event(self, clip_writer):
        self.event_id += 1
        event_dir = clip_writer.save_event(self.event_id, self.future_frames)
        self._save_metadata(event_dir)
        print("EVENT FINISHED\n")

    def _save_metadata(self, event_dir):
        metadata = {
            "event_id": self.event_id,
            "timestamp": time.time(),
            "notes": "Anomaly detected event for review"
        }
        with open(f"{event_dir}/metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)