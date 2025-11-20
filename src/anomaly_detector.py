import numpy as np
from sklearn.cluster import MiniBatchKMeans
import joblib
import time
import os
from datetime import datetime

class OnlineAnomalyDetector:
    def __init__(self, n_clusters=8, model_dir="data/models"):
        self.model = MiniBatchKMeans(n_clusters=n_clusters, n_init=3)
        self.trained = False
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _default_model_path(self):
        return os.path.join(self.model_dir, "model_latest.pkl")

    def update(self, embeddings):
        """Incrementally fit the clustering model."""
        self.model.partial_fit(embeddings)
        self.trained = True

    def score(self, emb):
        if not self.trained:
            return 0.0  # no anomaly until model has learned something

        cluster_id = self.model.predict(emb.reshape(1, -1))[0]
        center = self.model.cluster_centers_[cluster_id]
        dist = np.linalg.norm(emb - center)

        return dist  # Instead of a probability, anomaly = distance from center

    def save(self, path = None):
        if path is None:
            path = self._default_model_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        return path

    def load(self, path = None):
        if path is None:
            path = self._default_model_path()
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.trained = True
            print(f"[Model Loaded] {path}")
            return True
        return False

    def save_checkpoint(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"model_{timestamp}.pkl"
        path = os.path.join(self.model_dir, filename)
        os.makedirs(self.model_dir, exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[Checkpoint Saved] {filename}")
        return path