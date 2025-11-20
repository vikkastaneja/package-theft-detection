import numpy as np
from pathlib import Path
from datetime import datetime

import anomaly_detector


def test_update_and_score_basic():
    det = anomaly_detector.OnlineAnomalyDetector(n_clusters=2)

    # Two tight clusters around 0 and 10
    cluster1 = np.zeros((20, 4), dtype=float)
    cluster2 = np.ones((20, 4), dtype=float) * 10.0
    X = np.vstack([cluster1, cluster2])

    det.update(X)
    assert det.trained is True

    # Near first cluster center
    near = np.array([0.1, -0.1, 0.05, 0.0])
    # Near second cluster center
    far = np.array([10.0, 9.9, 10.1, 10.0])

    score_near = det.score(near)
    score_far = det.score(far)

    # Both should be finite and low
    assert score_near >= 0
    assert score_far >= 0
    # Should be closer to one of the cluster centers than if we pick a random point
    random_point = np.array([5.0, 5.0, 5.0, 5.0])
    score_random = det.score(random_point)
    assert score_random > score_near
    assert score_random > score_far


def test_score_untrained_returns_zero():
    det = anomaly_detector.OnlineAnomalyDetector()
    emb = np.zeros(4, dtype=float)
    assert det.score(emb) == 0.0


def test_save_and_load(tmp_models_dir):
    det = anomaly_detector.OnlineAnomalyDetector(n_clusters=2)
    X = np.random.randn(50, 4)
    det.update(X)

    model_path = tmp_models_dir / "model_latest.pkl"
    det.save(str(model_path))
    assert model_path.exists()

    # New detector should be able to load
    det2 = anomaly_detector.OnlineAnomalyDetector(n_clusters=2)
    loaded = det2.load(str(model_path))
    assert loaded is True
    assert det2.trained is True
    assert det2.model.n_clusters == det.model.n_clusters

    # Cluster centers should be similar
    assert det2.model.cluster_centers_.shape == det.model.cluster_centers_.shape


def test_save_checkpoint_creates_timestamped_file(tmp_models_dir, monkeypatch):
    det = anomaly_detector.OnlineAnomalyDetector(model_dir=str(tmp_models_dir))
    X = np.random.randn(50, 4)
    det.update(X)

    det.save_checkpoint()
    # Check inside tmp_models_dir
    files = list(Path(tmp_models_dir).glob("model_*.pkl"))
    assert len(files) >= 1

def test_default_save_and_load(tmp_path):
    model_dir = tmp_path / "models"

    det = anomaly_detector.OnlineAnomalyDetector(model_dir=str(model_dir))
    X = np.random.randn(40, 4)
    det.update(X)

    # Save using default path
    saved_path = det.save()
    assert Path(saved_path).exists()

    # Load using default path
    det2 = anomaly_detector.OnlineAnomalyDetector(model_dir=str(model_dir))
    assert det2.load() is True
    assert det2.trained is True
    assert det2.model.cluster_centers_.shape == det.model.cluster_centers_.shape


def test_load_returns_false_when_missing(tmp_path):
    missing_dir = tmp_path / "empty_models"
    missing_dir.mkdir()

    det = anomaly_detector.OnlineAnomalyDetector(model_dir=str(missing_dir))
    assert det.load() is False
    assert det.trained is False