# tests/test_vit_embedder.py
import torch
import numpy as np
import pytest

import vit_embedder


class DummyModel(torch.nn.Module):
    def __init__(self, out_dim=16):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, x):
        # x: (B, C, H, W)
        batch_size = x.shape[0]
        return torch.zeros((batch_size, self.out_dim), dtype=torch.float32)


@pytest.fixture
def patched_vit(monkeypatch):
    """Patch timm.create_model to avoid heavy model downloads."""
    def _create_model(*args, **kwargs):
        return DummyModel(out_dim=16)

    monkeypatch.setattr(vit_embedder.timm, "create_model", _create_model)
    return vit_embedder.ViTEmbedder()


def test_preprocess_shape_dtype(patched_vit, dummy_frame):
    emb = patched_vit.preprocess(dummy_frame)
    # shape: (1, 3, 224, 224)
    assert emb.shape == (1, 3, patched_vit.input_size, patched_vit.input_size)
    assert isinstance(emb, torch.Tensor)
    assert emb.dtype == torch.float32


def test_get_embedding_returns_flat_numpy(patched_vit, dummy_frame):
    embedding = patched_vit.get_embedding(dummy_frame)
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    # DummyModel emits 16-dim
    assert embedding.shape[0] == 16