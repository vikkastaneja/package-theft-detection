import sys
from pathlib import Path

import numpy as np
import pytest



# Make `src` importable when running `pytest` from project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

import main

@pytest.fixture(autouse=True)
def force_fake_camera(monkeypatch):
    """
    Global test fixture:
    - Ensures no test ever tries to use the real Raspberry Pi camera.
    - Makes main.is_raspberry_pi() effectively false for all tests.
    """
    monkeypatch.setattr(main, "FORCE_FAKE_CAMERA", True)

@pytest.fixture
def dummy_frame():
    """Return a simple black frame (HWC, uint8)."""
    return np.zeros((240, 320, 3), dtype=np.uint8)


@pytest.fixture
def tmp_models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def tmp_events_dir(tmp_path):
    d = tmp_path / "events"
    d.mkdir()
    return d