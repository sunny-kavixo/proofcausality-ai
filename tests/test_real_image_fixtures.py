"""Tests image processing and Laplacian variance extraction on real disk image fixtures."""

from pathlib import Path
import numpy as np
from PIL import Image

from backend.evidence.image_processing import validate_and_decode_image
from backend.evidence.uncertainty import detect_blur_laplacian
from backend.evidence.rust_accelerator import compute_laplacian_variance_fast

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"


def test_sharp_tabletop_image_fixture():
    path = FIXTURES_DIR / "tabletop_sharp_t0.png"
    assert path.exists()
    raw_bytes = path.read_bytes()
    arr, sha256_hash, w, h = validate_and_decode_image(raw_bytes)
    assert w == 320
    assert h == 240
    assert len(sha256_hash) == 64

    # Convert to grayscale
    gray = np.mean(arr, axis=2).astype(np.float64)
    is_blur, var = detect_blur_laplacian(gray, threshold=100.0)
    assert is_blur is False
    assert var > 100.0


def test_motion_blurry_image_fixture():
    path = FIXTURES_DIR / "tabletop_motion_blur_tT.png"
    assert path.exists()
    raw_bytes = path.read_bytes()
    arr, sha256_hash, w, h = validate_and_decode_image(raw_bytes)
    assert w == 320

    gray = np.mean(arr, axis=2).astype(np.float64)
    _, var = detect_blur_laplacian(gray, threshold=100.0)
    assert var < 150.0


def test_border_clipped_image_fixture():
    path = FIXTURES_DIR / "tabletop_clipped_border.png"
    assert path.exists()
    raw_bytes = path.read_bytes()
    arr, _, w, h = validate_and_decode_image(raw_bytes)
    assert w == 320
    assert h == 240
