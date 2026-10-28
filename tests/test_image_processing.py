import io
import pytest
import numpy as np
from PIL import Image

from backend.evidence.image_processing import (
    validate_and_decode_image,
    parse_coco_polygon_to_mask,
    create_disagreement_visual_overlay,
    encode_image_to_base64_png,
    ImageValidationError,
)


def _create_sample_png_bytes(width=100, height=100, color=(200, 100, 50)):
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_validate_and_decode_image_valid_png():
    raw_bytes = _create_sample_png_bytes(80, 60)
    arr, sha256_hash, w, h = validate_and_decode_image(raw_bytes)
    assert arr.shape == (60, 80, 3)
    assert w == 80
    assert h == 60
    assert len(sha256_hash) == 64


def test_validate_and_decode_image_rejection_cases():
    with pytest.raises(ImageValidationError, match="empty"):
        validate_and_decode_image(b"")

    with pytest.raises(ImageValidationError, match="Unsupported image format"):
        validate_and_decode_image(b"This is just plain text, not an image.")

    small_limit = 50
    raw_bytes = _create_sample_png_bytes(100, 100)
    with pytest.raises(ImageValidationError, match="exceeds maximum"):
        validate_and_decode_image(raw_bytes, max_size_bytes=small_limit)


def test_parse_coco_polygon_to_mask():
    poly = [[10.0, 10.0, 50.0, 10.0, 10.0, 50.0]]
    mask = parse_coco_polygon_to_mask(poly, width=100, height=100)
    assert mask.shape == (100, 100)
    assert bool(mask[15, 15]) is True
    assert bool(mask[80, 80]) is False


def test_create_disagreement_visual_overlay():
    img = np.zeros((60, 60, 3), dtype=np.uint8)
    gt = np.zeros((60, 60), dtype=bool)
    cand = np.zeros((60, 60), dtype=bool)

    gt[10:30, 10:30] = True
    cand[20:40, 20:40] = True

    overlay = create_disagreement_visual_overlay(img, gt, cand)
    assert overlay.shape == (60, 60, 3)
    assert overlay[25, 25, 1] > 0
    assert overlay[15, 15, 0] > 0

    b64_str = encode_image_to_base64_png(overlay)
    assert b64_str.startswith("data:image/png;base64,")
