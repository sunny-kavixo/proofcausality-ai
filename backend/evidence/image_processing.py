"""Real image processing, COCO annotation parser, and visual overlay generator.

Provides:
- Strict binary magic-byte image validation (PNG, JPEG).
- Deterministic SHA-256 hashing of source images.
- COCO polygon rasterization to boolean binary masks.
- Color-coded disagreement heatmaps (Green = Agreement, Red = Missing/FN, Cyan = False Positive).
- Base64 data-URI encoding for web visualization.
"""

from __future__ import annotations

import base64
import hashlib
import io
from typing import Any
import numpy as np
from PIL import Image, ImageDraw

from backend.evidence.schemas import BoundingBox

class ImageValidationError(ValueError):
    """Raised when an uploaded image fails validation."""
    pass


# Prevent decompression bomb attacks
Image.MAX_IMAGE_PIXELS = 25_000_000



def validate_and_decode_image(
    file_bytes: bytes,
    max_size_bytes: int = 10 * 1024 * 1024,
    max_dim: int = 4096,
) -> tuple[np.ndarray, str, int, int]:
    """Validate image bytes, compute SHA-256 digest, and decode to RGB numpy array.

    Args:
        file_bytes: Raw bytes of the image file.
        max_size_bytes: Maximum allowed size (default: 10 MiB).
        max_dim: Maximum allowed width or height.

    Returns:
        tuple[np.ndarray, str, int, int]:
            - Decoded image as uint8 RGB array (height, width, 3).
            - Hexadecimal SHA-256 digest of original raw bytes.
            - Width in pixels.
            - Height in pixels.

    Raises:
        ImageValidationError: If format, dimensions, or size are invalid.
    """
    if not file_bytes:
        raise ImageValidationError("Image file is empty (0 bytes).")

    if len(file_bytes) > max_size_bytes:
        raise ImageValidationError(
            f"Image file size ({len(file_bytes)} bytes) exceeds maximum permitted limit ({max_size_bytes} bytes)."
        )

    # Magic byte inspection
    is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")

    if not (is_png or is_jpeg):
        raise ImageValidationError(
            "Unsupported image format. Only authentic PNG and JPEG images are permitted."
        )

    # Compute raw cryptographic hash
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()

    try:
        pil_img = Image.open(io.BytesIO(file_bytes))
        pil_img.verify()
        # Re-open after verify() as recommended by PIL
        pil_img = Image.open(io.BytesIO(file_bytes))
        width, height = pil_img.size
    except Exception as e:
        raise ImageValidationError(f"Malformed or corrupt image file: {e}") from e

    if width <= 0 or height <= 0:
        raise ImageValidationError(f"Invalid image dimensions: {width}x{height}.")

    if width > max_dim or height > max_dim:
        raise ImageValidationError(
            f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {max_dim}px."
        )

    rgb_img = pil_img.convert("RGB")
    rgb_arr = np.array(rgb_img, dtype=np.uint8)
    return rgb_arr, sha256_hash, width, height


def parse_coco_polygon_to_mask(
    segmentation: list[list[float]] | list[float],
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize COCO polygon segmentation format into a boolean binary mask.

    COCO format can be:
      - A list of polygons: [[x1, y1, x2, y2, ...], [x1, y1, ...]]
      - A single flattened polygon: [x1, y1, x2, y2, ...]
    """
    if width <= 0 or height <= 0:
        raise ImageValidationError(f"Invalid rasterization canvas dimensions: {width}x{height}")

    mask_img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    if not segmentation:
        return np.zeros((height, width), dtype=bool)

    # Normalize to list of polygons
    polygons: list[list[float]]
    if isinstance(segmentation[0], (int, float)):
        polygons = [segmentation]  # type: ignore[list-item]
    else:
        polygons = segmentation  # type: ignore[assignment]

    for poly in polygons:
        if len(poly) < 6:  # Need at least 3 points (x, y)
            continue
        # Convert flat [x1, y1, x2, y2, ...] to list of tuples [(x1, y1), (x2, y2), ...]
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]
        draw.polygon(pts, fill=1)

    return np.array(mask_img, dtype=bool)


def create_disagreement_visual_overlay(
    image_rgb: np.ndarray,
    human_mask: np.ndarray,
    candidate_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """Render a visual disagreement overlay on the input RGB image.

    Color coding:
      - Green ([0, 255, 64]): Concordance / Agreement (True Positive).
      - Red ([255, 48, 48]): Human ground-truth only (False Negative / Deficit).
      - Cyan ([0, 200, 255]): Candidate prediction only (False Positive / Excess).
    """
    h, w, c = image_rgb.shape
    gt = human_mask.astype(bool)
    cand = candidate_mask.astype(bool)

    overlay = image_rgb.astype(np.float32).copy()

    agreement = gt & cand
    human_only = gt & ~cand
    candidate_only = ~gt & cand

    # Blend colors
    if np.any(agreement):
        overlay[agreement] = (1.0 - alpha) * overlay[agreement] + alpha * np.array([0, 255, 64], dtype=np.float32)

    if np.any(human_only):
        overlay[human_only] = (1.0 - alpha) * overlay[human_only] + alpha * np.array([255, 48, 48], dtype=np.float32)

    if np.any(candidate_only):
        overlay[candidate_only] = (1.0 - alpha) * overlay[candidate_only] + alpha * np.array([0, 200, 255], dtype=np.float32)

    return np.clip(overlay, 0, 255).astype(np.uint8)


def create_single_mask_overlay(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    color_rgb: tuple[int, int, int] = (0, 200, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """Render a single highlighted mask overlay on the RGB image."""
    overlay = image_rgb.astype(np.float32).copy()
    m = mask.astype(bool)
    if np.any(m):
        color = np.array(color_rgb, dtype=np.float32)
        overlay[m] = (1.0 - alpha) * overlay[m] + alpha * color
    return np.clip(overlay, 0, 255).astype(np.uint8)



def encode_image_to_base64_png(image_array: np.ndarray) -> str:
    """Encode RGB or RGBA numpy array to a data:image/png;base64 string."""
    pil_img = Image.fromarray(image_array)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG", optimize=True)
    b64_bytes = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64_bytes}"
