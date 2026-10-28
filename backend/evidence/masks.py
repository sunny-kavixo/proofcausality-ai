"""Pure NumPy and Pillow mask processing operations."""

from __future__ import annotations

import base64
import io
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw

from backend.evidence.schemas import BoundingBox


def polygon_to_mask(polygon_pts: List[float], height: int, width: int) -> np.ndarray:
    """Rasterize a flat polygon [x1, y1, x2, y2, ...] into a binary boolean NumPy mask."""
    if not polygon_pts or len(polygon_pts) < 6:
        return np.zeros((height, width), dtype=bool)

    img = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(img)
    points = [(polygon_pts[i], polygon_pts[i + 1]) for i in range(0, len(polygon_pts), 2)]
    draw.polygon(points, outline=1, fill=1)
    return np.array(img, dtype=bool)


def bbox_to_mask(bbox: BoundingBox, height: int, width: int) -> np.ndarray:
    """Rasterize a bounding box into a binary boolean NumPy mask."""
    mask = np.zeros((height, width), dtype=bool)
    x0 = max(0, min(width - 1, int(round(bbox.x_min))))
    x1 = max(0, min(width, int(round(bbox.x_max))))
    y0 = max(0, min(height - 1, int(round(bbox.y_min))))
    y1 = max(0, min(height, int(round(bbox.y_max))))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = True
    return mask


def mask_to_bbox(mask: np.ndarray) -> Optional[BoundingBox]:
    """Calculate the minimal bounding box enclosing all True pixels in a binary mask."""
    if not np.any(mask):
        return None
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return BoundingBox(
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_max + 1),
        y_max=float(y_max + 1),
    )


def extract_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract 1-pixel boundary of a binary mask using NumPy neighbor difference."""
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)

    # 4-connectivity erosion
    eroded = np.copy(mask)
    eroded[1:, :] &= mask[:-1, :]
    eroded[:-1, :] &= mask[1:, :]
    eroded[:, 1:] &= mask[:, :-1]
    eroded[:, :-1] &= mask[:, 1:]

    return mask & (~eroded)


def mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    """Compute the spatial centroid (x, y) of True pixels."""
    if not np.any(mask):
        return (0.0, 0.0)
    y_indices, x_indices = np.where(mask)
    return (float(np.mean(x_indices)), float(np.mean(y_indices)))


def generate_overlay_heatmap(
    human_mask: np.ndarray,
    ai_mask: np.ndarray,
    background_rgb: Optional[np.ndarray] = None,
) -> str:
    """Generate a color-coded base64 PNG heatmap showing agreement and discrepancies.
    
    Colors:
    - Green: Agreement (Intersection)
    - Red: Human-only (AI missed / False Negative)
    - Blue: AI-only (Human missed / False Positive)
    """
    height, width = human_mask.shape
    if background_rgb is not None:
        canvas = np.copy(background_rgb).astype(np.float32) * 0.5
    else:
        canvas = np.full((height, width, 3), 32, dtype=np.float32)

    intersection = human_mask & ai_mask
    human_only = human_mask & (~ai_mask)
    ai_only = ai_mask & (~human_mask)

    # Green for agreement
    canvas[intersection] = canvas[intersection] * 0.3 + np.array([46, 204, 113], dtype=np.float32) * 0.7
    # Red for human only
    canvas[human_only] = canvas[human_only] * 0.3 + np.array([231, 76, 60], dtype=np.float32) * 0.7
    # Blue for AI only
    canvas[ai_only] = canvas[ai_only] * 0.3 + np.array([52, 152, 219], dtype=np.float32) * 0.7

    clipped = np.clip(canvas, 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(clipped)

    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
