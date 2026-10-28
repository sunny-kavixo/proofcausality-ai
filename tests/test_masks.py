import numpy as np
from backend.evidence.masks import (
    bbox_to_mask,
    extract_boundary,
    generate_overlay_heatmap,
    mask_centroid,
    mask_to_bbox,
    polygon_to_mask,
)
from backend.evidence.schemas import BoundingBox


def test_bbox_to_mask_and_inverse():
    bbox = BoundingBox(x_min=10, y_min=10, x_max=30, y_max=40)
    mask = bbox_to_mask(bbox, height=100, width=100)
    assert mask.shape == (100, 100)
    assert mask.sum() == 20 * 30  # 600

    recovered_bbox = mask_to_bbox(mask)
    assert recovered_bbox is not None
    assert recovered_bbox.x_min == 10
    assert recovered_bbox.y_min == 10
    assert recovered_bbox.x_max == 30
    assert recovered_bbox.y_max == 40


def test_polygon_to_mask():
    # Square polygon
    poly = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]
    mask = polygon_to_mask(poly, height=50, width=50)
    assert mask.shape == (50, 50)
    assert np.any(mask)
    assert bool(mask[20, 20]) is True
    assert bool(mask[0, 0]) is False


def test_extract_boundary():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    boundary = extract_boundary(mask)
    assert np.any(boundary)
    # Center pixel should not be on boundary
    assert bool(boundary[10, 10]) is False
    # Outer edge pixel should be on boundary
    assert bool(boundary[5, 5]) is True


def test_mask_centroid():
    mask = np.zeros((100, 100), dtype=bool)
    mask[20:40, 40:60] = True
    cx, cy = mask_centroid(mask)
    assert round(cx) == 50
    assert round(cy) == 30


def test_generate_overlay_heatmap():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[10:30, 10:30] = True
    m2[20:40, 20:40] = True
    b64 = generate_overlay_heatmap(m1, m2)
    assert isinstance(b64, str)
    assert len(b64) > 100
