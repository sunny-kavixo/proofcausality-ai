"""Spatial geometric relations between objects and bounding regions."""

from __future__ import annotations

import math
from backend.evidence.schemas import BoundingBox


def compute_centroid_distance(bbox_a: BoundingBox, bbox_b: BoundingBox) -> float:
    """Euclidean distance between bounding box centroids."""
    ca = bbox_a.centroid
    cb = bbox_b.centroid
    return math.hypot(ca[0] - cb[0], ca[1] - cb[1])


def is_contained_in_region(bbox: BoundingBox, region: BoundingBox, tolerance_ratio: float = 0.25) -> bool:
    """Check if bbox is largely contained within a designated target region."""
    # Intersection coordinates
    ix_min = max(bbox.x_min, region.x_min)
    iy_min = max(bbox.y_min, region.y_min)
    ix_max = min(bbox.x_max, region.x_max)
    iy_max = min(bbox.y_max, region.y_max)

    if ix_max <= ix_min or iy_max <= iy_min:
        return False

    intersection_area = (ix_max - ix_min) * (iy_max - iy_min)
    coverage = intersection_area / max(1.0, bbox.area)
    return coverage >= (1.0 - tolerance_ratio)
