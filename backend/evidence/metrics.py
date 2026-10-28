"""Deterministic spatial agreement and geometric discrepancy metrics."""

from __future__ import annotations

import math
from typing import Tuple
import numpy as np

from backend.evidence.schemas import BoundingBox
from backend.evidence.masks import extract_boundary, mask_centroid


def calculate_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Calculate exact Intersection-over-Union (Jaccard Index) between two binary masks."""
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection) / float(union)


def calculate_boundary_agreement(mask_a: np.ndarray, mask_b: np.ndarray, tolerance_px: int = 3) -> float:
    """Calculate boundary agreement: fraction of boundary A pixels within tolerance of boundary B, and vice-versa."""
    b_a = extract_boundary(mask_a)
    b_b = extract_boundary(mask_b)

    pts_a = np.argwhere(b_a)  # [y, x]
    pts_b = np.argwhere(b_b)

    if len(pts_a) == 0 and len(pts_b) == 0:
        return 1.0
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0

    # For bounded performance without scipy: compute distance from sample points
    # Sample up to 300 points for efficiency
    if len(pts_a) > 300:
        indices = np.linspace(0, len(pts_a) - 1, 300, dtype=int)
        pts_a_sample = pts_a[indices]
    else:
        pts_a_sample = pts_a

    if len(pts_b) > 300:
        indices = np.linspace(0, len(pts_b) - 1, 300, dtype=int)
        pts_b_sample = pts_b[indices]
    else:
        pts_b_sample = pts_b

    # Distance matrix A to B
    diff = pts_a_sample[:, np.newaxis, :] - pts_b_sample[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=-1)
    min_dist_a = np.sqrt(dist_sq.min(axis=1))
    prec = np.mean(min_dist_a <= tolerance_px)

    # Distance matrix B to A
    min_dist_b = np.sqrt(dist_sq.min(axis=0))
    rec = np.mean(min_dist_b <= tolerance_px)

    if (prec + rec) == 0:
        return 0.0
    return float(2 * (prec * rec) / (prec + rec))


def calculate_bbox_displacement(bbox_a: BoundingBox, bbox_b: BoundingBox, image_diag: float = 1000.0) -> float:
    """Euclidean distance between bounding box centroids normalized by image diagonal."""
    c_a = bbox_a.centroid
    c_b = bbox_b.centroid
    dist = math.sqrt((c_a[0] - c_b[0]) ** 2 + (c_a[1] - c_b[1]) ** 2)
    return min(1.0, dist / max(1.0, image_diag))


def calculate_area_ratio(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Ratio of smaller area over larger area, in [0.0, 1.0]."""
    area_a = float(np.sum(mask_a))
    area_b = float(np.sum(mask_b))
    if area_a == 0 and area_b == 0:
        return 1.0
    if area_a == 0 or area_b == 0:
        return 0.0
    return min(area_a, area_b) / max(area_a, area_b)
