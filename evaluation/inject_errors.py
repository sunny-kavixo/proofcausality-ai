"""Controlled annotation error injection for LabelCourt auditing benchmarks."""

from __future__ import annotations

import copy
import random
from typing import List, Tuple
from backend.evidence.schemas import AnnotationItem, BoundingBox


def shift_annotation_bbox(item: AnnotationItem, dx_px: float, dy_px: float) -> AnnotationItem:
    """Inject a spatial shift into an annotation bounding box."""
    corrupted = copy.deepcopy(item)
    corrupted.bbox = BoundingBox(
        x_min=item.bbox.x_min + dx_px,
        y_min=item.bbox.y_min + dy_px,
        x_max=item.bbox.x_max + dx_px,
        y_max=item.bbox.y_max + dy_px,
    )
    if corrupted.segmentation:
        shifted_seg = []
        for poly in corrupted.segmentation:
            shifted_poly = []
            for i in range(0, len(poly), 2):
                shifted_poly.append(poly[i] + dx_px)
                shifted_poly.append(poly[i + 1] + dy_px)
            shifted_seg.append(shifted_poly)
        corrupted.segmentation = shifted_seg
    return corrupted


def swap_annotation_class(item: AnnotationItem, wrong_class: str, wrong_id: int = 999) -> AnnotationItem:
    """Inject a class labeling error."""
    corrupted = copy.deepcopy(item)
    corrupted.category_name = wrong_class
    corrupted.category_id = wrong_id
    return corrupted


def inject_boundary_erosion(item: AnnotationItem, shrink_ratio: float = 0.5) -> AnnotationItem:
    """Simulate severe boundary under-segmentation by shrinking the bounding box and mask."""
    corrupted = copy.deepcopy(item)
    cx, cy = item.bbox.centroid
    half_w = (item.bbox.width * shrink_ratio) / 2.0
    half_h = (item.bbox.height * shrink_ratio) / 2.0
    corrupted.bbox = BoundingBox(
        x_min=max(0.0, cx - half_w),
        y_min=max(0.0, cy - half_h),
        x_max=cx + half_w,
        y_max=cy + half_h,
    )
    if corrupted.segmentation:
        shrunk_seg = []
        for poly in corrupted.segmentation:
            shrunk_poly = []
            for i in range(0, len(poly), 2):
                px = cx + (poly[i] - cx) * shrink_ratio
                py = cy + (poly[i + 1] - cy) * shrink_ratio
                shrunk_poly.extend([px, py])
            shrunk_seg.append(shrunk_poly)
        corrupted.segmentation = shrunk_seg
    return corrupted
