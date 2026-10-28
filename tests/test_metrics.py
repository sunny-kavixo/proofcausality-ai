import numpy as np
from backend.evidence.metrics import (
    calculate_area_ratio,
    calculate_bbox_displacement,
    calculate_boundary_agreement,
    calculate_mask_iou,
)
from backend.evidence.schemas import BoundingBox


def test_mask_iou_identical():
    m = np.zeros((50, 50), dtype=bool)
    m[10:30, 10:30] = True
    assert calculate_mask_iou(m, m) == 1.0


def test_mask_iou_disjoint():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[0:10, 0:10] = True
    m2[20:30, 20:30] = True
    assert calculate_mask_iou(m1, m2) == 0.0


def test_mask_iou_partial():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[10:30, 10:30] = True  # area 400
    m2[20:40, 10:30] = True  # area 400, overlap 200
    # union = 400 + 400 - 200 = 600
    # iou = 200 / 600 = 0.3333...
    assert round(calculate_mask_iou(m1, m2), 3) == 0.333


def test_boundary_agreement():
    m1 = np.zeros((60, 60), dtype=bool)
    m2 = np.zeros((60, 60), dtype=bool)
    m1[10:40, 10:40] = True
    m2[11:41, 11:41] = True  # 1px shift, well within tolerance
    agreement = calculate_boundary_agreement(m1, m2, tolerance_px=2)
    assert agreement > 0.85


def test_bbox_displacement():
    b1 = BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10)
    b2 = BoundingBox(x_min=30, y_min=40, x_max=40, y_max=50)
    # c1 = (5, 5), c2 = (35, 45) -> dx=30, dy=40 -> dist = 50
    disp = calculate_bbox_displacement(b1, b2, image_diag=100.0)
    assert round(disp, 2) == 0.50


def test_area_ratio():
    m1 = np.ones((10, 10), dtype=bool)  # 100
    m2 = np.ones((5, 10), dtype=bool)   # 50
    assert calculate_area_ratio(m1, m2) == 0.50
