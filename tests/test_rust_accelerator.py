import pytest
import numpy as np

from backend.evidence.rust_accelerator import (
    is_rust_accelerator_available,
    get_accelerator_info,
    compute_mask_iou_fast,
    compute_mask_dice_fast,
    compute_batch_mask_iou_fast,
    compute_laplacian_variance_fast,
)


def test_rust_accelerator_status():
    info = get_accelerator_info()
    assert "rust_accelerator_active" in info
    assert "backend_engine" in info


def test_mask_iou_and_dice_parity():
    rng = np.random.default_rng(123)
    m1 = rng.random((128, 128)) > 0.5
    m2 = rng.random((128, 128)) > 0.45

    iou = compute_mask_iou_fast(m1, m2)
    dice = compute_mask_dice_fast(m1, m2)

    inter = np.count_nonzero(m1 & m2)
    expected_iou = round(float(inter / np.count_nonzero(m1 | m2)), 4)
    expected_dice = round(float(2 * inter / (np.count_nonzero(m1) + np.count_nonzero(m2))), 4)

    assert iou == expected_iou
    assert dice == expected_dice


def test_empty_masks_graceful_handling():
    empty1 = np.zeros((64, 64), dtype=bool)
    empty2 = np.zeros((64, 64), dtype=bool)

    assert compute_mask_iou_fast(empty1, empty2) == 1.0
    assert compute_mask_dice_fast(empty1, empty2) == 1.0


def test_batch_mask_iou():
    rng = np.random.default_rng(999)
    b1 = rng.random((10, 64, 64)) > 0.5
    b2 = rng.random((10, 64, 64)) > 0.5

    batch_res = compute_batch_mask_iou_fast(b1, b2)
    assert len(batch_res) == 10
    for i in range(10):
        single_res = compute_mask_iou_fast(b1[i], b2[i])
        assert abs(batch_res[i] - single_res) < 1e-4


def test_laplacian_variance_parity():
    gray = np.random.default_rng(777).random((150, 150))
    var_fast = compute_laplacian_variance_fast(gray)

    lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    expected_var = round(float(np.var(lap)), 2)

    assert abs(var_fast - expected_var) <= 0.05
