import numpy as np
from backend.evidence.uncertainty import (
    detect_blur_laplacian,
    detect_boundary_contact,
    detect_missing_temporal_state,
    detect_occlusion_area_drop,
    evaluate_all_uncertainty_signals,
)


def test_blur_detector_sharp_vs_flat():
    # Flat image has 0 variance (blurry)
    flat = np.full((100, 100), 128, dtype=np.uint8)
    is_blur, var = detect_blur_laplacian(flat, threshold=100.0)
    assert is_blur is True
    assert var == 0.0

    # High frequency checkerboard has high variance
    sharp = np.zeros((100, 100), dtype=np.uint8)
    sharp[::2, ::2] = 255
    is_blur_sharp, var_sharp = detect_blur_laplacian(sharp, threshold=100.0)
    assert is_blur_sharp is False
    assert var_sharp > 500.0


def test_boundary_contact_detector():
    mask = np.zeros((100, 100), dtype=bool)
    mask[50:60, 50:60] = True  # Inside center
    touches, px = detect_boundary_contact(mask, margin_px=3)
    assert bool(touches) is False

    mask[0:10, 50:60] = True  # Touches top border
    touches_border, px_border = detect_boundary_contact(mask, margin_px=3)
    assert bool(touches_border) is True
    assert px_border > 0


def test_occlusion_area_drop():
    m_before = np.ones((40, 40), dtype=bool)  # 1600
    m_after = np.ones((20, 20), dtype=bool)   # 400 -> ratio 0.25 (dropped 75%)
    is_occ, ratio = detect_occlusion_area_drop(m_before, m_after, drop_threshold=0.35)
    assert is_occ is True
    assert ratio == 0.25


def test_missing_temporal_state():
    is_missing, _ = detect_missing_temporal_state(has_final_frame=False)
    assert is_missing is True
    is_ok, _ = detect_missing_temporal_state(has_final_frame=True, time_gap_seconds=1.5)
    assert is_ok is False
