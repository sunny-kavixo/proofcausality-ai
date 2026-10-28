"""Deterministic bounded detectors for visual uncertainty signals."""

from __future__ import annotations

from typing import List, Optional
import numpy as np

from backend.evidence.schemas import UncertaintyReason, UncertaintySignal


def detect_blur_laplacian(image_gray: np.ndarray, threshold: float = 100.0) -> tuple[bool, float]:
    """Pure NumPy 2D Laplacian variance blur detector.
    
    Computes convolution with 3x3 Laplacian kernel [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    and returns (is_blurry, variance).
    """
    if image_gray.ndim != 2 or image_gray.shape[0] < 3 or image_gray.shape[1] < 3:
        return False, 999.0

    img = image_gray.astype(np.float64)
    # 2D Laplacian discrete kernel
    laplacian = (
        img[:-2, 1:-1] +
        img[2:, 1:-1] +
        img[1:-1, :-2] +
        img[1:-1, 2:] -
        4.0 * img[1:-1, 1:-1]
    )
    variance = float(np.var(laplacian))
    is_blurry = variance < threshold
    return is_blurry, variance


def detect_boundary_contact(mask: np.ndarray, margin_px: int = 3) -> tuple[bool, float]:
    """Detect if mask touches or approaches image borders (Framing / Out-of-frame risk)."""
    if not np.any(mask):
        return False, 0.0

    h, w = mask.shape
    top_contact = np.any(mask[:margin_px, :])
    bottom_contact = np.any(mask[h - margin_px:, :])
    left_contact = np.any(mask[:, :margin_px])
    right_contact = np.any(mask[:, w - margin_px:])

    touches = top_contact or bottom_contact or left_contact or right_contact
    border_pixels = (
        int(np.sum(mask[:margin_px, :])) +
        int(np.sum(mask[h - margin_px:, :])) +
        int(np.sum(mask[:, :margin_px])) +
        int(np.sum(mask[:, w - margin_px:]))
    )
    return touches, float(border_pixels)


def detect_occlusion_area_drop(
    before_mask: np.ndarray,
    after_mask: np.ndarray,
    drop_threshold: float = 0.35,
) -> tuple[bool, float]:
    """Detect sudden drop in visible mask area between consecutive observations (Occlusion risk)."""
    area_before = float(np.sum(before_mask))
    area_after = float(np.sum(after_mask))
    if area_before == 0:
        return False, 0.0

    ratio = area_after / area_before
    is_occluded = ratio < (1.0 - drop_threshold)
    return is_occluded, ratio


def detect_missing_temporal_state(
    has_final_frame: bool,
    time_gap_seconds: float = 0.0,
    max_permitted_gap: float = 8.0,
) -> tuple[bool, float]:
    """Detect missing observation after action or excessive time gap."""
    if not has_final_frame:
        return True, 0.0
    is_gap_excessive = time_gap_seconds > max_permitted_gap
    return is_gap_excessive, time_gap_seconds


def evaluate_all_uncertainty_signals(
    image_gray: Optional[np.ndarray] = None,
    current_mask: Optional[np.ndarray] = None,
    previous_mask: Optional[np.ndarray] = None,
    has_final_state: bool = True,
    blur_threshold: float = 100.0,
    boundary_margin_px: int = 3,
    occlusion_drop_threshold: float = 0.35,
) -> List[UncertaintySignal]:
    """Evaluate all deterministic uncertainty signals and return detected warnings."""
    signals: List[UncertaintySignal] = []

    # 1. Blur detection
    if image_gray is not None:
        is_blur, var = detect_blur_laplacian(image_gray, threshold=blur_threshold)
        if is_blur:
            signals.append(
                UncertaintySignal(
                    reason=UncertaintyReason.BLUR,
                    severity="HIGH" if var < (blur_threshold * 0.5) else "MEDIUM",
                    metric_name="laplacian_variance",
                    metric_value=round(var, 2),
                    threshold=blur_threshold,
                    description=f"Image exhibits high motion blur (Laplacian variance {var:.1f} < threshold {blur_threshold}).",
                )
            )

    # 2. Boundary contact (out-of-frame)
    if current_mask is not None:
        touches_border, border_px = detect_boundary_contact(current_mask, margin_px=boundary_margin_px)
        if touches_border:
            signals.append(
                UncertaintySignal(
                    reason=UncertaintyReason.OUT_OF_FRAME,
                    severity="HIGH" if border_px > 30 else "MEDIUM",
                    metric_name="border_touching_pixels",
                    metric_value=float(border_px),
                    threshold=1.0,
                    description=f"Object mask contacts frame edge ({int(border_px)} pixels within {boundary_margin_px}px of border); partial clipping risk.",
                )
            )

    # 3. Occlusion area drop
    if current_mask is not None and previous_mask is not None:
        is_occluded, ratio = detect_occlusion_area_drop(
            previous_mask, current_mask, drop_threshold=occlusion_drop_threshold
        )
        if is_occluded:
            signals.append(
                UncertaintySignal(
                    reason=UncertaintyReason.OCCLUSION,
                    severity="HIGH" if ratio < 0.4 else "MEDIUM",
                    metric_name="area_retention_ratio",
                    metric_value=round(ratio, 3),
                    threshold=round(1.0 - occlusion_drop_threshold, 3),
                    description=f"Severe area reduction ({ratio * 100:.1f}% remaining); object is likely partially obscured or occluded.",
                )
            )

    # 4. Missing final state
    if not has_final_state:
        signals.append(
            UncertaintySignal(
                reason=UncertaintyReason.MISSING_FINAL_STATE,
                severity="HIGH",
                metric_name="final_state_observed",
                metric_value=0.0,
                threshold=1.0,
                description="Recording or observation terminates before action conclusion is visible.",
            )
        )

    return signals
