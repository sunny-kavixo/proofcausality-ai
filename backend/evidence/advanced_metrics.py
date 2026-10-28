"""Advanced mathematical and statistical metrics engine for ProofCausality AI.

Implements:
- Wilson score confidence intervals for binomial proportions.
- Bootstrap resampling confidence intervals.
- Cohen's Kappa for inter-rater annotation agreement.
- McNemar's test for paired binary evaluation outcomes.
- Comprehensive mask geometry: Dice coefficient, pixel Precision/Recall/F1,
  boundary F1 within tolerance, centroid displacement vectors, and relative area ratios.
"""

from __future__ import annotations

import math
from typing import Any, Callable
import numpy as np

from backend.evidence.schemas import BoundingBox


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    Args:
        k: Number of successes / positive counts (0 <= k <= n).
        n: Total number of trials / sample size (n >= 0).
        confidence: Confidence level (default: 0.95 for 95% CI).

    Returns:
        tuple[float, float]: Lower and upper bounds bounded in [0.0, 1.0].
    """
    if n <= 0:
        return (0.0, 0.0)
    if k < 0:
        k = 0
    if k > n:
        k = n

    # Standard normal quantile z (e.g. 1.95996 for 95% CI)
    # Using inverse error function approximation for standard values
    z_map = {0.90: 1.644853, 0.95: 1.959964, 0.99: 2.575829}
    z = z_map.get(confidence, 1.959964)

    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p_hat + z2 / (2.0 * n)) / denom
    margin = (z / denom) * math.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return (round(lower, 4), round(upper, 4))


def bootstrap_confidence_interval(
    data: np.ndarray | list[float],
    statistic_func: Callable[[np.ndarray], float],
    num_samples: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute non-parametric bootstrap confidence interval for a given statistic."""
    arr = np.asarray(data)
    n = len(arr)
    if n == 0:
        return (0.0, 0.0)

    rng = np.random.default_rng(seed)
    boot_stats = np.empty(num_samples, dtype=np.float64)

    for i in range(num_samples):
        resample = rng.choice(arr, size=n, replace=True)
        boot_stats[i] = statistic_func(resample)

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_stats, 100.0 * (alpha / 2.0)))
    upper = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return (round(lower, 4), round(upper, 4))


def cohen_kappa(y_true: list[str], y_pred: list[str], categories: list[str] | None = None) -> float:
    """Compute Cohen's Kappa coefficient for inter-annotator agreement."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0

    if categories is None:
        categories = sorted(list(set(y_true) | set(y_pred)))

    n = len(y_true)
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    num_cats = len(categories)

    conf_mat = np.zeros((num_cats, num_cats), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if yt in cat_to_idx and yp in cat_to_idx:
            conf_mat[cat_to_idx[yt], cat_to_idx[yp]] += 1

    p_observed = float(np.trace(conf_mat)) / float(n)

    sum_rows = conf_mat.sum(axis=1).astype(np.float64)
    sum_cols = conf_mat.sum(axis=0).astype(np.float64)
    p_expected = float(np.dot(sum_rows, sum_cols)) / float(n * n)

    if math.isclose(p_expected, 1.0):
        return 1.0

    kappa = (p_observed - p_expected) / (1.0 - p_expected)
    return round(float(kappa), 4)


def mcnemar_test(contingency_table: list[list[int]] | np.ndarray) -> dict[str, float]:
    """Compute McNemar's test for paired binary outcomes with continuity correction.

    Contingency table:
      [[Model A correct, Model B correct], [Model A correct, Model B incorrect (b)]]
      [[Model A incorrect, Model B correct (c)], [Model A incorrect, Model B incorrect]]
    """
    mat = np.asarray(contingency_table)
    b = int(mat[0, 1])
    c = int(mat[1, 0])

    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "discordant_pairs": 0}

    # Edwards continuity correction
    chi2 = float((abs(b - c) - 1.0) ** 2) / float(b + c)
    # 1 degree of freedom p-value via regularized incomplete gamma / chi2 cdf approximation
    # Using 1 - erf(sqrt(chi2 / 2))
    p_value = float(math.erfc(math.sqrt(chi2 / 2.0)))

    return {
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 6),
        "discordant_pairs": b + c,
    }


def calculate_dice_coefficient(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Sørensen-Dice coefficient: 2 * |A ∩ B| / (|A| + |B|)."""
    a_bool = mask_a.astype(bool)
    b_bool = mask_b.astype(bool)

    intersection = np.count_nonzero(a_bool & b_bool)
    sum_sizes = np.count_nonzero(a_bool) + np.count_nonzero(b_bool)

    if sum_sizes == 0:
        return 1.0
    return round(float(2.0 * intersection / sum_sizes), 4)


def calculate_pixel_precision_recall_f1(
    ground_truth_mask: np.ndarray,
    candidate_mask: np.ndarray,
) -> dict[str, float]:
    """Compute pixel-level Precision, Recall, and F1 score."""
    gt = ground_truth_mask.astype(bool)
    cand = candidate_mask.astype(bool)

    tp = np.count_nonzero(gt & cand)
    fp = np.count_nonzero(~gt & cand)
    fn = np.count_nonzero(gt & ~cand)

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0

    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
    }


def calculate_boundary_f1(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    tolerance_px: int = 2,
) -> dict[str, float]:
    """Compute Boundary F1 score within a spatial tolerance margin."""
    from backend.evidence.masks import extract_boundary

    b_a = extract_boundary(mask_a)
    b_b = extract_boundary(mask_b)

    pts_a = np.count_nonzero(b_a)
    pts_b = np.count_nonzero(b_b)

    if pts_a == 0 and pts_b == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if pts_a == 0 or pts_b == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    # Pure NumPy tolerance dilation kernel
    r = tolerance_px
    dilated_a = b_a.copy()
    dilated_b = b_b.copy()

    # Simple 2D window dilation without external scipy dependency
    h, w = b_a.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy == 0 and dx == 0:
                continue
            shifted_a = np.zeros_like(b_a)
            shifted_b = np.zeros_like(b_b)

            y1_s = max(0, dy)
            y2_s = min(h, h + dy)
            x1_s = max(0, dx)
            x2_s = min(w, w + dx)

            y1_d = max(0, -dy)
            y2_d = min(h, h - dy)
            x1_d = max(0, -dx)
            x2_d = min(w, w - dx)

            shifted_a[y1_s:y2_s, x1_s:x2_s] = b_a[y1_d:y2_d, x1_d:x2_d]
            shifted_b[y1_s:y2_s, x1_s:x2_s] = b_b[y1_d:y2_d, x1_d:x2_d]

            dilated_a |= shifted_a
            dilated_b |= shifted_b

    # Precision: fraction of candidate boundary points within tolerance of GT boundary
    # Recall: fraction of GT boundary points within tolerance of candidate boundary
    prec_matches = np.count_nonzero(b_b & dilated_a)
    rec_matches = np.count_nonzero(b_a & dilated_b)

    precision = float(prec_matches / pts_b) if pts_b > 0 else 1.0
    recall = float(rec_matches / pts_a) if pts_a > 0 else 1.0

    if precision + recall > 0:
        f1 = (2.0 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def calculate_centroid_displacement(bbox_a: BoundingBox, bbox_b: BoundingBox) -> dict[str, float]:
    """Compute directional centroid displacement vector and Euclidean distance."""
    cx_a = (bbox_a.x_min + bbox_a.x_max) / 2.0
    cy_a = (bbox_a.y_min + bbox_a.y_max) / 2.0

    cx_b = (bbox_b.x_min + bbox_b.x_max) / 2.0
    cy_b = (bbox_b.y_min + bbox_b.y_max) / 2.0

    dx = cx_b - cx_a
    dy = cy_b - cy_a
    dist = math.sqrt(dx * dx + dy * dy)

    return {
        "delta_x": round(dx, 2),
        "delta_y": round(dy, 2),
        "euclidean_distance": round(dist, 2),
        "centroid_a": (round(cx_a, 2), round(cy_a, 2)),
        "centroid_b": (round(cx_b, 2), round(cy_b, 2)),
    }


def calculate_relative_area_change(initial_mask: np.ndarray, final_mask: np.ndarray) -> float:
    """Calculate relative surface area change ratio: Final Area / Initial Area."""
    init_area = float(np.count_nonzero(initial_mask))
    final_area = float(np.count_nonzero(final_mask))

    if init_area == 0.0:
        return 0.0 if final_area == 0.0 else float("inf")

    return round(final_area / init_area, 4)
