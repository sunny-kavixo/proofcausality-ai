"""Image-grouped uncertainty utilities for correlated evaluation scenarios."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def assert_disjoint_groups(development_ids: Sequence[int], test_ids: Sequence[int]) -> None:
    overlap = set(development_ids) & set(test_ids)
    if overlap:
        raise ValueError(f"development/test image overlap: {sorted(overlap)}")


def grouped_bootstrap_interval(
    rows: Sequence[dict],
    statistic: Callable[[Sequence[dict]], float],
    *,
    group_key: str = "image_id",
    iterations: int = 2000,
    seed: int = 20261028,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Resample whole source-image clusters, preserving within-image correlation."""
    if not rows or iterations <= 0 or not 0 < confidence < 1:
        raise ValueError("rows, iterations, and confidence must define a valid bootstrap")
    groups: dict[object, list[dict]] = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    keys = list(groups)
    rng = np.random.default_rng(seed)
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        selected = rng.choice(len(keys), size=len(keys), replace=True)
        sample = [row for selected_index in selected for row in groups[keys[int(selected_index)]]]
        estimates[index] = statistic(sample)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return float(low), float(high)

