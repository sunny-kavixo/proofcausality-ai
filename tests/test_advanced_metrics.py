import pytest
import numpy as np

from backend.evidence.advanced_metrics import (
    wilson_score_interval,
    bootstrap_confidence_interval,
    cohen_kappa,
    mcnemar_test,
    calculate_dice_coefficient,
    calculate_pixel_precision_recall_f1,
    calculate_boundary_f1,
    calculate_centroid_displacement,
    calculate_relative_area_change,
)
from backend.evidence.schemas import BoundingBox


def test_wilson_score_interval():
    # 0 out of 15 (e.g. 0% FSR)
    low, high = wilson_score_interval(0, 15, confidence=0.95)
    assert low == 0.0
    assert 0.15 <= high <= 0.25

    # 15 out of 15 (e.g. 100% RRR)
    low_full, high_full = wilson_score_interval(15, 15, confidence=0.95)
    assert 0.75 <= low_full <= 0.85
    assert high_full == 1.0

    # n = 0
    assert wilson_score_interval(0, 0) == (0.0, 0.0)

    # 50 out of 100
    low_50, high_50 = wilson_score_interval(50, 100)
    assert 0.39 <= low_50 <= 0.42
    assert 0.58 <= high_50 <= 0.61


def test_bootstrap_confidence_interval():
    data = [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
    low, high = bootstrap_confidence_interval(data, np.mean, num_samples=500, seed=42)
    assert 0.3 <= low <= high <= 1.0


def test_cohen_kappa():
    # Perfect agreement
    y1 = ["ACCEPT", "REVIEW", "ACCEPT", "UNCERTAIN"]
    y2 = ["ACCEPT", "REVIEW", "ACCEPT", "UNCERTAIN"]
    assert cohen_kappa(y1, y2) == 1.0

    # Partial disagreement
    y3 = ["ACCEPT", "ACCEPT", "REVIEW", "UNCERTAIN"]
    kappa = cohen_kappa(y1, y3)
    assert 0.0 < kappa < 1.0


def test_mcnemar_test():
    table = [[10, 10], [1, 4]]
    res = mcnemar_test(table)
    assert res["discordant_pairs"] == 11
    assert res["chi2"] > 0
    assert res["p_value"] < 0.05


def test_dice_and_pixel_metrics():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[10:30, 10:30] = True
    m2[10:30, 10:30] = True

    assert calculate_dice_coefficient(m1, m2) == 1.0
    scores = calculate_pixel_precision_recall_f1(m1, m2)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
    assert scores["f1"] == 1.0


def test_boundary_f1():
    m1 = np.zeros((50, 50), dtype=bool)
    m2 = np.zeros((50, 50), dtype=bool)
    m1[10:30, 10:30] = True
    m2[11:31, 11:31] = True

    bf1 = calculate_boundary_f1(m1, m2, tolerance_px=2)
    assert bf1["f1"] == 1.0


def test_centroid_displacement_and_area():
    b1 = BoundingBox(x_min=10, y_min=10, x_max=30, y_max=30)
    b2 = BoundingBox(x_min=40, y_min=50, x_max=60, y_max=70)
    disp = calculate_centroid_displacement(b1, b2)
    assert disp["delta_x"] == 30.0
    assert disp["delta_y"] == 40.0
    assert disp["euclidean_distance"] == 50.0

    m_init = np.zeros((50, 50), dtype=bool)
    m_final = np.zeros((50, 50), dtype=bool)
    m_init[0:10, 0:10] = True
    m_final[0:10, 0:5] = True
    assert calculate_relative_area_change(m_init, m_final) == 0.5
