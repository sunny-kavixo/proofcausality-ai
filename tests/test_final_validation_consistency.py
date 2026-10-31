"""Automated regression tests ensuring benchmark consistency across all evaluation artifacts.

This test suite guarantees that all reported metrics (IoU 0.50 and 0.75 matches,
scale breakdowns, recall, precision, and receipt verification) mathematically
reconcile and strictly match the authoritative machine-readable artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from backend.evidence.receipts import verify_receipt_integrity
from backend.evidence.schemas import EvidenceReceipt

EVAL_DIR = Path(__file__).resolve().parent.parent / "evaluation" / "experiments" / "final_validation"
SUMMARY_FILE = EVAL_DIR / "summary.json"
PER_GT_FILE = EVAL_DIR / "per_ground_truth.json"
MANIFEST_FILE = EVAL_DIR / "dataset_manifest.json"
RAW_PREDS_FILE = EVAL_DIR / "raw_predictions.json"


@pytest.fixture(scope="module")
def summary_data():
    assert SUMMARY_FILE.exists(), f"Missing required summary file: {SUMMARY_FILE}"
    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def per_gt_rows():
    assert PER_GT_FILE.exists(), f"Missing required per_ground_truth file: {PER_GT_FILE}"
    with open(PER_GT_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def raw_predictions():
    assert RAW_PREDS_FILE.exists(), f"Missing required raw_predictions file: {RAW_PREDS_FILE}"
    with open(RAW_PREDS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def test_ground_truth_and_prediction_totals(summary_data, per_gt_rows, raw_predictions):
    """Verify total counts across ground truth instances and prediction inventory."""
    # 50 untouched images
    assert summary_data["dataset_partition"]["coco_untouched_images"] == 50
    assert summary_data["dataset_partition"]["coco_polygon_annotations"] == 332

    # Per ground-truth rows
    assert len(per_gt_rows) == 332

    # Raw predictions total count
    total_preds = sum(p["predictions_count"] for p in raw_predictions)
    assert total_preds == 366
    assert summary_data["evaluation_comparison"]["baseline_exact"]["total_predictions"] == 366
    assert summary_data["evaluation_comparison"]["taxonomy_aware_v1"]["total_predictions"] == 366


def test_scale_breakdown_population_reconciliation(summary_data, per_gt_rows):
    """Verify that scale subsets strictly sum to overall ground-truth count."""
    scale_counts = {"small": 0, "medium": 0, "large": 0}
    for r in per_gt_rows:
        scale_counts[r["scale"]] += 1

    assert scale_counts["small"] == 110
    assert scale_counts["medium"] == 116
    assert scale_counts["large"] == 106
    assert sum(scale_counts.values()) == 332


def test_baseline_exact_metrics_reconciliation(summary_data):
    """Verify baseline exact matching counts, recall, precision, and scale breakdown."""
    base = summary_data["evaluation_comparison"]["baseline_exact"]

    # IoU 0.50
    assert base["class_aware_matches_iou50"] == 159
    assert base["total_ground_truth"] == 332
    assert base["total_predictions"] == 366
    assert abs(base["recall_class_aware_iou50"] - (159 / 332)) < 1e-4
    assert abs(base["precision_class_aware_iou50"] - (159 / 366)) < 1e-4
    assert base["false_positives_aware_iou50"] == 366 - 159  # 207
    assert base["false_negatives_aware_iou50"] == 332 - 159  # 173

    # IoU 0.75 Authoritative result
    assert base["class_aware_matches_iou75"] == 105
    assert abs(base["recall_class_aware_iou75"] - (105 / 332)) < 1e-4
    assert abs(base["precision_class_aware_iou75"] - (105 / 366)) < 1e-4

    # Scale breakdown @ IoU 0.50
    scale = base["scale_breakdown"]
    assert scale["small"]["gt_count"] == 110
    assert scale["small"]["matched_iou50"] == 42
    assert scale["medium"]["gt_count"] == 116
    assert scale["medium"]["matched_iou50"] == 63
    assert scale["large"]["gt_count"] == 106
    assert scale["large"]["matched_iou50"] == 54
    assert scale["small"]["matched_iou50"] + scale["medium"]["matched_iou50"] + scale["large"]["matched_iou50"] == 159


def test_taxonomy_aware_v1_metrics_reconciliation(summary_data, per_gt_rows):
    """Verify taxonomy-aware v1 matching counts, recall, precision, and scale breakdown."""
    tax = summary_data["evaluation_comparison"]["taxonomy_aware_v1"]

    # IoU 0.50
    assert tax["class_aware_matches_iou50"] == 171
    assert tax["total_ground_truth"] == 332
    assert tax["total_predictions"] == 366
    assert abs(tax["recall_class_aware_iou50"] - (171 / 332)) < 1e-4
    assert abs(tax["precision_class_aware_iou50"] - (171 / 366)) < 1e-4
    assert tax["false_positives_aware_iou50"] == 366 - 171  # 195
    assert tax["false_negatives_aware_iou50"] == 332 - 171  # 161

    # IoU 0.75 Authoritative result
    assert tax["class_aware_matches_iou75"] == 110
    assert abs(tax["recall_class_aware_iou75"] - (110 / 332)) < 1e-4
    assert abs(tax["precision_class_aware_iou75"] - (110 / 366)) < 1e-4

    # Direct verification against per_ground_truth.json rows
    gt_tp50 = sum(1 for r in per_gt_rows if r["matched_class_aware_iou50"])
    gt_tp75 = sum(1 for r in per_gt_rows if r["matched_class_aware_iou75"])
    assert gt_tp50 == 171
    assert gt_tp75 == 110

    # Scale breakdown @ IoU 0.50
    scale = tax["scale_breakdown"]
    assert scale["small"]["gt_count"] == 110
    assert scale["small"]["matched_iou50"] == 42
    assert scale["medium"]["gt_count"] == 116
    assert scale["medium"]["matched_iou50"] == 67
    assert scale["large"]["gt_count"] == 106
    assert scale["large"]["matched_iou50"] == 62
    assert scale["small"]["matched_iou50"] + scale["medium"]["matched_iou50"] + scale["large"]["matched_iou50"] == 171


def test_delta_impact_reconciliation(summary_data):
    """Verify that delta improvements reconcile between baseline and taxonomy v1."""
    comp = summary_data["evaluation_comparison"]
    base = comp["baseline_exact"]
    tax = comp["taxonomy_aware_v1"]

    delta_tp50 = tax["class_aware_matches_iou50"] - base["class_aware_matches_iou50"]
    delta_tp75 = tax["class_aware_matches_iou75"] - base["class_aware_matches_iou75"]

    assert delta_tp50 == 12  # +12 matches recovered at IoU 0.50
    assert delta_tp75 == 5   # +5 matches recovered at IoU 0.75
    assert abs(comp["delta_recall_aware_iou50"] - (12 / 332)) < 1e-4
    assert abs(comp["delta_precision_aware_iou50"] - (12 / 366)) < 1e-4


def test_evidence_receipt_cryptographic_integrity(summary_data):
    """Verify that the summary report carries a valid, unkeyed SHA-256 evidence receipt."""
    receipt_dict = summary_data.get("receipt")
    assert receipt_dict is not None
    assert summary_data.get("receipt_verified") is True

    receipt_obj = EvidenceReceipt(**receipt_dict)
    assert verify_receipt_integrity(receipt_obj) is True
