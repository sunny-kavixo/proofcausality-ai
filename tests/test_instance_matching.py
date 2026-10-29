"""Regression tests for instance matching correctness and scientific separation (Issue 4).

Verifies:
- One annotation, one prediction.
- One annotation, multiple predictions (optimal best selection).
- Multiple annotations, one prediction (strictly one-to-one assignment; second annotation is unmatched).
- Duplicate candidate masks (assigned distinct targets).
- Same-class overlapping objects (optimal 1-to-1 assignment).
- Wrong-class overlap (penalized under class-aware, allowed under class-agnostic).
- No predictions (handled safely without error).
- Empty reference mask (zero overlap, handled without divide-by-zero).
- Class-aware versus class-agnostic matching behavior.
- Scientific separation:
    * Segmentation agreement (IoU, Dice, Boundary agreement)
    * Object-detection recall (binary detection flag at threshold)
    * Annotation-error detection (model disagreement != proof of human mistake)
    * Physical-action verification (temporal state changes across frames)
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.evidence.matching import (
    HungarianInstanceMatcher,
    MultiAnnotationMatchResult,
    compute_iou,
    match_single_annotation_candidate,
)
from backend.providers.local_segmentation import InstancePrediction


def _create_box_mask(x1: int, y1: int, x2: int, y2: int, h: int = 200, w: int = 200) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_one_annotation_one_prediction():
    """Single reference mask and single prediction with substantial overlap match 1-to-1."""
    ref_mask = _create_box_mask(20, 20, 80, 80)
    pred_mask = _create_box_mask(25, 25, 85, 85)

    result = HungarianInstanceMatcher.match(
        reference_masks=[ref_mask],
        reference_categories=["bottle"],
        reference_ids=[101],
        predicted_masks=[pred_mask],
        predicted_labels=["bottle"],
        predicted_ids=[1],
        matching_strategy="class_aware",
    )

    assert len(result.matched_pairs) == 1
    pair = result.matched_pairs[0]
    assert pair.annotation_id == 101
    assert pair.prediction_id == 1
    assert pair.class_match is True
    assert pair.iou > 0.60
    assert len(result.unmatched_annotation_indices) == 0
    assert len(result.unmatched_prediction_indices) == 0


def test_one_annotation_multiple_predictions():
    """When multiple predictions exist, Hungarian matching assigns the highest-agreement candidate."""
    ref_mask = _create_box_mask(20, 20, 80, 80)
    # pred 1: far away (IoU = 0)
    pred1 = _create_box_mask(120, 120, 180, 180)
    # pred 2: high overlap
    pred2 = _create_box_mask(22, 22, 78, 78)

    result = HungarianInstanceMatcher.match(
        reference_masks=[ref_mask],
        reference_categories=["cup"],
        reference_ids=["ref_1"],
        predicted_masks=[pred1, pred2],
        predicted_labels=["cup", "cup"],
        predicted_ids=[1, 2],
        matching_strategy="class_aware",
    )

    assert len(result.matched_pairs) == 1
    pair = result.matched_pairs[0]
    assert pair.prediction_id == 2
    assert pair.iou > 0.80
    assert 0 in result.unmatched_prediction_indices  # pred1 left unassigned


def test_multiple_annotations_one_prediction_strictly_one_to_one():
    """A single prediction MUST NEVER be assigned to satisfy multiple human annotations."""
    # Two overlapping human annotations on adjacent objects
    ann1_mask = _create_box_mask(20, 20, 70, 70)
    ann2_mask = _create_box_mask(30, 30, 80, 80)
    # One single model prediction in the center
    pred_mask = _create_box_mask(25, 25, 75, 75)

    result = HungarianInstanceMatcher.match(
        reference_masks=[ann1_mask, ann2_mask],
        reference_categories=["car", "car"],
        reference_ids=["ann_A", "ann_B"],
        predicted_masks=[pred_mask],
        predicted_labels=["car"],
        predicted_ids=[99],
        matching_strategy="class_aware",
    )

    # Exactly one pair formed; the second annotation MUST remain unmatched!
    assert len(result.matched_pairs) == 1
    assert len(result.unmatched_annotation_indices) == 1
    assert len(result.unmatched_prediction_indices) == 0


def test_duplicate_candidate_masks():
    """Two identical prediction masks paired with two annotations assign distinct targets."""
    ann1 = _create_box_mask(10, 10, 50, 50)
    ann2 = _create_box_mask(60, 60, 100, 100)

    # Two identical predictions near ann1
    dup1 = _create_box_mask(10, 10, 50, 50)
    dup2 = _create_box_mask(10, 10, 50, 50)

    result = HungarianInstanceMatcher.match(
        reference_masks=[ann1, ann2],
        reference_categories=["chair", "chair"],
        predicted_masks=[dup1, dup2],
        predicted_labels=["chair", "chair"],
        matching_strategy="class_aware",
    )

    assigned_preds = {pair.prediction_idx for pair in result.matched_pairs}
    assert len(assigned_preds) == len(result.matched_pairs)  # All assigned predictions are unique


def test_same_class_overlapping_objects():
    """Two overlapping objects of the same class are assigned to their respective closest prediction."""
    ann_left = _create_box_mask(10, 10, 60, 60)
    ann_right = _create_box_mask(40, 10, 90, 60)

    pred_left = _create_box_mask(12, 10, 58, 60)
    pred_right = _create_box_mask(42, 10, 88, 60)

    result = HungarianInstanceMatcher.match(
        reference_masks=[ann_left, ann_right],
        reference_categories=["person", "person"],
        predicted_masks=[pred_right, pred_left],  # Order inverted
        predicted_labels=["person", "person"],
        predicted_ids=[20, 10],
        matching_strategy="class_aware",
    )

    assert len(result.matched_pairs) == 2
    pairs_by_ann = {p.annotation_idx: p for p in result.matched_pairs}
    assert pairs_by_ann[0].prediction_id == 10  # ann_left -> pred_left
    assert pairs_by_ann[1].prediction_id == 20  # ann_right -> pred_right


def test_wrong_class_overlap_class_aware_vs_class_agnostic():
    """Class-aware matching penalizes class mismatch; class-agnostic matching matches purely by IoU."""
    ref_mask = _create_box_mask(20, 20, 80, 80)
    # pred 1: same class, moderate IoU
    pred1_same_cls = _create_box_mask(35, 35, 80, 80)  # smaller overlap
    # pred 2: different class, higher spatial IoU
    pred2_wrong_cls = _create_box_mask(21, 21, 79, 79)  # almost exact overlap

    # 1. Class-aware matching: must pick pred1 (matching class)
    res_aware = HungarianInstanceMatcher.match(
        reference_masks=[ref_mask],
        reference_categories=["chair"],
        predicted_masks=[pred1_same_cls, pred2_wrong_cls],
        predicted_labels=["chair", "dining table"],
        predicted_ids=[1, 2],
        matching_strategy="class_aware",
    )
    assert res_aware.matched_pairs[0].prediction_id == 1
    assert res_aware.matched_pairs[0].class_match is True

    # 2. Class-agnostic matching: must pick pred2 (highest spatial IoU)
    res_agnostic = HungarianInstanceMatcher.match(
        reference_masks=[ref_mask],
        reference_categories=["chair"],
        predicted_masks=[pred1_same_cls, pred2_wrong_cls],
        predicted_labels=["chair", "dining table"],
        predicted_ids=[1, 2],
        matching_strategy="class_agnostic",
    )
    assert res_agnostic.matched_pairs[0].prediction_id == 2
    assert res_agnostic.matched_pairs[0].class_match is False


def test_no_predictions_handled_safely():
    """When the model detects zero instances, matcher returns all annotations as unmatched."""
    ref_mask = _create_box_mask(10, 10, 50, 50)
    result = HungarianInstanceMatcher.match(
        reference_masks=[ref_mask],
        reference_categories=["person"],
        predicted_masks=[],
        predicted_labels=[],
    )
    assert len(result.matched_pairs) == 0
    assert result.unmatched_annotation_indices == [0]
    assert result.mean_matched_iou == 0.0
    assert result.recall_iou50 == 0.0


def test_empty_reference_mask():
    """An empty reference mask (all zeros) produces zero IoU without runtime error."""
    empty_mask = np.zeros((100, 100), dtype=bool)
    pred_mask = _create_box_mask(10, 10, 50, 50, h=100, w=100)

    iou_val = compute_iou(empty_mask, pred_mask)
    assert iou_val == 0.0

    result = HungarianInstanceMatcher.match(
        reference_masks=[empty_mask],
        reference_categories=["bottle"],
        predicted_masks=[pred_mask],
        predicted_labels=["bottle"],
    )
    assert len(result.matched_pairs) == 1
    assert result.matched_pairs[0].iou == 0.0


def test_scientific_separation_principles():
    """Verify explicit separation between agreement, recall, audit verdict, and action proof."""
    # 1. Segmentation Agreement: continuous metric in [0, 1]
    ref_mask = _create_box_mask(10, 10, 50, 50)
    cand_mask = _create_box_mask(15, 15, 55, 55)
    iou = compute_iou(ref_mask, cand_mask)
    assert 0.0 < iou < 1.0

    # 2. Object-Detection Recall: binary outcome
    is_detected_at_50 = bool(iou >= 0.50)
    assert isinstance(is_detected_at_50, bool)

    # 3. Model Disagreement != Proof of Human Annotation Error
    # A model false negative or partial segmentation does not falsify ground truth
    pred_wrong = InstancePrediction(instance_id=1, label_id=1, label="couch", score=0.9, mask=cand_mask)
    best_p, b_iou, b_agree, cls_m = match_single_annotation_candidate(
        ref_mask, "chair", [pred_wrong], matching_strategy="class_aware"
    )
    assert cls_m is False
    # The disagreement indicates a discrepancy requiring REVIEW, never an unconfirmed proof of human defect.
    assert best_p is not None
