"""Instance Matching and Multi-Object Assignment Engine.

Scientific Integrity Framework:
================================
1. SINGLE-ANNOTATION CANDIDATE MATCHING:
   In the single-annotation endpoint (/api/vision/audit), the system evaluates one query annotation
   against all model predictions on the image canvas. It selects the best matching candidate according
   to the chosen strategy (class-aware or class-agnostic). This is a point query, NOT a global bijection.

2. GLOBAL ONE-TO-ONE MULTI-OBJECT MATCHING (HUNGARIAN ASSIGNMENT):
   When multiple reference annotations exist on an image, greedy independent selection is invalid:
   it allows a single predicted mask to satisfy multiple distinct human annotations.
   Global one-to-one matching uses the Kuhn-Munkres (Hungarian) algorithm to solve the optimal
   bipartite assignment minimizing total matching cost:
       Cost(i, j) = 1.0 - IoU(Annotation_i, Prediction_j) [+ penalty if class mismatch in class-aware mode]
   Under this assignment, each prediction can satisfy AT MOST ONE human annotation.

3. SCIENTIFIC SEPARATION OF EVALUATION CONCEPTS:
   - Segmentation Agreement: Continuous spatial overlap (IoU, Dice, Boundary Agreement in [0, 1]).
   - Object-Detection Recall: Binary classification of whether an instance is detected with IoU >= threshold.
   - Annotation-Error Detection: Audit verdict (ACCEPT/REVIEW/UNCERTAIN) identifying potential human mistakes.
     CRITICAL SCIENTIFIC PRINCIPLE: A model disagreement alone is NOT proof of a human annotation error.
     The model may suffer from occlusion, class confusion, or false negatives.
   - Physical-Action Verification: Causal verification of spatial-temporal state transitions across frames,
     which cannot be proved by single-frame static segmentation alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from backend.evidence.metrics import calculate_boundary_agreement
from backend.evidence.rust_accelerator import compute_mask_iou_fast
from backend.evidence.taxonomy import are_classes_equivalent


@dataclass(frozen=True)
class MatchedPair:
    annotation_idx: int
    prediction_idx: int
    annotation_id: Optional[int | str]
    prediction_id: int
    annotation_category: str
    prediction_label: str
    iou: float
    boundary_agreement: float
    class_match: bool


@dataclass(frozen=True)
class MultiAnnotationMatchResult:
    matched_pairs: List[MatchedPair]
    unmatched_annotation_indices: List[int]
    unmatched_prediction_indices: List[int]
    cost_matrix: np.ndarray
    mean_matched_iou: float
    recall_iou50: float
    matching_strategy: str


def compute_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute IoU safely, handling empty masks cleanly."""
    if mask_a.shape != mask_b.shape:
        return 0.0
    return float(compute_mask_iou_fast(mask_a, mask_b))


class HungarianInstanceMatcher:
    """Global one-to-one bipartite instance matching using the Hungarian algorithm."""

    @staticmethod
    def match(
        reference_masks: List[np.ndarray],
        reference_categories: List[str],
        reference_ids: Optional[List[int | str]] = None,
        predicted_masks: Optional[List[np.ndarray]] = None,
        predicted_labels: Optional[List[str]] = None,
        predicted_ids: Optional[List[int]] = None,
        matching_strategy: str = "class_aware",
        min_iou_threshold: float = 0.0,
    ) -> MultiAnnotationMatchResult:
        """Perform optimal global 1-to-1 assignment between reference annotations and model predictions.

        Guarantees:
        - At most min(N_annotations, N_predictions) pairs are formed.
        - No prediction is assigned to multiple annotations.
        - No annotation is assigned to multiple predictions.
        """
        n_ann = len(reference_masks)
        n_pred = len(predicted_masks) if predicted_masks else 0

        ref_ids = reference_ids or list(range(n_ann))
        pred_ids = predicted_ids or list(range(n_pred))
        pred_masks = predicted_masks or []
        pred_labels = predicted_labels or []

        if n_ann == 0 or n_pred == 0:
            return MultiAnnotationMatchResult(
                matched_pairs=[],
                unmatched_annotation_indices=list(range(n_ann)),
                unmatched_prediction_indices=list(range(n_pred)),
                cost_matrix=np.empty((n_ann, n_pred), dtype=np.float64),
                mean_matched_iou=0.0,
                recall_iou50=0.0,
                matching_strategy=matching_strategy,
            )

        # Build cost matrix
        # Cost is in [0.0, 1.0] for matching class, or [2.0, 3.0] for class mismatch in class-aware mode
        cost_matrix = np.zeros((n_ann, n_pred), dtype=np.float64)
        iou_matrix = np.zeros((n_ann, n_pred), dtype=np.float64)
        boundary_matrix = np.zeros((n_ann, n_pred), dtype=np.float64)
        class_match_matrix = np.zeros((n_ann, n_pred), dtype=bool)

        for i in range(n_ann):
            ref_mask = reference_masks[i]
            ref_cat = reference_categories[i].strip().lower()
            for j in range(n_pred):
                p_mask = pred_masks[j]
                p_label = pred_labels[j].strip().lower()
                is_cls_match = are_classes_equivalent(ref_cat, p_label)
                class_match_matrix[i, j] = is_cls_match

                # Fast IoU calculation
                iou_val = compute_iou(ref_mask, p_mask)
                iou_matrix[i, j] = iou_val
                boundary_matrix[i, j] = float(calculate_boundary_agreement(ref_mask, p_mask))

                if matching_strategy == "class_aware":
                    # Heavy penalty for class mismatch so same-class instances are always favored
                    cost_matrix[i, j] = (1.0 - iou_val) if is_cls_match else (2.0 + (1.0 - iou_val))
                else:
                    cost_matrix[i, j] = 1.0 - iou_val

        # Solve optimal bipartite matching
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched_pairs: List[MatchedPair] = []
        assigned_ann = set()
        assigned_pred = set()

        for r, c in zip(row_ind, col_ind):
            iou_val = float(iou_matrix[r, c])
            if iou_val >= min_iou_threshold:
                matched_pairs.append(
                    MatchedPair(
                        annotation_idx=int(r),
                        prediction_idx=int(c),
                        annotation_id=ref_ids[r],
                        prediction_id=pred_ids[c],
                        annotation_category=reference_categories[r],
                        prediction_label=pred_labels[c],
                        iou=round(iou_val, 4),
                        boundary_agreement=round(float(boundary_matrix[r, c]), 4),
                        class_match=bool(class_match_matrix[r, c]),
                    )
                )
                assigned_ann.add(int(r))
                assigned_pred.add(int(c))

        unmatched_ann = [i for i in range(n_ann) if i not in assigned_ann]
        unmatched_pred = [j for j in range(n_pred) if j not in assigned_pred]

        matched_ious = [p.iou for p in matched_pairs]
        mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
        recall50 = float(sum(1 for iou in matched_ious if iou >= 0.50) / n_ann) if n_ann > 0 else 0.0

        return MultiAnnotationMatchResult(
            matched_pairs=matched_pairs,
            unmatched_annotation_indices=unmatched_ann,
            unmatched_prediction_indices=unmatched_pred,
            cost_matrix=cost_matrix,
            mean_matched_iou=round(mean_iou, 4),
            recall_iou50=round(recall50, 4),
            matching_strategy=matching_strategy,
        )


def match_single_annotation_candidate(
    reference_mask: np.ndarray,
    reference_category: str,
    predictions: list[Any],
    matching_strategy: str = "class_aware",
) -> Tuple[Optional[Any], float, float, bool]:
    """Execute documented single-annotation candidate matching for /api/vision/audit.

    Selects the best prediction for one query annotation:
    - If class_aware: prioritizes predictions with matching class label, sorted by IoU descending.
      If no matching class exists, falls back to the highest spatial IoU candidate with class_match=False.
    - If class_agnostic: selects prediction with highest spatial IoU regardless of class label.

    Returns:
        (best_prediction, best_iou, best_boundary_agreement, class_match)
    """
    if not predictions:
        return None, 0.0, 0.0, False

    ref_cat = reference_category.strip().lower()
    evaluated = []

    for pred in predictions:
        iou = compute_iou(reference_mask, pred.mask)
        boundary = float(calculate_boundary_agreement(reference_mask, pred.mask))
        cls_match = are_classes_equivalent(pred.label, ref_cat)
        evaluated.append((pred, iou, boundary, cls_match))

    if matching_strategy == "class_aware":
        matching_class = [item for item in evaluated if item[3]]
        if matching_class:
            matching_class.sort(key=lambda item: item[1], reverse=True)
            best = matching_class[0]
            return best[0], best[1], best[2], best[3]
        else:
            evaluated.sort(key=lambda item: item[1], reverse=True)
            best = evaluated[0]
            return best[0], best[1], best[2], False
    else:
        evaluated.sort(key=lambda item: item[1], reverse=True)
        best = evaluated[0]
        return best[0], best[1], best[2], best[3]
