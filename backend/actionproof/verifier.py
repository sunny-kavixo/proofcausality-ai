"""ActionProof physical action verification engine."""

from __future__ import annotations

import math
import uuid
from typing import Optional
import numpy as np

from backend.actionproof.relations import compute_centroid_distance, is_contained_in_region
from backend.evidence.masks import mask_to_bbox
from backend.evidence.recapture import plan_recapture
from backend.evidence.receipts import calculate_sha256, generate_receipt
from backend.evidence.schemas import (
    ActionVerdict,
    ActionVerificationRequest,
    ActionVerificationResult,
    BoundingBox,
    EvidenceReceipt,
)
from backend.evidence.uncertainty import evaluate_all_uncertainty_signals


class ActionProofVerifier:
    """Verifies observable physical actions from before and after visual evidence frames."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        boundary_margin_px: int = 3,
        occlusion_drop_threshold: float = 0.35,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.boundary_margin_px = boundary_margin_px
        self.occlusion_drop_threshold = occlusion_drop_threshold

    def verify_action(
        self,
        request: ActionVerificationRequest,
        before_mask: np.ndarray,
        after_mask: np.ndarray,
        after_image_gray: Optional[np.ndarray] = None,
        has_final_state: bool = True,
    ) -> tuple[ActionVerificationResult, EvidenceReceipt]:
        """Verify before/after physical action with explicit uncertainty decomposition."""
        report_id = f"AP-{uuid.uuid4().hex[:12].upper()}"

        bbox_before = mask_to_bbox(before_mask)
        bbox_after = mask_to_bbox(after_mask)

        # 1. Evaluate uncertainty signals on the after frame
        signals = evaluate_all_uncertainty_signals(
            image_gray=after_image_gray,
            current_mask=after_mask,
            previous_mask=before_mask,
            has_final_state=has_final_state,
            blur_threshold=self.blur_threshold,
            boundary_margin_px=self.boundary_margin_px,
            occlusion_drop_threshold=self.occlusion_drop_threshold,
        )

        has_uncertainty = any(s.severity in ("HIGH", "MEDIUM") for s in signals)

        # 2. Compute observable displacement
        if bbox_before is not None and bbox_after is not None:
            displacement = compute_centroid_distance(bbox_before, bbox_after)
        else:
            displacement = 0.0

        distance_to_target = None
        verdict = ActionVerdict.UNCERTAIN

        # 3. Task specific verification
        if request.task_type == "moved_A_to_B":
            if bbox_before is None or bbox_after is None:
                verdict = ActionVerdict.UNCERTAIN if has_uncertainty else ActionVerdict.FAILED
            else:
                in_a = is_contained_in_region(bbox_before, request.region_a) if request.region_a else True
                in_b = is_contained_in_region(bbox_after, request.region_b) if request.region_b else True

                if has_uncertainty:
                    verdict = ActionVerdict.UNCERTAIN
                elif in_a and in_b and displacement >= request.distance_threshold_px:
                    verdict = ActionVerdict.SUCCESS
                else:
                    verdict = ActionVerdict.FAILED

        elif request.task_type == "placed_near":
            if bbox_after is None or request.target_object_bbox is None:
                verdict = ActionVerdict.UNCERTAIN if has_uncertainty else ActionVerdict.FAILED
            else:
                distance_to_target = compute_centroid_distance(bbox_after, request.target_object_bbox)
                if has_uncertainty:
                    verdict = ActionVerdict.UNCERTAIN
                elif distance_to_target <= request.distance_threshold_px:
                    verdict = ActionVerdict.SUCCESS
                else:
                    verdict = ActionVerdict.FAILED

        elif request.task_type == "object_present":
            is_present = np.any(after_mask)
            if has_uncertainty:
                verdict = ActionVerdict.UNCERTAIN
            else:
                verdict = ActionVerdict.SUCCESS if is_present else ActionVerdict.FAILED

        elif request.task_type == "object_absent":
            is_absent = not np.any(after_mask)
            if has_uncertainty:
                verdict = ActionVerdict.UNCERTAIN
            else:
                verdict = ActionVerdict.SUCCESS if is_absent else ActionVerdict.FAILED

        else:
            verdict = ActionVerdict.UNCERTAIN

        # 4. Plan ReCapture if uncertain
        recapture_plan = plan_recapture(signals) if (verdict == ActionVerdict.UNCERTAIN or len(signals) > 0) else None

        confidence = 0.95 if verdict == ActionVerdict.SUCCESS else (0.90 if verdict == ActionVerdict.FAILED else 0.40)

        result = ActionVerificationResult(
            report_id=report_id,
            task_type=request.task_type,
            verdict=verdict,
            confidence=confidence,
            displacement_px=round(displacement, 2),
            distance_to_target_px=round(distance_to_target, 2) if distance_to_target is not None else None,
            uncertainty_signals=signals,
            recapture_plan=recapture_plan,
            evidence_receipt_id=report_id,
        )

        # 5. Generate Evidence Receipt
        inputs_digest = calculate_sha256(
            f"{request.task_type}_{request.object_name}_{displacement:.2f}_{verdict.value}"
        )

        receipt = generate_receipt(
            report_id=report_id,
            engine_name="ActionProof Physical Action Verification Engine",
            software_version="0.1.0",
            inputs_hash=inputs_digest,
            measurements={
                "task_type": request.task_type,
                "displacement_px": round(displacement, 2),
                "distance_to_target_px": round(distance_to_target, 2) if distance_to_target is not None else None,
                "has_final_state": has_final_state,
            },
            signals=signals,
            verdict=verdict.value,
            recapture_directive=recapture_plan.directive if recapture_plan else None,
        )

        return result, receipt
