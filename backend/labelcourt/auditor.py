"""LabelCourt visual annotation auditor comparing human annotations against model evidence."""

from __future__ import annotations

import uuid
from typing import Optional
import numpy as np

from backend.evidence.masks import (
    bbox_to_mask,
    generate_overlay_heatmap,
    polygon_to_mask,
)
from backend.evidence.metrics import (
    calculate_area_ratio,
    calculate_bbox_displacement,
    calculate_boundary_agreement,
    calculate_mask_iou,
)
from backend.evidence.recapture import plan_recapture
from backend.evidence.receipts import calculate_sha256, generate_receipt
from backend.evidence.schemas import (
    AnnotationAuditResult,
    AnnotationItem,
    AuditVerdict,
    EvidenceReceipt,
)
from backend.evidence.uncertainty import evaluate_all_uncertainty_signals


class LabelCourtAuditor:
    """Audits image annotations by measuring spatial agreement and visual uncertainty."""

    def __init__(
        self,
        iou_accept_threshold: float = 0.85,
        iou_review_threshold: float = 0.50,
        boundary_agreement_threshold: float = 0.75,
        blur_threshold: float = 100.0,
        boundary_margin_px: int = 3,
    ) -> None:
        self.iou_accept_threshold = iou_accept_threshold
        self.iou_review_threshold = iou_review_threshold
        self.boundary_agreement_threshold = boundary_agreement_threshold
        self.blur_threshold = blur_threshold
        self.boundary_margin_px = boundary_margin_px

    def audit_annotation(
        self,
        human_annotation: AnnotationItem,
        candidate_mask: np.ndarray,
        image_gray: Optional[np.ndarray] = None,
        image_rgb: Optional[np.ndarray] = None,
        independent_class_candidate: Optional[str] = None,
    ) -> tuple[AnnotationAuditResult, EvidenceReceipt]:
        """Perform an objective audit on a single annotation item against a candidate segmentation mask."""
        height, width = candidate_mask.shape

        # 1. Rasterize human annotation
        if human_annotation.segmentation and len(human_annotation.segmentation) > 0:
            human_mask = polygon_to_mask(human_annotation.segmentation[0], height, width)
        else:
            human_mask = bbox_to_mask(human_annotation.bbox, height, width)

        # 2. Compute spatial metrics
        iou = calculate_mask_iou(human_mask, candidate_mask)
        boundary_agreement = calculate_boundary_agreement(human_mask, candidate_mask)
        area_ratio = calculate_area_ratio(human_mask, candidate_mask)

        # Candidate bounding box
        candidate_indices = np.where(candidate_mask)
        if len(candidate_indices[0]) > 0:
            from backend.evidence.schemas import BoundingBox
            c_bbox = BoundingBox(
                x_min=float(np.min(candidate_indices[1])),
                y_min=float(np.min(candidate_indices[0])),
                x_max=float(np.max(candidate_indices[1]) + 1),
                y_max=float(np.max(candidate_indices[0]) + 1),
            )
            bbox_disp = calculate_bbox_displacement(human_annotation.bbox, c_bbox, image_diag=float(np.hypot(height, width)))
        else:
            bbox_disp = 1.0

        # 3. Class match evaluation
        class_match = True
        if independent_class_candidate is not None:
            class_match = (human_annotation.category_name.strip().lower() == independent_class_candidate.strip().lower())

        # 4. Evaluate uncertainty signals
        signals = evaluate_all_uncertainty_signals(
            image_gray=image_gray,
            current_mask=candidate_mask,
            blur_threshold=self.blur_threshold,
            boundary_margin_px=self.boundary_margin_px,
        )

        has_high_uncertainty = any(s.severity == "HIGH" for s in signals)

        # 5. Decide audit verdict
        if not class_match:
            verdict = AuditVerdict.REVIEW
        elif has_high_uncertainty:
            verdict = AuditVerdict.UNCERTAIN
        elif iou >= self.iou_accept_threshold and boundary_agreement >= self.boundary_agreement_threshold:
            verdict = AuditVerdict.ACCEPT
        elif iou >= self.iou_review_threshold:
            verdict = AuditVerdict.REVIEW
        else:
            verdict = AuditVerdict.UNCERTAIN if len(signals) > 0 else AuditVerdict.REVIEW

        # 6. ReCapture planning
        recapture_plan = plan_recapture(signals) if (verdict == AuditVerdict.UNCERTAIN or len(signals) > 0) else None

        # 7. Generate disagreement heatmap
        heatmap_b64 = generate_overlay_heatmap(human_mask, candidate_mask, background_rgb=image_rgb)

        audit_result = AnnotationAuditResult(
            annotation_id=human_annotation.id,
            verdict=verdict,
            iou=round(iou, 4),
            boundary_agreement=round(boundary_agreement, 4),
            bbox_displacement=round(bbox_disp, 4),
            area_ratio=round(area_ratio, 4),
            class_match=class_match,
            uncertainty_signals=signals,
            recapture_plan=recapture_plan,
            disagreement_heatmap_base64=heatmap_b64,
        )

        # 8. Cryptographic Evidence Receipt
        report_id = f"LC-{uuid.uuid4().hex[:12].upper()}"
        inputs_digest = calculate_sha256(
            f"{human_annotation.id}_{human_annotation.category_name}_{human_annotation.bbox.model_dump_json()}_{iou:.4f}"
        )

        receipt = generate_receipt(
            report_id=report_id,
            engine_name="LabelCourt Visual Annotation Evidence Auditor",
            software_version="0.1.0",
            inputs_hash=inputs_digest,
            measurements={
                "iou": round(iou, 4),
                "boundary_agreement": round(boundary_agreement, 4),
                "area_ratio": round(area_ratio, 4),
                "bbox_displacement": round(bbox_disp, 4),
                "class_match": class_match,
            },
            signals=signals,
            verdict=verdict.value,
            recapture_directive=recapture_plan.directive if recapture_plan else None,
        )

        return audit_result, receipt
