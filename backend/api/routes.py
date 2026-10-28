"""FastAPI application routes for ProofCausality AI."""

from __future__ import annotations

import json
import os
from typing import Any, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
import numpy as np

from backend.actionproof.verifier import ActionProofVerifier
from backend.evidence.masks import bbox_to_mask
from backend.evidence.receipts import verify_receipt_integrity
from backend.evidence.schemas import (
    ActionVerificationRequest,
    AnnotationItem,
    BoundingBox,
    EvidenceReceipt,
    UncertaintySignal,
)
from backend.evidence.recapture import plan_recapture
from backend.labelcourt.auditor import LabelCourtAuditor
from backend.providers.meta_adapter import MetaAdapter

router = APIRouter(prefix="/api")

# In-memory receipt store
_receipt_store: dict[str, EvidenceReceipt] = {}

# Initialize engines
auditor = LabelCourtAuditor()
verifier = ActionProofVerifier()
meta_adapter = MetaAdapter()


class AuditRequest(BaseModel):
    human_annotation: AnnotationItem
    candidate_bbox: Optional[BoundingBox] = None
    candidate_polygon: Optional[list[float]] = None
    image_width: int = 600
    image_height: int = 400
    is_blurry: bool = False
    class_candidate: Optional[str] = None


class VerifyActionPayload(BaseModel):
    request: ActionVerificationRequest
    before_bbox: BoundingBox
    after_bbox: BoundingBox
    image_width: int = 600
    image_height: int = 400
    is_blurry: bool = False
    area_drop: bool = False
    touches_border: bool = False
    has_final_state: bool = True


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe reporting active version and engine identity."""
    return {
        "status": "ok",
        "version": "0.1.0",
        "project": "ProofCausality AI",
        "tagline": "Prove what happened. Detect what is missing. Capture the evidence that changes the decision.",
        "engines": [
            "LabelCourt",
            "ActionProof",
            "EvidenceCore",
            "ReCapture",
        ],
    }


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    """Diagnostics on Meta adapter, active models, and deterministic fallback status."""
    return meta_adapter.get_diagnostics()


@router.post("/labelcourt/audit")
def audit_annotation(payload: AuditRequest) -> dict[str, Any]:
    """Execute LabelCourt audit on an annotation item."""
    h, w = payload.image_height, payload.image_width
    if payload.candidate_bbox:
        candidate_mask = bbox_to_mask(payload.candidate_bbox, h, w)
    else:
        candidate_mask = bbox_to_mask(payload.human_annotation.bbox, h, w)

    # Optional synthetic image canvas
    image_gray = np.full((h, w), 128, dtype=np.uint8)
    if not payload.is_blurry:
        image_gray[::4, ::4] = 240
        image_gray[1::4, 1::4] = 10

    result, receipt = auditor.audit_annotation(
        human_annotation=payload.human_annotation,
        candidate_mask=candidate_mask,
        image_gray=image_gray,
        independent_class_candidate=payload.class_candidate,
    )
    _receipt_store[receipt.report_id] = receipt

    return {
        "audit_result": result.model_dump(),
        "evidence_receipt": receipt.model_dump(),
        "receipt_valid": verify_receipt_integrity(receipt),
    }


@router.post("/actionproof/verify")
def verify_action(payload: VerifyActionPayload) -> dict[str, Any]:
    """Execute ActionProof physical action verification."""
    h, w = payload.image_height, payload.image_width
    before_mask = bbox_to_mask(payload.before_bbox, h, w)
    after_mask = bbox_to_mask(payload.after_bbox, h, w)

    if payload.area_drop:
        rows = np.where(after_mask)[0]
        if len(rows) > 0:
            mid_row = int(np.mean(rows))
            after_mask[mid_row:, :] = False

    image_gray = np.full((h, w), 128, dtype=np.uint8)
    if not payload.is_blurry:
        image_gray[::4, ::4] = 240
        image_gray[1::4, 1::4] = 10

    result, receipt = verifier.verify_action(
        request=payload.request,
        before_mask=before_mask,
        after_mask=after_mask,
        after_image_gray=image_gray,
        has_final_state=payload.has_final_state,
    )
    _receipt_store[receipt.report_id] = receipt

    return {
        "verification_result": result.model_dump(),
        "evidence_receipt": receipt.model_dump(),
        "receipt_valid": verify_receipt_integrity(receipt),
    }


@router.post("/recapture/plan")
def plan_recapture_endpoint(signals: list[UncertaintySignal]) -> dict[str, Any]:
    """Generate deterministic ReCapture recovery instructions from uncertainty signals."""
    plan = plan_recapture(signals)
    if not plan:
        return {"has_plan": False, "plan": None}
    return {"has_plan": True, "plan": plan.model_dump()}


@router.get("/receipts/{report_id}")
def get_receipt(report_id: str) -> dict[str, Any]:
    """Retrieve an evidence receipt by ID and check cryptographic SHA-256 integrity."""
    if report_id not in _receipt_store:
        raise HTTPException(status_code=404, detail="Receipt not found")
    receipt = _receipt_store[report_id]
    is_valid = verify_receipt_integrity(receipt)
    return {
        "receipt": receipt.model_dump(),
        "cryptographic_integrity_verified": is_valid,
    }


@router.get("/benchmark/summary")
def get_benchmark_summary() -> dict[str, Any]:
    """Return latest scientific evaluation results and metrics."""
    bench_file = str(Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks" / "benchmark_results.json")
    if not os.path.exists(bench_file):
        from evaluation.benchmark import run_benchmarks
        return run_benchmarks()

    with open(bench_file, "r") as f:
        return json.load(f)
