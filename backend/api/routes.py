"""FastAPI application routes for ProofCausality AI."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
import numpy as np

from backend.actionproof.verifier import ActionProofVerifier
from backend.evidence.advanced_metrics import wilson_score_interval
from backend.evidence.image_processing import (
    ImageValidationError,
    create_disagreement_visual_overlay,
    encode_image_to_base64_png,
    parse_coco_polygon_to_mask,
    validate_and_decode_image,
)
from backend.evidence.masks import bbox_to_mask
from backend.evidence.receipts import verify_receipt_integrity
from backend.evidence.rust_accelerator import (
    compute_laplacian_variance_fast,
    compute_mask_dice_fast,
    compute_mask_iou_fast,
    get_accelerator_info,
    is_rust_accelerator_available,
)
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
    image_base64: Optional[str] = None


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
    after_image_base64: Optional[str] = None


class ImageUploadPayload(BaseModel):
    image_base64: str
    filename: Optional[str] = "upload.png"


@router.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe reporting active version, engine identity, and acceleration status."""
    acc_info = get_accelerator_info()
    return {
        "status": "ok",
        "version": "0.1.1",
        "project": "ProofCausality AI",
        "tagline": "Prove what happened. Detect what is missing. Capture the evidence that changes the decision.",
        "engines": [
            "LabelCourt",
            "ActionProof",
            "EvidenceCore",
            "ReCapture",
        ],
        "hardware_acceleration": {
            "rust_native_active": acc_info["rust_accelerator_active"],
            "backend_engine": acc_info["backend_engine"],
        },
    }


@router.get("/model-info")
def model_info() -> dict[str, Any]:
    """Diagnostics on Meta adapter, active models, and deterministic fallback status."""
    diagnostics = meta_adapter.get_diagnostics()
    diagnostics["rust_accelerator"] = get_accelerator_info()
    return diagnostics


@router.post("/image/upload")
def upload_image(payload: ImageUploadPayload) -> dict[str, Any]:
    """Validate and decode uploaded image, returning dimensions, SHA-256 hash, and blur variance."""
    raw_b64 = payload.image_base64
    if "," in raw_b64:
        raw_b64 = raw_b64.split(",", 1)[1]

    try:
        raw_bytes = base64.b64decode(raw_b64)
        rgb_arr, sha256_hash, width, height = validate_and_decode_image(raw_bytes)
        # Compute real image Laplacian variance
        gray_arr = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        lap_var = compute_laplacian_variance_fast(gray_arr)
        is_blurry = bool(lap_var < 100.0)

        return {
            "status": "valid",
            "sha256": sha256_hash,
            "width": width,
            "height": height,
            "laplacian_variance": lap_var,
            "is_blurry": is_blurry,
            "format": "RGB",
        }
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process image payload: {e}")


@router.post("/labelcourt/audit")
def audit_annotation(payload: AuditRequest) -> dict[str, Any]:
    """Execute LabelCourt audit on an annotation item with optional real image overlay."""
    h, w = payload.image_height, payload.image_width

    # Parse candidate mask
    if payload.candidate_polygon:
        candidate_mask = parse_coco_polygon_to_mask(payload.candidate_polygon, w, h)
    elif payload.candidate_bbox:
        candidate_mask = bbox_to_mask(payload.candidate_bbox, h, w)
    else:
        candidate_mask = bbox_to_mask(payload.human_annotation.bbox, h, w)

    # Process image if base64 supplied
    image_gray = None
    overlay_data_uri = None
    input_sha256 = None

    if payload.image_base64:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
            rgb_arr, sha256_hash, iw, ih = validate_and_decode_image(raw_bytes)
            input_sha256 = sha256_hash
            image_gray = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
            # Human ground-truth mask
            human_mask = bbox_to_mask(payload.human_annotation.bbox, ih, iw)
            overlay = create_disagreement_visual_overlay(rgb_arr, human_mask, candidate_mask)
            overlay_data_uri = encode_image_to_base64_png(overlay)
        except Exception:
            pass

    if image_gray is None:
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
    if input_sha256:
        receipt.measurements["source_image_sha256"] = input_sha256

    _receipt_store[receipt.report_id] = receipt

    response_data: dict[str, Any] = {
        "audit_result": result.model_dump(),
        "evidence_receipt": receipt.model_dump(),
        "receipt_valid": verify_receipt_integrity(receipt),
    }
    if overlay_data_uri:
        response_data["overlay_data_uri"] = overlay_data_uri

    return response_data


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

    input_sha256 = None
    image_gray = None

    if payload.after_image_base64:
        raw_b64 = payload.after_image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
            rgb_arr, sha256_hash, _, _ = validate_and_decode_image(raw_bytes)
            input_sha256 = sha256_hash
            image_gray = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        except Exception:
            pass

    if image_gray is None:
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
    if input_sha256:
        receipt.measurements["after_image_sha256"] = input_sha256

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
def get_benchmark_summary(dataset: str = Query("all")) -> dict[str, Any]:
    """Return scientific evaluation results partitioned into Synthetic Baseline and Real-World Evaluation."""
    base_dir = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks"
    syn_file = base_dir / "benchmark_results_synthetic.json"
    real_file = base_dir / "benchmark_results_real.json"

    # Fallback to standard if synthetic file not yet split
    if not syn_file.exists():
        syn_file = base_dir / "benchmark_results.json"

    syn_data = None
    real_data = None

    if syn_file.exists():
        with open(syn_file, "r") as f:
            syn_data = json.load(f)

    if real_file.exists():
        with open(real_file, "r") as f:
            real_data = json.load(f)
    else:
        from evaluation.benchmark_real import run_real_benchmarks
        real_data = run_real_benchmarks()

    if dataset == "synthetic":
        return syn_data or {}
    if dataset == "real":
        return real_data or {}

    return {
        "synthetic_baseline": syn_data,
        "real_world_evaluation": real_data,
        "metadata": {
            "disclaimer": "Synthetic baseline establishes mathematical rule consistency; real evaluation tests physical tabletop pairs.",
            "metrics_reporting_standard": "All percentages accompanied by numerators, denominators, and 95% Wilson score confidence intervals.",
        },
    }


@router.get("/benchmark/latency")
def get_latency_benchmark() -> dict[str, Any]:
    """Return NumPy baseline vs Rust native acceleration latency profiles."""
    bench_file = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks" / "benchmark_latency_comparison.json"
    if not bench_file.exists():
        from evaluation.benchmark_rust_speedup import run_latency_benchmark
        return run_latency_benchmark()

    with open(bench_file, "r") as f:
        return json.load(f)
