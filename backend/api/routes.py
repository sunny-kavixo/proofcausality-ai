from __future__ import annotations

import uuid
from backend import __version__
from backend.evidence.calculus.engine import EvidenceCalculusEngine
from backend.evidence.calculus.graph import build_canonical_task_graph
from backend.evidence.calculus.budget_planner import EvidenceBudgetPlanner
from backend.evidence.calculus.schemas import (
    EvidenceDebtReport,
    ObservablePredicate,
    RecoveryAction,
    TaskSpecification,
)

"""FastAPI application routes for ProofCausality AI."""


import base64
import collections
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Optional
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
import numpy as np

logger = logging.getLogger("proofcausality.api")


def sanitize_error_message(exc: Exception) -> str:
    """Sanitize error messages to eliminate local filesystem paths, secrets, or internal traces."""
    msg = str(exc)
    # Strip absolute filesystem paths (Unix / macOS)
    msg = re.sub(r'(/[a-zA-Z0-9_\.\-]+){2,}', '[PATH_REDACTED]', msg)
    # Strip Windows filesystem paths
    msg = re.sub(r'[A-Za-z]:\\[\w\.\-\\]+', '[PATH_REDACTED]', msg)
    # Strip sensitive hex strings > 32 characters (e.g. secret keys or hashes)
    msg = re.sub(r'([A-Fa-f0-9]{32,64})', '[REDACTED_HASH]', msg)
    # Strip environment variable or secret patterns
    msg = re.sub(r'(TOKEN|KEY|SECRET|PASSWORD)=[\S]+', '[REDACTED_SECRET]', msg, flags=re.IGNORECASE)
    return msg


import time
from backend.actionproof.verifier import ActionProofVerifier
from backend.evidence.advanced_metrics import wilson_score_interval
from backend.evidence.dataset_manifest import verify_authoritative_provenance
from backend.evidence.image_processing import (
    ImageValidationError,
    create_disagreement_visual_overlay,
    create_single_mask_overlay,
    encode_image_to_base64_png,
    parse_coco_polygon_to_mask,
    validate_and_decode_image,
)
from backend.evidence.masks import bbox_to_mask
from backend.evidence.metrics import calculate_boundary_agreement
from backend.evidence.receipts import (
    calculate_sha256,
    canonical_json_dump,
    generate_receipt,
    verify_receipt_integrity,
)
from backend.evidence.rust_accelerator import (
    compute_laplacian_variance_fast,
    compute_mask_dice_fast,
    compute_mask_iou_fast,
    get_accelerator_info,
    is_rust_accelerator_available,
)
from backend.evidence.schemas import (
    ActionVerdict,
    ActionVerificationRequest,
    AnnotationItem,
    BoundingBox,
    EvidenceReceipt,
    MatchedModelInstance,
    UncertaintyReason,
    UncertaintySignal,
    VisionAuditAnnotation,
    VisionAuditRequest,
    VisionAuditResponse,
    VisionAuditSettings,
    VisionAuditVerdict,
)
from backend.evidence.recapture import plan_recapture
from backend.labelcourt.auditor import LabelCourtAuditor
from backend.providers.local_segmentation import (
    InstancePrediction,
    LocalMask2FormerProvider,
    encode_binary_mask_rle,
    get_local_mask2former_provider,
)
from backend.providers.meta_adapter import MetaAdapter

router = APIRouter(prefix="/api")

# In-memory receipt store (bounded to 1000 items to prevent memory exhaustion)
MAX_RECEIPT_STORE_SIZE = 1000
_receipt_store: collections.OrderedDict[str, EvidenceReceipt] = collections.OrderedDict()


def store_receipt(receipt: EvidenceReceipt) -> None:
    """Store an evidence receipt with LRU eviction when capacity is exceeded."""
    _receipt_store[receipt.report_id] = receipt
    if len(_receipt_store) > MAX_RECEIPT_STORE_SIZE:
        _receipt_store.popitem(last=False)


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
        "version": __version__,
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
    if len(raw_b64) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image payload exceeds 20MB limit")
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
    except HTTPException:
        raise
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_message(str(e)))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to process image payload: {sanitize_error_message(str(e))}")



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
    provenance_dict = None

    if payload.image_base64:
        raw_b64 = payload.image_base64
        if len(raw_b64) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image payload exceeds 20MB limit")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Malformed base64 image data: {sanitize_error_message(str(e))}")

        try:
            rgb_arr, sha256_hash, iw, ih = validate_and_decode_image(raw_bytes)
        except ImageValidationError as e:
            raise HTTPException(status_code=400, detail=sanitize_error_message(str(e)))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Image decoding failed: {sanitize_error_message(str(e))}")

        if (ih, iw) != (h, w):
            raise HTTPException(
                status_code=422,
                detail=f"Dimension mismatch: uploaded image is {iw}x{ih}, but candidate coordinate frame is {w}x{h}."
            )

        input_sha256 = sha256_hash
        image_gray = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        human_mask = bbox_to_mask(payload.human_annotation.bbox, ih, iw)
        overlay = create_disagreement_visual_overlay(rgb_arr, human_mask, candidate_mask)
        overlay_data_uri = encode_image_to_base64_png(overlay)
        provenance_dict = {
            "category": "DECODED_IMAGE",
            "source_id": input_sha256,
            "verification_method": "DECODED_RGB_PIXELS",
            "dimensions": [iw, ih],
        }
    else:
        image_gray = np.full((h, w), 128, dtype=np.uint8)
        if not payload.is_blurry:
            image_gray[::4, ::4] = 240
            image_gray[1::4, 1::4] = 10
        provenance_dict = {
            "category": "SYNTHETIC_FIXTURE",
            "source_id": "SYNTHETIC_CANVAS",
            "verification_method": "SYNTHETIC_UNIFORM_PATTERN",
            "dimensions": [w, h],
        }

    extra_meas = {"source_image_sha256": input_sha256} if input_sha256 else {}

    result, receipt = auditor.audit_annotation(
        human_annotation=payload.human_annotation,
        candidate_mask=candidate_mask,
        image_gray=image_gray,
        independent_class_candidate=payload.class_candidate,
        extra_measurements=extra_meas,
        provenance=provenance_dict,
    )

    store_receipt(receipt)

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
    provenance_dict = None

    if payload.after_image_base64:
        raw_b64 = payload.after_image_base64
        if len(raw_b64) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image payload exceeds 20MB limit")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Malformed base64 image data: {sanitize_error_message(str(e))}")

        try:
            rgb_arr, sha256_hash, iw, ih = validate_and_decode_image(raw_bytes)
        except ImageValidationError as e:
            raise HTTPException(status_code=400, detail=sanitize_error_message(str(e)))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Image decoding failed: {sanitize_error_message(str(e))}")

        if (ih, iw) != (h, w):
            raise HTTPException(
                status_code=422,
                detail=f"Dimension mismatch: after-frame image is {iw}x{ih}, but coordinate frame is {w}x{h}."
            )

        input_sha256 = sha256_hash
        image_gray = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        provenance_dict = {
            "category": "DECODED_IMAGE",
            "source_id": input_sha256,
            "verification_method": "DECODED_RGB_PIXELS",
            "dimensions": [iw, ih],
        }
    else:
        image_gray = np.full((h, w), 128, dtype=np.uint8)
        if not payload.is_blurry:
            image_gray[::4, ::4] = 240
            image_gray[1::4, 1::4] = 10
        provenance_dict = {
            "category": "SYNTHETIC_FIXTURE",
            "source_id": "SYNTHETIC_CANVAS",
            "verification_method": "SYNTHETIC_UNIFORM_PATTERN",
            "dimensions": [w, h],
        }

    extra_meas = {"after_image_sha256": input_sha256} if input_sha256 else {}

    result, receipt = verifier.verify_action(
        request=payload.request,
        before_mask=before_mask,
        after_mask=after_mask,
        after_image_gray=image_gray,
        has_final_state=payload.has_final_state,
        extra_measurements=extra_meas,
        provenance=provenance_dict,
    )

    store_receipt(receipt)

    return {
        "verification_result": result.model_dump(),
        "evidence_receipt": receipt.model_dump(),
        "receipt_valid": verify_receipt_integrity(receipt),
    }


class ReCaptureFollowupRequest(BaseModel):
    task_id: str
    primary_object: str
    initial_verdict: str  # e.g. "UNCERTAIN"
    detected_defect: str  # e.g. "BLUR", "OCCLUSION", "BORDER_CLIPPING", "MISSING_REST_STATE"
    directive_issued: str
    original_before_bbox: BoundingBox
    followup_bbox: Optional[BoundingBox] = None
    target_region: BoundingBox
    initial_region: BoundingBox
    canvas_width: int = 640
    canvas_height: int = 480
    followup_laplacian_variance: float = 250.0
    followup_area_drop_ratio: float = 0.0
    followup_has_final_state: bool = True
    instruction_followed: bool = True
    effort_seconds: float = 5.0
    followup_image_base64: Optional[str] = None
    original_image_base64: Optional[str] = None


@router.post("/recapture/evaluate-followup")
def evaluate_recapture_followup(req: ReCaptureFollowupRequest) -> dict[str, Any]:
    """Evaluates a newly captured observation against previous uncertain verdict to measure decision transition."""
    engine = EvidenceCalculusEngine()
    spec = engine.build_standard_relocation_spec(
        task_id=f"{req.task_id}_FOLLOWUP",
        primary_object=req.primary_object,
        initial_region=req.initial_region,
        target_region=req.target_region,
    )

    followup_sha256 = None
    followup_var = req.followup_laplacian_variance

    if req.followup_image_base64:
        raw_b64 = req.followup_image_base64
        if len(raw_b64) > 20 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Image payload exceeds 20MB limit")
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Malformed followup base64: {sanitize_error_message(str(e))}")

        try:
            rgb_arr, sha256_hash, iw, ih = validate_and_decode_image(raw_bytes)
        except ImageValidationError as e:
            raise HTTPException(status_code=400, detail=sanitize_error_message(str(e)))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Followup image decoding failed: {sanitize_error_message(str(e))}")

        if (ih, iw) != (req.canvas_height, req.canvas_width):
            raise HTTPException(
                status_code=422,
                detail=f"Dimension mismatch: followup image is {iw}x{ih}, but canvas is {req.canvas_width}x{req.canvas_height}."
            )

        followup_sha256 = sha256_hash
        gray_arr = 0.299 * rgb_arr[:, :, 0] + 0.587 * rgb_arr[:, :, 1] + 0.114 * rgb_arr[:, :, 2]
        followup_var = float(compute_laplacian_variance_fast(gray_arr))
        provenance_mode = "VERIFIED_IMAGE_FOLLOWUP"
        provenance_dict = {
            "category": "DECODED_IMAGE",
            "source_id": followup_sha256,
            "verification_method": "DECODED_PIXELS_FAST_LAPLACIAN",
            "dimensions": [iw, ih],
            "measured_laplacian_variance": round(followup_var, 2),
        }
    else:
        provenance_mode = "METADATA_ONLY"
        provenance_dict = {
            "category": "USER_SUPPLIED_METADATA",
            "source_id": f"METADATA_{req.task_id}",
            "verification_method": "CLIENT_DECLARED_NUMERICAL_INPUTS",
            "note": "METADATA_ONLY: No follow-up media provided. Transition based on client metadata, not independently verified observation."
        }

    new_verdict, debt_report, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=req.original_before_bbox,
        after_bbox=req.followup_bbox,
        canvas_width=req.canvas_width,
        canvas_height=req.canvas_height,
        laplacian_variance=followup_var,
        area_drop_ratio=req.followup_area_drop_ratio,
        has_final_state=req.followup_has_final_state,
    )

    is_resolved = (new_verdict == ActionVerdict.SUCCESS) or (
        req.initial_verdict == "UNCERTAIN" and new_verdict != ActionVerdict.UNCERTAIN
    )
    transition = f"{req.initial_verdict} -> {new_verdict.value}"

    # Cryptographic Evidence Receipt for the ReCapture Transition
    report_id = f"RC-{uuid.uuid4().hex[:12].upper()}"
    media_tag = followup_sha256 or 'NOMEDIA'
    inputs_hash = calculate_sha256(f"{req.task_id}_{transition}_{req.effort_seconds:.1f}_{media_tag}")
    receipt = generate_receipt(
        report_id=report_id,
        engine_name="ReCapture Closed-Loop Recovery Engine",
        software_version="0.2.4",
        inputs_hash=inputs_hash,
        measurements={
            "task_id": req.task_id,
            "initial_verdict": req.initial_verdict,
            "new_verdict": new_verdict.value,
            "transition": transition,
            "resolved": is_resolved,
            "detected_defect": req.detected_defect,
            "directive_issued": req.directive_issued,
            "instruction_followed": req.instruction_followed,
            "effort_seconds": req.effort_seconds,
            "followup_laplacian_variance": round(followup_var, 2),
            "followup_area_drop_ratio": req.followup_area_drop_ratio,
            "provenance_mode": provenance_mode,
            "is_independently_verified_media": bool(followup_sha256 is not None),
            "followup_image_sha256": followup_sha256,
        },
        signals=[],
        verdict=new_verdict.value,
        recapture_directive=req.directive_issued,
        provenance=provenance_dict,
    )
    store_receipt(receipt)

    return {
        "report_id": report_id,
        "task_id": req.task_id,
        "initial_verdict": req.initial_verdict,
        "new_verdict": new_verdict.value,
        "verdict_transition": transition,
        "resolved": is_resolved,
        "instruction_followed": req.instruction_followed,
        "effort_seconds": req.effort_seconds,
        "provenance_mode": provenance_mode,
        "debt_report": debt_report.model_dump(),
        "structured_finding": updated_spec.structured_finding,
        "evidence_receipt_id": report_id,
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
    if not re.match(r"^[A-Za-z0-9_\-]{1,64}$", report_id):
        raise HTTPException(status_code=400, detail="Invalid report_id format.")
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

    coco_file = Path(__file__).resolve().parent.parent.parent / "evaluation" / "experiments" / "LABELCOURT_REAL_BENCHMARK_RESULTS.json"
    coco_data = None
    if coco_file.exists():
        with open(coco_file, "r") as f:
            coco_data = json.load(f)
    genuine_file = Path(__file__).resolve().parent.parent.parent / "evaluation" / "experiments" / "genuine_model_predictions" / "summary.json"
    genuine_data = None
    if genuine_file.exists():
        with open(genuine_file, "r") as f:
            genuine_data = json.load(f)

    if dataset == "synthetic":
        return syn_data or {}
    if dataset == "real":
        return real_data or {}
    if dataset == "coco":
        return coco_data or {}

    return {
        "synthetic_baseline": syn_data,
        "real_world_evaluation": real_data,
        "coco_photographic_benchmark": coco_data,
        "genuine_model_predictions": genuine_data,
        "metadata": {
            "disclaimer": "Synthetic baseline establishes mathematical rule consistency; real evaluation tests physical tabletop pairs.",
            "metrics_reporting_standard": "Controlled scenarios and genuine model predictions are separate; correlated cases use image-grouped bootstrap intervals.",
        },
    }


@router.get("/benchmark/coco-real")
def get_coco_real_benchmark() -> dict[str, Any]:
    """Return authentic photographic COCO 2017 validation benchmark results."""
    coco_file = Path(__file__).resolve().parent.parent.parent / "evaluation" / "experiments" / "LABELCOURT_REAL_BENCHMARK_RESULTS.json"
    if coco_file.exists():
        with open(coco_file, "r") as f:
            return json.load(f)
    return {"status": "benchmark_not_run"}


@router.get("/benchmark/latency")
def get_latency_benchmark() -> dict[str, Any]:
    """Return NumPy baseline vs Rust native acceleration latency profiles."""
    bench_file = Path(__file__).resolve().parent.parent.parent / "docs" / "benchmarks" / "benchmark_latency_comparison.json"
    if not bench_file.exists():
        from evaluation.benchmark_rust_speedup import run_latency_benchmark
        return run_latency_benchmark()

    with open(bench_file, "r") as f:
        return json.load(f)


# ==============================================================================
# EVIDENCE CALCULUS ENGINE ENDPOINTS (Phase 2 Architecture)
# ==============================================================================

class CalculusEvaluateRequest(BaseModel):
    task_id: str = "TASK_RELOCATE_001"
    primary_object: str = "cup"
    before_bbox: BoundingBox
    after_bbox: Optional[BoundingBox] = None
    target_region: Optional[BoundingBox] = None
    initial_region: Optional[BoundingBox] = None
    canvas_width: int = 640
    canvas_height: int = 480
    laplacian_variance: float = 250.0
    area_drop_ratio: float = 0.0
    has_final_state: bool = True


@router.post("/calculus/evaluate")
def evaluate_evidence_calculus(req: CalculusEvaluateRequest) -> dict[str, Any]:
    """Evaluates task predicates, Evidence Debt D(E,T), and grounded tri-state verdict."""
    engine = EvidenceCalculusEngine()
    target = req.target_region or BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)
    init_reg = req.initial_region or req.before_bbox

    spec = engine.build_standard_relocation_spec(
        task_id=req.task_id,
        primary_object=req.primary_object,
        initial_region=init_reg,
        target_region=target,
    )

    verdict, debt_report, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=req.before_bbox,
        after_bbox=req.after_bbox,
        canvas_width=req.canvas_width,
        canvas_height=req.canvas_height,
        laplacian_variance=req.laplacian_variance,
        area_drop_ratio=req.area_drop_ratio,
        has_final_state=req.has_final_state,
    )

    # Optional auto-plan if debt exists
    planner = EvidenceBudgetPlanner()
    recommended_action = planner.plan_recovery_action(debt_report, updated_spec.predicates)

    return {
        "verdict": verdict.value,
        "debt_report": debt_report.model_dump(),
        "task_specification": updated_spec.model_dump(),
        "predicates": [p.model_dump() for p in updated_spec.predicates],
        "recommended_action": recommended_action.model_dump() if recommended_action else None,
    }


@router.get("/calculus/graph")
def get_canonical_evidence_graph() -> dict[str, Any]:
    """Returns canonical 6-stage Evidence Dependency Graph nodes and edges for visualization."""
    g = build_canonical_task_graph()
    return g.export_graph()


class PlanRecoveryRequest(BaseModel):
    predicates: List[ObservablePredicate]
    debt_report: EvidenceDebtReport
    budget_limit_sec: Optional[float] = None


@router.post("/calculus/plan-recovery")
def plan_evidence_recovery(req: PlanRecoveryRequest) -> dict[str, Any]:
    """Ranks and selects recovery actions maximizing Value of Information (VOI)."""
    planner = EvidenceBudgetPlanner()
    all_candidates = planner.generate_all_candidate_actions(req.predicates)
    best_action = planner.plan_recovery_action(req.debt_report, req.predicates, req.budget_limit_sec)

    return {
        "best_action": best_action.model_dump() if best_action else None,
        "all_candidate_actions": [a.model_dump() for a in all_candidates],
        "candidate_count": len(all_candidates),
    }


@router.get("/calculus/experiments")
def get_calculus_experiments_summary() -> dict[str, Any]:
    """Returns empirical research experiment results across H01, H02, H03, and H04."""
    exp_file = Path(__file__).resolve().parent.parent.parent / "evaluation" / "experiments" / "EXPERIMENT_RESULTS.json"
    if exp_file.exists():
        with open(exp_file, "r") as f:
            return json.load(f)
    return {"status": "Experiments file pending execution"}


# ==============================================================================
# INTEGRATED NEURAL VISION AUDIT WORKFLOW (Phase 7 Architecture)
# ==============================================================================

_coco_pilot_cache: Optional[list[dict[str, Any]]] = None


@router.get("/vision/samples")
def get_vision_samples() -> list[dict[str, Any]]:
    """Return authentic photographic COCO 2017 validation pilot samples with annotations."""
    global _coco_pilot_cache
    if _coco_pilot_cache is not None:
        return _coco_pilot_cache

    pilot_dir = Path(__file__).resolve().parent.parent.parent / "data" / "coco_val2017_subset"
    ann_file = pilot_dir / "annotations_coco_pilot20.json"
    if not ann_file.exists():
        return []

    with open(ann_file, "r") as f:
        coco_data = json.load(f)

    cat_map = {c["id"]: c["name"] for c in coco_data.get("categories", [])}
    img_map = {i["id"]: i for i in coco_data.get("images", [])}

    target_anns = [
        (397133, 82445, "Authentic COCO val2017: Kitchen dining table with glass bottle"),
        (397133, 119568, "Authentic COCO val2017: Dining table foreground surface"),
        (174482, 128189, "Authentic COCO val2017: Bicycle parked in outdoor street scene"),
        (87038, 125737, "Authentic COCO val2017: Bicycle alongside urban pedestrian street"),
        (296649, 134920, "Authentic COCO val2017: Car parked along roadside curb"),
        (37777, 102453, "Authentic COCO val2017: Indoor dining chair"),
    ]

    samples: list[dict[str, Any]] = []
    anns_by_id = {a["id"]: a for a in coco_data.get("annotations", [])}

    for img_id, ann_id, desc in target_anns:
        ann = anns_by_id.get(ann_id)
        img_info = img_map.get(img_id)
        if not ann or not img_info:
            continue
        img_path = pilot_dir / img_info["file_name"]
        if not img_path.exists():
            img_path = pilot_dir / "images" / img_info["file_name"]
        if not img_path.exists():
            continue

        with open(img_path, "rb") as f:
            raw_bytes = f.read()
            b64_img = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('ascii')}"

        bbox_raw = ann["bbox"]  # [x, y, w, h]
        bbox_obj = {
            "x_min": float(bbox_raw[0]),
            "y_min": float(bbox_raw[1]),
            "x_max": float(bbox_raw[0] + bbox_raw[2]),
            "y_max": float(bbox_raw[1] + bbox_raw[3]),
        }
        cat_name = cat_map.get(ann["category_id"], "unknown")

        samples.append({
            "sample_id": f"coco_sample_{img_id}_{ann_id}",
            "manifest_image_id": img_id,
            "file_name": img_info["file_name"],
            "image_width": img_info["width"],
            "image_height": img_info["height"],
            "description": desc,
            "annotation": {
                "id": ann["id"],
                "category_id": ann["category_id"],
                "category_name": cat_name,
                "bbox": bbox_obj,
                "segmentation": ann.get("segmentation"),
            },
            "image_base64": b64_img,
            "perturbation_presets": {
                "authentic": {
                    "label": "Authentic COCO Ground Truth",
                    "category_name": cat_name,
                    "bbox": bbox_obj,
                    "segmentation": ann.get("segmentation"),
                },
                "shifted_25px": {
                    "label": "Perturbed: Spatial Shift (+25px)",
                    "category_name": cat_name,
                    "bbox": {
                        "x_min": min(float(img_info["width"] - 1), bbox_obj["x_min"] + 25.0),
                        "y_min": min(float(img_info["height"] - 1), bbox_obj["y_min"] + 25.0),
                        "x_max": min(float(img_info["width"]), bbox_obj["x_max"] + 25.0),
                        "y_max": min(float(img_info["height"]), bbox_obj["y_max"] + 25.0),
                    },
                    "segmentation": [[pt + 25.0 for pt in poly] for poly in ann.get("segmentation", [])] if (isinstance(ann.get("segmentation"), list) and ann.get("segmentation") and isinstance(ann.get("segmentation")[0], list)) else None,
                },
                "category_mismatch": {
                    "label": "Perturbed: Incorrect Class Label",
                    "category_name": "airplane" if cat_name != "airplane" else "person",
                    "bbox": bbox_obj,
                    "segmentation": ann.get("segmentation"),
                },
            },
        })

    _coco_pilot_cache = samples
    return samples


@router.post("/vision/audit", response_model=VisionAuditResponse)
def audit_vision_annotation(req: VisionAuditRequest) -> VisionAuditResponse:
    """Executes genuine end-to-end vision audit: image -> Mask2Former inference -> matching -> LabelCourt -> Evidence Receipt.

    Scientific guarantees:
    - Original photograph pixels are validated and passed to Mask2Former.
    - Reference annotations are NEVER passed to the segmentation model (zero leakage).
    - Disagreements and uncertainties produce ACCEPT, REVIEW, or UNCERTAIN verdicts with explicit causal reasons.
    - An unkeyed SHA-256 tamper-evident Evidence Receipt is generated and verified before response.
    - Full disaggregated wall-clock latency is measured from request arrival to response return.
    """
    t_start = time.perf_counter()

    # 1. Image decoding & validation
    b64_str = req.image_base64.strip()
    if len(b64_str) > 20 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image payload exceeds maximum allowed size limit of 20MB base64.",
        )
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        raw_bytes = base64.b64decode(b64_str)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

    try:
        rgb_arr, image_sha256, width, height = validate_and_decode_image(raw_bytes)
    except ImageValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Authoritative server-side provenance verification (Issue 1 Trust-Boundary Remediation)
    prov_result = verify_authoritative_provenance(
        image_bytes=raw_bytes,
        image_sha256=image_sha256,
        width=width,
        height=height,
        claimed_source_type=req.image_source_type,
        manifest_image_id=req.manifest_image_id,
        annotation=req.annotation,
        claimed_annotation_source=getattr(req, "annotation_source_type", None),
        dataset_manifest_ref=req.dataset_manifest_ref,
    )
    if prov_result.failure_reason is not None:
        raise HTTPException(
            status_code=422,
            detail=f"Provenance verification failed: {prov_result.failure_reason}",
        )

    t_decode = time.perf_counter()

    # 2. Preprocessing & Quality / Uncertainty Evaluation
    settings = req.settings or VisionAuditSettings()

    # Grayscale conversion for blur analysis
    image_gray = np.dot(rgb_arr[..., :3].astype(np.float32), [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    laplacian_var = float(compute_laplacian_variance_fast(image_gray))

    signals: list[UncertaintySignal] = []
    is_blurry = laplacian_var < settings.blur_threshold
    if is_blurry:
        signals.append(
            UncertaintySignal(
                reason=UncertaintyReason.BLUR,
                severity="HIGH",
                metric_name="laplacian_variance",
                metric_value=round(laplacian_var, 2),
                threshold=settings.blur_threshold,
                description=f"Severe visual blur: Laplacian variance {laplacian_var:.1f} is below threshold {settings.blur_threshold}.",
            )
        )

    # Reference mask rasterization
    if req.annotation.segmentation:
        reference_mask = parse_coco_polygon_to_mask(req.annotation.segmentation, width, height)
    else:
        reference_mask = bbox_to_mask(req.annotation.bbox, height, width)

    # Border proximity check
    margin = settings.boundary_margin_px
    touches_border = bool(
        np.any(reference_mask[:margin, :])
        or np.any(reference_mask[-margin:, :])
        or np.any(reference_mask[:, :margin])
        or np.any(reference_mask[:, -margin:])
    )
    if touches_border:
        signals.append(
            UncertaintySignal(
                reason=UncertaintyReason.OCCLUSION,
                severity="MEDIUM",
                metric_name="touches_border",
                metric_value=1.0,
                threshold=0.0,
                description="Reference annotation touches image boundary; partial frame clipping introduces uncertainty.",
            )
        )

    t_preproc = time.perf_counter()

    # 3. Model Inference (Ground truth excluded structurally)
    try:
        provider = get_local_mask2former_provider()
        predictions, execution = provider.predict_from_rgb(
            rgb_arr=rgb_arr,
            threshold=settings.confidence_threshold,
            raw_bytes=raw_bytes,
        )
    except Exception as e:
        correlation_id = f"INF-{uuid.uuid4().hex[:12].upper()}"
        logger.error(
            "Mask2Former inference execution failed [correlation_id=%s]: %s: %s",
            correlation_id,
            type(e).__name__,
            e,
            exc_info=True,
        )
        safe_detail = sanitize_error_message(e)
        raise HTTPException(
            status_code=500,
            detail=f"Mask2Former inference execution failed (Request ID: {correlation_id}): {safe_detail}",
        )

    t_inference = time.perf_counter()

    # 4. Mask Postprocessing & Instance Matching
    best_pred: Optional[InstancePrediction] = None
    best_match_instance: Optional[MatchedModelInstance] = None
    best_iou = 0.0
    best_boundary = 0.0
    class_match = False
    candidate_mask = np.zeros((height, width), dtype=bool)

    evaluated_instances: list[dict[str, Any]] = []

    for pred in predictions:
        iou = float(compute_mask_iou_fast(reference_mask, pred.mask))
        boundary = float(calculate_boundary_agreement(reference_mask, pred.mask))
        is_cls_match = (pred.label.strip().lower() == req.annotation.category_name.strip().lower())

        cand_indices = np.where(pred.mask)
        if len(cand_indices[0]) > 0:
            c_bbox = BoundingBox(
                x_min=float(np.min(cand_indices[1])),
                y_min=float(np.min(cand_indices[0])),
                x_max=float(np.max(cand_indices[1]) + 1),
                y_max=float(np.max(cand_indices[0]) + 1),
            )
        else:
            c_bbox = BoundingBox(x_min=0.0, y_min=0.0, x_max=0.0, y_max=0.0)

        rle_str = encode_binary_mask_rle(pred.mask)
        inst_summary = {
            "instance_id": pred.instance_id,
            "label_id": pred.label_id,
            "label": pred.label,
            "score": round(pred.score, 4),
            "iou_with_reference": round(iou, 4),
            "boundary_agreement": round(boundary, 4),
            "class_match": is_cls_match,
            "bbox": c_bbox.model_dump(),
            "rle": rle_str,
        }
        evaluated_instances.append(inst_summary)

    # Matching Strategy: Select best candidate
    if predictions:
        if settings.matching_strategy == "class_aware":
            matching_class_preds = [
                (p, evaluated_instances[idx])
                for idx, p in enumerate(predictions)
                if evaluated_instances[idx]["class_match"]
            ]
            if matching_class_preds:
                matching_class_preds.sort(key=lambda item: item[1]["iou_with_reference"], reverse=True)
                best_pred, best_dict = matching_class_preds[0]
            else:
                sorted_by_iou = sorted(
                    zip(predictions, evaluated_instances),
                    key=lambda item: item[1]["iou_with_reference"],
                    reverse=True,
                )
                best_pred, best_dict = sorted_by_iou[0]
        else:
            sorted_by_iou = sorted(
                zip(predictions, evaluated_instances),
                key=lambda item: item[1]["iou_with_reference"],
                reverse=True,
            )
            best_pred, best_dict = sorted_by_iou[0]

        if best_pred is not None:
            candidate_mask = best_pred.mask
            best_iou = best_dict["iou_with_reference"]
            best_boundary = best_dict["boundary_agreement"]
            class_match = best_dict["class_match"]
            best_match_instance = MatchedModelInstance(
                instance_id=best_pred.instance_id,
                label=best_pred.label,
                label_id=best_pred.label_id,
                score=round(best_pred.score, 4),
                iou_with_reference=round(best_iou, 4),
                boundary_agreement=round(best_boundary, 4),
                class_match=class_match,
                bbox=BoundingBox(**best_dict["bbox"]),
                rle=best_dict["rle"],
            )

    t_postproc = time.perf_counter()

    # 5. LabelCourt Evaluation & Grounded Verdict Determination
    if is_blurry:
        verdict = VisionAuditVerdict.UNCERTAIN
        verdict_reason = "HIGH_VISUAL_BLUR"
        verdict_explanation = (
            f"Image Laplacian blur variance ({laplacian_var:.1f}) is below threshold "
            f"({settings.blur_threshold}). Visual evidence is degraded and inconclusive."
        )
    elif len(predictions) == 0:
        verdict = VisionAuditVerdict.REVIEW
        verdict_reason = "MODEL_ZERO_DETECTIONS"
        verdict_explanation = (
            f"Mask2Former detected no instances at confidence threshold >= {settings.confidence_threshold:.2f}. "
            "Annotation requires manual review."
        )
    elif not class_match:
        verdict = VisionAuditVerdict.REVIEW
        verdict_reason = "CLASS_MISMATCH"
        verdict_explanation = (
            f"Human annotation category '{req.annotation.category_name}' disagrees with predicted model class "
            f"'{best_pred.label}' (IoU={best_iou:.3f}, confidence={best_pred.score:.2f})."
        )
    elif touches_border and best_iou < settings.iou_accept_threshold:
        verdict = VisionAuditVerdict.UNCERTAIN
        verdict_reason = "BORDER_TRUNCATION_UNCERTAIN"
        verdict_explanation = (
            f"Annotation contacts image canvas border and predicted IoU ({best_iou:.3f}) is below accept "
            f"threshold ({settings.iou_accept_threshold:.2f}). Boundary truncation prevents conclusive audit."
        )
    elif best_iou >= settings.iou_accept_threshold and best_boundary >= settings.boundary_agreement_threshold:
        verdict = VisionAuditVerdict.ACCEPT
        verdict_reason = "CONCORDANT_SEGMENTATION"
        verdict_explanation = (
            f"Model prediction is highly concordant with reference annotation "
            f"(IoU={best_iou:.3f} >= {settings.iou_accept_threshold:.2f}, "
            f"Boundary={best_boundary:.3f} >= {settings.boundary_agreement_threshold:.2f})."
        )
    elif best_iou >= settings.iou_review_threshold:
        verdict = VisionAuditVerdict.REVIEW
        verdict_reason = "PARTIAL_DISAGREEMENT"
        verdict_explanation = (
            f"Model prediction partially disagrees with reference annotation "
            f"(IoU={best_iou:.3f} in [{settings.iou_review_threshold:.2f}, {settings.iou_accept_threshold:.2f}]). "
            f"Boundary agreement is {best_boundary:.3f}. Manual review required."
        )
    else:
        verdict = VisionAuditVerdict.REVIEW
        verdict_reason = "LOW_SPATIAL_OVERLAP"
        verdict_explanation = (
            f"Predicted candidate has insufficient overlap with annotation "
            f"(IoU={best_iou:.3f} < {settings.iou_review_threshold:.2f})."
        )

    t_court = time.perf_counter()

    # 6. Visual Overlays
    disagreement_arr = create_disagreement_visual_overlay(rgb_arr, reference_mask, candidate_mask)
    ref_overlay_arr = create_single_mask_overlay(rgb_arr, reference_mask, color_rgb=(255, 180, 0), alpha=0.45)
    pred_overlay_arr = create_single_mask_overlay(rgb_arr, candidate_mask, color_rgb=(0, 220, 255), alpha=0.45)

    disagreement_b64 = encode_image_to_base64_png(disagreement_arr)
    ref_overlay_b64 = encode_image_to_base64_png(ref_overlay_arr)
    pred_overlay_b64 = encode_image_to_base64_png(pred_overlay_arr)

    t_overlays = time.perf_counter()

    # 7. Tamper-evident Evidence Receipt & Provenance
    report_id = f"VREC-{uuid.uuid4().hex[:12].upper()}"

    provenance: dict[str, Any] = {
        "source_image_hash": image_sha256,
        "image_source_type": prov_result.image_source_type,
        "provenance_category": prov_result.category,
        "is_independently_verified": prov_result.is_verified,
        "canonical_source_url": prov_result.canonical_source_url,
        "license_status": prov_result.license_status,
        "license_name": prov_result.license_name,
        "license_url": prov_result.license_url,
        "redistribution_status": prov_result.redistribution_status,
        "annotation_source_type": prov_result.annotation_source_type,
        "annotation_verified": prov_result.annotation_verified,
        "image_dimensions": {"width": width, "height": height},
        "manifest_image_id": prov_result.manifest_image_id,
        "human_annotation_hash": calculate_sha256(canonical_json_dump(req.annotation.model_dump())),
        "model_name": execution.model_identifier,
        "model_revision": execution.model_revision,
        "checkpoint_sha256": execution.checkpoint_sha256,
        "device": execution.device,
        "inference_mode": execution.inference_mode,
        "output_sha256": execution.output_sha256,
        "rust_accelerated": is_rust_accelerator_available(),
    }

    measurements = {
        "iou": round(best_iou, 4),
        "boundary_agreement": round(best_boundary, 4),
        "class_match": class_match,
        "laplacian_variance": round(laplacian_var, 2),
        "touches_border": touches_border,
        "detected_instances_count": len(predictions),
        "best_confidence": round(best_pred.score, 4) if best_pred else 0.0,
    }

    inputs_hash = calculate_sha256(
        f"{image_sha256}:{req.annotation.id}:{req.annotation.category_name}:{best_iou:.4f}"
    )

    receipt = generate_receipt(
        report_id=report_id,
        engine_name="LabelCourt Real Vision Inference Auditor",
        software_version=__version__,
        inputs_hash=inputs_hash,
        measurements=measurements,
        signals=signals,
        verdict=verdict.value,
        recapture_directive=None,
        provenance=provenance,
    )

    receipt_is_valid = verify_receipt_integrity(receipt)
    if not receipt_is_valid:
        raise HTTPException(status_code=500, detail="Generated evidence receipt failed integrity verification.")

    store_receipt(receipt)
    t_receipt = time.perf_counter()

    # 8. Full wall-clock disaggregated latency
    t_total = time.perf_counter()
    latency_breakdown = {
        "image_decoding_ms": round((t_decode - t_start) * 1000.0, 2),
        "preprocessing_ms": round((t_preproc - t_decode) * 1000.0, 2),
        "model_inference_ms": round((t_inference - t_preproc) * 1000.0, 2),
        "mask_postprocessing_ms": round((t_postproc - t_inference) * 1000.0, 2),
        "labelcourt_evaluation_ms": round((t_court - t_postproc) * 1000.0, 2),
        "overlay_generation_ms": round((t_overlays - t_court) * 1000.0, 2),
        "receipt_generation_ms": round((t_receipt - t_overlays) * 1000.0, 2),
        "total_wall_clock_ms": round((t_total - t_start) * 1000.0, 2),
    }

    return VisionAuditResponse(
        verdict=verdict,
        verdict_reason=verdict_reason,
        explanation=verdict_explanation,
        metrics=measurements,
        matched_prediction=best_match_instance,
        detected_instances_count=len(predictions),
        all_detected_instances=evaluated_instances,
        model_execution={
            "model_name": execution.model_identifier,
            "model_revision": execution.model_revision,
            "device": execution.device,
            "inference_mode": execution.inference_mode,
            "checkpoint_sha256": execution.checkpoint_sha256,
            "checkpoint_size_bytes": execution.checkpoint_size_bytes,
            "output_sha256": execution.output_sha256,
            "runtime_ms": execution.latency_ms,
        },
        evidence_receipt=receipt.model_dump(),
        receipt_verified=receipt_is_valid,
        provenance=provenance,
        latency_breakdown_ms=latency_breakdown,
        visual_overlays={
            "disagreement": disagreement_b64,
            "reference": ref_overlay_b64,
            "predicted": pred_overlay_b64,
        },
    )

