from __future__ import annotations

import uuid
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
from backend.evidence.receipts import calculate_sha256, generate_receipt, verify_receipt_integrity
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
        "version": "0.2.5",
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
    provenance_dict = None

    if payload.image_base64:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Malformed base64 image data: {e}")

        try:
            rgb_arr, sha256_hash, iw, ih = validate_and_decode_image(raw_bytes)
        except ImageValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Image decoding failed: {e}")

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
    provenance_dict = None

    if payload.after_image_base64:
        raw_b64 = payload.after_image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Malformed base64 image data: {e}")

        try:
            rgb_arr, sha256_hash, iw, ih = validate_and_decode_image(raw_bytes)
        except ImageValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Image decoding failed: {e}")

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

    _receipt_store[receipt.report_id] = receipt

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
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        try:
            raw_bytes = base64.b64decode(raw_b64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Malformed followup base64: {e}")

        try:
            rgb_arr, sha256_hash, iw, ih = validate_and_decode_image(raw_bytes)
        except ImageValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Followup image decoding failed: {e}")

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
    _receipt_store[report_id] = receipt

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
