"""Tests for the Evidence Calculus Engine, Evidence Debt, and Phase 3 Bug Fixes."""

import pytest
from backend.evidence.calculus.engine import EvidenceCalculusEngine
from backend.evidence.calculus.schemas import (
    EvidenceDebtReport,
    EvidenceState,
    ObservablePredicate,
    PredicateType,
    TaskSpecification,
)
from backend.evidence.schemas import ActionVerdict, BoundingBox


def test_standard_relocation_spec_generation():
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    target_box = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)

    spec = engine.build_standard_relocation_spec(
        task_id="TASK_001",
        primary_object="cup",
        initial_region=initial_box,
        target_region=target_box,
        distance_threshold_px=75.0,
    )

    assert spec.task_id == "TASK_001"
    assert spec.distance_threshold_px == 75.0
    assert len(spec.predicates) == 8
    pred_types = [p.predicate_type for p in spec.predicates]
    assert PredicateType.OBJECT_IDENTIFICATION in pred_types
    assert PredicateType.INITIAL_STATE_OBSERVED in pred_types
    assert PredicateType.FINAL_STATE_OBSERVED in pred_types
    assert PredicateType.SPATIAL_RELATION in pred_types
    assert PredicateType.OBSERVATION_SHARPNESS in pred_types
    assert PredicateType.CANVAS_CONTAINMENT in pred_types
    assert PredicateType.SURFACE_VISIBILITY in pred_types
    assert PredicateType.PHYSICS_NO_CONTRADICTION in pred_types

    # Spatial predicate should reflect configured threshold
    p_spatial = [p for p in spec.predicates if p.predicate_type == PredicateType.SPATIAL_RELATION][0]
    assert p_spatial.threshold == 75.0


def test_evidence_debt_and_success_verdict():
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    target_box = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)

    spec = engine.build_standard_relocation_spec(
        task_id="TASK_SUCCESS",
        primary_object="cup",
        initial_region=initial_box,
        target_region=target_box,
    )

    # Valid final state inside target zone
    final_box = BoundingBox(x_min=310, y_min=105, x_max=390, y_max=195)

    verdict, debt_report, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=initial_box,
        after_bbox=final_box,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=250.0,
    )

    assert verdict == ActionVerdict.SUCCESS
    assert debt_report.total_debt == 0.0
    assert debt_report.is_acceptable is True
    assert len(debt_report.contradictions) == 0


def test_missing_evidence_yields_uncertain_and_debt():
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    target_box = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)

    spec = engine.build_standard_relocation_spec(
        task_id="TASK_MISSING",
        primary_object="cup",
        initial_region=initial_box,
        target_region=target_box,
    )

    # Final box is missing (e.g. object occluded or unobserved)
    verdict, debt_report, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=initial_box,
        after_bbox=None,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=250.0,
        has_final_state=False,
    )

    assert verdict == ActionVerdict.UNCERTAIN
    assert debt_report.total_debt > 0.0
    assert debt_report.missing_evidence_debt > 0.0
    assert debt_report.is_acceptable is False
    assert len(debt_report.critical_unresolved_predicates) > 0


def test_contradiction_falsification_yields_failed():
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    target_box = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)

    spec = engine.build_standard_relocation_spec(
        task_id="TASK_FAILED",
        primary_object="cup",
        initial_region=initial_box,
        target_region=target_box,
    )

    # Final box is observed, but placed outside target zone (spatial relation contradicted)
    final_box = BoundingBox(x_min=50, y_min=350, x_max=150, y_max=450)

    verdict, debt_report, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=initial_box,
        after_bbox=final_box,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=250.0,
    )

    assert verdict == ActionVerdict.FAILED
    assert debt_report.contradictory_evidence_debt > 0.0
    assert len(debt_report.contradictions) > 0


def test_pixel_displacement_anomaly_heuristic():
    """Tests Issue B: Pixel displacement anomaly heuristic replaces misleading impossibility claim."""
    engine = EvidenceCalculusEngine(max_physical_speed_px_per_frame=200.0)
    initial_box = BoundingBox(x_min=50, y_min=50, x_max=100, y_max=100)
    target_box = BoundingBox(x_min=500, y_min=400, x_max=550, y_max=450)

    spec = engine.build_standard_relocation_spec(
        task_id="TASK_DISP_ANOMALY",
        primary_object="cup",
        initial_region=initial_box,
        target_region=target_box,
    )

    # Object displaced by > 500px in a single frame transition with max speed 200px
    verdict, debt_report, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=initial_box,
        after_bbox=target_box,
        canvas_width=800,
        canvas_height=600,
        laplacian_variance=200.0,
    )

    assert verdict == ActionVerdict.FAILED
    contra_text = " ".join(debt_report.contradictions)
    assert "Pixel displacement anomaly detected" in contra_text
    assert "not calibrated metric physics proof" in contra_text


def test_blur_threshold_abstention():
    engine = EvidenceCalculusEngine(blur_threshold=100.0)
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    target_box = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)

    spec = engine.build_standard_relocation_spec(
        task_id="TASK_BLURRY",
        primary_object="cup",
        initial_region=initial_box,
        target_region=target_box,
    )

    final_box = BoundingBox(x_min=305, y_min=105, x_max=395, y_max=195)

    # Final frame is blurry (blur variance 42.0 < threshold 100.0)
    verdict, debt_report, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=initial_box,
        after_bbox=final_box,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=42.0,
    )

    assert verdict == ActionVerdict.UNCERTAIN
    assert debt_report.missing_evidence_debt > 0.0
    unresolved_text = " ".join(debt_report.critical_unresolved_predicates)
    assert "OBSERVATION_SHARPNESS" in unresolved_text
    # Final rest state also flagged as unconfirmed due to blur (Issue D fix)
    assert "FINAL_STATE_OBSERVED" in unresolved_text


# ==============================================================================
# PHASE 3 REGRESSION TESTS FOR ISSUES A, B, C, D
# ==============================================================================

def test_configurable_displacement_threshold_below_at_above():
    """Tests Issue A: Exactly below, on, and above threshold, plus zero displacement."""
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)  # centroid: (150, 150)
    # Target region centered at (350, 150)
    target_region = BoundingBox(x_min=200, y_min=50, x_max=500, y_max=250)

    # Spec configured with distance_threshold_px = 100.0
    spec = engine.build_standard_relocation_spec(
        task_id="TASK_THRESH_EVAL",
        primary_object="tool",
        initial_region=initial_box,
        target_region=target_region,
        distance_threshold_px=100.0,
    )

    # 1. Below threshold (centroid at (249, 150) -> dist = 99.0px < 100.0px)
    box_below = BoundingBox(x_min=199, y_min=100, x_max=299, y_max=200)
    v_below, debt_below, _ = engine.evaluate_task(
        spec=spec, before_bbox=initial_box, after_bbox=box_below, canvas_width=640, canvas_height=480
    )
    assert v_below == ActionVerdict.FAILED
    assert any("below configured relocation threshold 100.0px" in c for c in debt_below.contradictions)

    # 2. Exactly on threshold (centroid at (250, 150) -> dist = 100.0px == 100.0px)
    box_on = BoundingBox(x_min=200, y_min=100, x_max=300, y_max=200)
    v_on, debt_on, _ = engine.evaluate_task(
        spec=spec, before_bbox=initial_box, after_bbox=box_on, canvas_width=640, canvas_height=480
    )
    assert v_on == ActionVerdict.SUCCESS
    assert debt_on.total_debt == 0.0

    # 3. Above threshold (centroid at (251, 150) -> dist = 101.0px > 100.0px)
    box_above = BoundingBox(x_min=201, y_min=100, x_max=301, y_max=200)
    v_above, debt_above, _ = engine.evaluate_task(
        spec=spec, before_bbox=initial_box, after_bbox=box_above, canvas_width=640, canvas_height=480
    )
    assert v_above == ActionVerdict.SUCCESS
    assert debt_above.total_debt == 0.0

    # 4. Zero displacement (after == initial)
    v_zero, debt_zero, _ = engine.evaluate_task(
        spec=spec, before_bbox=initial_box, after_bbox=initial_box, canvas_width=640, canvas_height=480
    )
    assert v_zero == ActionVerdict.FAILED
    assert any("Zero displacement observed" in c for c in debt_zero.contradictions)


def test_invalid_displacement_threshold_rejection():
    """Tests Issue A: Non-positive or NaN displacement threshold must be rejected."""
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50)
    target_box = BoundingBox(x_min=100, y_min=100, x_max=150, y_max=150)

    with pytest.raises(ValueError, match="Invalid displacement threshold"):
        engine.build_standard_relocation_spec("T_ZERO", "tool", initial_box, target_box, distance_threshold_px=0.0)

    with pytest.raises(ValueError, match="Invalid displacement threshold"):
        engine.build_standard_relocation_spec("T_NEG", "tool", initial_box, target_box, distance_threshold_px=-15.0)


def test_object_identity_qualification():
    """Tests Issue C: Bounding box coordinates alone do not claim verified object identity."""
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    target_box = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)
    spec = engine.build_standard_relocation_spec("T_ID", "cup", initial_box, target_box)

    final_box = BoundingBox(x_min=310, y_min=105, x_max=390, y_max=195)
    _, _, updated_spec = engine.evaluate_task(
        spec=spec, before_bbox=initial_box, after_bbox=final_box, canvas_width=640, canvas_height=480
    )

    p_ident = [p for p in updated_spec.predicates if p.predicate_type == PredicateType.OBJECT_IDENTIFICATION][0]
    assert p_ident.state == EvidenceState.SUPPORTED
    assert "Geometric localization observed" in p_ident.rationale
    assert "semantic object identity is assumed from proposal" in p_ident.rationale


import pytest


def test_nonfinite_threshold_rejection():
    engine = EvidenceCalculusEngine()
    initial_box = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    target_box = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)

    # NaN displacement threshold
    with pytest.raises(ValueError, match="displacement threshold"):
        engine.build_standard_relocation_spec("T_NAN", "cup", initial_box, target_box, distance_threshold_px=float("nan"))

    # Positive Infinity displacement threshold
    with pytest.raises(ValueError, match="displacement threshold"):
        engine.build_standard_relocation_spec("T_INF", "cup", initial_box, target_box, distance_threshold_px=float("inf"))

    # Negative Infinity displacement threshold
    with pytest.raises(ValueError, match="displacement threshold"):
        engine.build_standard_relocation_spec("T_NEGINF", "cup", initial_box, target_box, distance_threshold_px=float("-inf"))

    # Zero displacement threshold
    with pytest.raises(ValueError, match="displacement threshold"):
        engine.build_standard_relocation_spec("T_ZERO", "cup", initial_box, target_box, distance_threshold_px=0.0)

    # Negative displacement threshold
    with pytest.raises(ValueError, match="displacement threshold"):
        engine.build_standard_relocation_spec("T_NEG", "cup", initial_box, target_box, distance_threshold_px=-15.0)

    # Nonfinite engine parameters
    with pytest.raises(ValueError, match="blur_threshold"):
        EvidenceCalculusEngine(blur_threshold=float("nan"))

    with pytest.raises(ValueError, match="max_physical_speed"):
        EvidenceCalculusEngine(max_physical_speed_px_per_frame=float("inf"))

    with pytest.raises(ValueError, match="boundary_margin"):
        EvidenceCalculusEngine(boundary_margin_px=-1)

    with pytest.raises(ValueError, match="occlusion_area_drop"):
        EvidenceCalculusEngine(occlusion_area_drop_threshold=1.5)

    with pytest.raises(ValueError, match="occlusion_area_drop"):
        EvidenceCalculusEngine(occlusion_area_drop_threshold=float("nan"))


def test_nonfinite_evaluate_task_inputs():
    engine = EvidenceCalculusEngine()
    box_a = BoundingBox(x_min=100, y_min=100, x_max=200, y_max=200)
    box_b = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)
    spec = engine.build_standard_relocation_spec("T_INPUTS", "cube", box_a, box_b)

    # NaN laplacian variance
    with pytest.raises(ValueError, match="laplacian_variance"):
        engine.evaluate_task(spec, box_a, box_b, canvas_width=640, canvas_height=480, laplacian_variance=float("nan"))

    # Inf area drop ratio
    with pytest.raises(ValueError, match="area_drop_ratio"):
        engine.evaluate_task(spec, box_a, box_b, canvas_width=640, canvas_height=480, area_drop_ratio=float("inf"))

    # Zero canvas width
    with pytest.raises(ValueError, match="canvas_width"):
        engine.evaluate_task(spec, box_a, box_b, canvas_width=0, canvas_height=480)


def test_rest_evidence_distinction_and_categories():
    engine = EvidenceCalculusEngine()
    box_a = BoundingBox(x_min=50, y_min=50, x_max=150, y_max=150)
    box_b = BoundingBox(x_min=300, y_min=100, x_max=400, y_max=200)
    final_placed = BoundingBox(x_min=310, y_min=105, x_max=390, y_max=195)

    # 1. Default mode (spatial relocation only): sharp final frame confirms position -> SUCCESS
    spec_spatial = engine.build_standard_relocation_spec("T_SPATIAL", "cup", box_a, box_b, require_rest_proof=False)
    v1, debt1, spec1 = engine.evaluate_task(spec_spatial, box_a, final_placed, canvas_width=640, canvas_height=480)
    assert v1 == ActionVerdict.SUCCESS
    assert spec1.structured_finding == "POSITION_VERIFIED"
    assert spec1.evidence_categories["object_visible"] is True
    assert spec1.evidence_categories["final_position_observed"] is True
    assert spec1.evidence_categories["rest_condition"] == "UNOBSERVABLE"

    # 2. Strict physical action mode (require_rest_proof=True) WITHOUT temporal evidence -> UNCERTAIN
    spec_strict = engine.build_standard_relocation_spec("T_STRICT", "cup", box_a, box_b, require_rest_proof=True)
    v2, debt2, spec2 = engine.evaluate_task(spec_strict, box_a, final_placed, canvas_width=640, canvas_height=480)
    assert v2 == ActionVerdict.UNCERTAIN
    assert debt2.total_debt > 0.0
    assert spec2.structured_finding == "POSITION_VERIFIED"
    p_final = [p for p in spec2.predicates if p.predicate_type == PredicateType.FINAL_STATE_OBSERVED][0]
    assert p_final.state == EvidenceState.UNOBSERVABLE
    assert "POSITION_VERIFIED" in p_final.rationale

    # 3. Strict physical action mode WITH verified temporal rest evidence -> SUCCESS (ACTION_COMPLETED)
    v3, debt3, spec3 = engine.evaluate_task(
        spec_strict,
        box_a,
        final_placed,
        canvas_width=640,
        canvas_height=480,
        motion_cessation_observed=True,
        release_observed=True,
    )
    assert v3 == ActionVerdict.SUCCESS
    assert debt3.total_debt == 0.0
    assert spec3.structured_finding == "ACTION_COMPLETED"
    assert spec3.evidence_categories["motion_cessation_observed"] is True
    assert spec3.evidence_categories["release_observed"] is True
    assert spec3.evidence_categories["rest_condition"] == "SUPPORTED"
