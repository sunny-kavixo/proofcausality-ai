"""Tests for the Evidence Calculus Engine and Evidence Debt formulation (H01)."""

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
    )

    assert spec.task_id == "TASK_001"
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


def test_physical_teleportation_contradiction():
    engine = EvidenceCalculusEngine(max_physical_speed_px_per_frame=200.0)
    initial_box = BoundingBox(x_min=50, y_min=50, x_max=100, y_max=100)
    target_box = BoundingBox(x_min=500, y_min=400, x_max=550, y_max=450)

    spec = engine.build_standard_relocation_spec(
        task_id="TASK_PHYSICAL_IMPOSSIBLE",
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
    assert "Physically impossible jump" in contra_text


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
