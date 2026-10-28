"""End-to-end tests for the three canonical demonstration situations from the evidence manifest."""

from __future__ import annotations

import json
import os
import numpy as np
from PIL import Image

from backend.evidence.calculus.engine import EvidenceCalculusEngine
from backend.evidence.schemas import ActionVerdict, BoundingBox
from backend.evidence.uncertainty import detect_blur_laplacian


def test_canonical_situation_a_success():
    """Situation A: Correctly completed physical action."""
    engine = EvidenceCalculusEngine()
    region_a = BoundingBox(x_min=50, y_min=50, x_max=170, y_max=170)
    region_b = BoundingBox(x_min=350, y_min=190, x_max=470, y_max=310)

    spec = engine.build_standard_relocation_spec(
        task_id="CANON_A",
        primary_object="red_mug",
        initial_region=region_a,
        target_region=region_b,
        distance_threshold_px=60.0,
    )

    before_box = BoundingBox(x_min=70, y_min=70, x_max=150, y_max=150)
    after_box = BoundingBox(x_min=370, y_min=210, x_max=450, y_max=290)

    # Decode actual disk image
    img_t1 = Image.open("tests/fixtures/images/canonical_action_success_t1.png").convert("L")
    gray_arr = np.array(img_t1)
    _, lap_var = detect_blur_laplacian(gray_arr)
    assert lap_var > 100.0, f"Expected sharp image, got lap_var={lap_var}"

    verdict, debt, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=before_box,
        after_bbox=after_box,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=lap_var,
    )

    assert verdict == ActionVerdict.SUCCESS
    assert debt.total_debt == 0.0
    assert updated_spec.structured_finding == "POSITION_VERIFIED"
    assert updated_spec.evidence_categories["object_visible"] is True
    assert updated_spec.evidence_categories["final_position_observed"] is True


def test_canonical_situation_b_failed():
    """Situation B: Incorrectly completed physical action (dropped outside region B)."""
    engine = EvidenceCalculusEngine()
    region_a = BoundingBox(x_min=50, y_min=50, x_max=170, y_max=170)
    region_b = BoundingBox(x_min=350, y_min=190, x_max=470, y_max=310)

    spec = engine.build_standard_relocation_spec(
        task_id="CANON_B",
        primary_object="red_mug",
        initial_region=region_a,
        target_region=region_b,
    )

    before_box = BoundingBox(x_min=70, y_min=70, x_max=150, y_max=150)
    after_box = BoundingBox(x_min=200, y_min=120, x_max=280, y_max=200) # Outside region B

    verdict, debt, updated_spec = engine.evaluate_task(
        spec=spec,
        before_bbox=before_box,
        after_bbox=after_box,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=250.0,
    )

    assert verdict == ActionVerdict.FAILED
    assert len(debt.contradictions) > 0
    assert updated_spec.structured_finding == "CONTRADICTION_DETECTED"


def test_canonical_situation_c_deficit_and_recapture_transition():
    """Situation C: Missing/obscured evidence -> UNCERTAIN, then ReCapture -> SUCCESS."""
    engine = EvidenceCalculusEngine()
    region_a = BoundingBox(x_min=50, y_min=50, x_max=170, y_max=170)
    region_b = BoundingBox(x_min=350, y_min=190, x_max=470, y_max=310)

    spec = engine.build_standard_relocation_spec(
        task_id="CANON_C",
        primary_object="red_mug",
        initial_region=region_a,
        target_region=region_b,
    )

    before_box = BoundingBox(x_min=70, y_min=70, x_max=150, y_max=150)
    after_box = BoundingBox(x_min=370, y_min=210, x_max=450, y_max=290)

    # 1. Initial blurred observation
    blur_img = Image.open("tests/fixtures/images/canonical_action_deficit_blur_t1.png").convert("L")
    _, blur_var = detect_blur_laplacian(np.array(blur_img))
    assert blur_var < 100.0, f"Expected blurred image, got blur_var={blur_var}"

    v_init, debt_init, spec_init = engine.evaluate_task(
        spec=spec,
        before_bbox=before_box,
        after_bbox=after_box,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=blur_var,
    )

    assert v_init == ActionVerdict.UNCERTAIN
    assert debt_init.total_debt > 0.0
    assert spec_init.structured_finding == "PERCEPTUAL_DEFICIT"

    # 2. ReCapture Follow-up observation (steady camera t2)
    recaptured_img = Image.open("tests/fixtures/images/canonical_action_recaptured_t2.png").convert("L")
    _, recaptured_var = detect_blur_laplacian(np.array(recaptured_img))
    assert recaptured_var > 100.0, f"Expected sharp recaptured image, got {recaptured_var}"

    v_recaptured, debt_recaptured, spec_recaptured = engine.evaluate_task(
        spec=spec,
        before_bbox=before_box,
        after_bbox=after_box,
        canvas_width=640,
        canvas_height=480,
        laplacian_variance=recaptured_var,
    )

    assert v_recaptured == ActionVerdict.SUCCESS
    assert debt_recaptured.total_debt == 0.0
    assert spec_recaptured.structured_finding == "POSITION_VERIFIED"
