from fastapi.testclient import TestClient
import math
import pytest

from backend.app import app
from backend.evidence.proofloop import (
    ProofLoopObservation,
    ProofLoopState,
    ProofLoopVerdict,
    ProofLoopVerifier,
    semantic_verdict_for,
)
from backend.evidence.schemas import ActionVerdict, BoundingBox
from evaluation.benchmark_proofloop import run_benchmark


client = TestClient(app)


def _box(x_min: float, y_min: float, x_max: float, y_max: float) -> dict[str, float]:
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def _observation(**overrides) -> ProofLoopObservation:
    values = {
        "task_id": "PROOFLOOP_TEST",
        "primary_object": "cup",
        "before_bbox": _box(50, 50, 100, 100),
        "after_bbox": _box(310, 110, 390, 190),
        "target_region": _box(300, 100, 400, 200),
        "initial_region": _box(20, 20, 120, 120),
        "canvas_width": 640,
        "canvas_height": 480,
        "laplacian_variance": 250.0,
        "area_drop_ratio": 0.0,
        "has_final_state": True,
    }
    values.update(overrides)
    return ProofLoopObservation(**values)


def test_semantic_mapping_preserves_legacy_contract():
    assert semantic_verdict_for(ActionVerdict.SUCCESS) == ProofLoopVerdict.SUPPORTED
    assert semantic_verdict_for(ActionVerdict.FAILED) == ProofLoopVerdict.CONTRADICTED
    assert semantic_verdict_for(ActionVerdict.UNCERTAIN) == ProofLoopVerdict.INSUFFICIENT_EVIDENCE


def test_observation_rejects_nonfinite_or_degenerate_geometry():
    with pytest.raises(ValueError):
        _observation(laplacian_variance=math.nan)
    with pytest.raises(ValueError, match="positive width"):
        _observation(target_region=_box(100, 100, 100, 200))


def test_initial_uncertainty_requests_specific_observation():
    result = ProofLoopVerifier().evaluate(
        _observation(after_bbox=None, has_final_state=False, area_drop_ratio=0.55)
    )

    assert result.semantic_verdict == ProofLoopVerdict.INSUFFICIENT_EVIDENCE
    assert result.state == ProofLoopState.OBSERVATION_REQUESTED
    assert result.state_trace == [
        ProofLoopState.OBSERVED,
        ProofLoopState.ANALYZED,
        ProofLoopState.UNCERTAIN,
        ProofLoopState.OBSERVATION_REQUESTED,
    ]
    assert result.observation_request is not None
    assert result.observation_request["directive"]


def test_low_confidence_abstains_without_falsifying_the_claim():
    result = ProofLoopVerifier().evaluate(_observation(model_confidence=0.2))

    assert result.legacy_action_verdict == ActionVerdict.UNCERTAIN
    assert result.semantic_verdict == ProofLoopVerdict.INSUFFICIENT_EVIDENCE
    assert result.decision_basis == "LOW_MODEL_CONFIDENCE"


def test_class_mismatch_is_explicit_contradiction():
    result = ProofLoopVerifier().evaluate(_observation(claim_class_agreement=False))

    assert result.legacy_action_verdict == ActionVerdict.FAILED
    assert result.semantic_verdict == ProofLoopVerdict.CONTRADICTED
    assert result.decision_basis == "CLAIM_CLASS_MISMATCH"
    assert result.state == ProofLoopState.FINAL_VERDICT


def test_api_reverify_resolves_uncertainty_and_receipt_is_valid():
    initial = client.post(
        "/api/proofloop/start",
        json=_observation(
            after_bbox=_box(310, 110, 390, 190),
            laplacian_variance=20.0,
        ).model_dump(mode="json"),
    )
    assert initial.status_code == 200
    initial_data = initial.json()
    assert initial_data["state"] == "OBSERVATION_REQUESTED"
    assert initial_data["semantic_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert initial_data["provenance_status"] == "METADATA_ONLY"

    followup = client.post(
        "/api/proofloop/reverify",
        json={
            "session_id": initial_data["session_id"],
            "after_bbox": _box(310, 110, 390, 190),
            "laplacian_variance": 250.0,
            "area_drop_ratio": 0.0,
            "has_final_state": True,
        },
    )
    assert followup.status_code == 200
    data = followup.json()
    assert data["semantic_verdict"] == "SUPPORTED"
    assert data["legacy_action_verdict"] == "SUCCESS"
    assert data["state"] == "FINAL_VERDICT"
    assert data["state_trace"][-3:] == ["NEW_EVIDENCE", "REVERIFIED", "FINAL_VERDICT"]
    assert data["receipt_valid"] is True
    assert data["evidence_receipt"]["provenance"]["category"] == "USER_SUPPLIED_METADATA"


def test_api_reverify_can_remain_insufficient_after_new_evidence():
    initial = client.post(
        "/api/proofloop/start",
        json=_observation(after_bbox=None, has_final_state=False).model_dump(mode="json"),
    )
    assert initial.status_code == 200
    session_id = initial.json()["session_id"]

    followup = client.post(
        "/api/proofloop/reverify",
        json={
            "session_id": session_id,
            "after_bbox": None,
            "has_final_state": False,
            "laplacian_variance": 12.0,
        },
    )
    assert followup.status_code == 200
    data = followup.json()
    assert data["semantic_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert data["state_trace"][-3:] == ["NEW_EVIDENCE", "REVERIFIED", "FINAL_VERDICT"]
    assert data["observation_request"] is not None


def test_closed_loop_benchmark_covers_resolution_and_abstention():
    report = run_benchmark()
    by_id = {case["case_id"]: case for case in report["results"]}

    assert report["case_count"] == 10
    assert report["initial_uncertain_cases"] == 7
    assert report["resolved_uncertainty_cases"] == 6
    assert by_id["class_mismatch"]["initial"]["semantic_verdict"] == "CONTRADICTED"
    assert by_id["unresolved_after_recapture"]["followup"]["resolved"] is False
    assert by_id["successfully_resolved_uncertainty"]["followup"]["resolved"] is True
