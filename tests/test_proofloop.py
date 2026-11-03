import base64
import io
import math
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app
from backend.evidence.proofloop import (
    ProofLoopObservation,
    ProofLoopState,
    ProofLoopVerdict,
    ProofLoopVerifier,
    semantic_verdict_for,
)
from backend.evidence.schemas import ActionVerdict, BoundingBox
from backend.providers.local_segmentation import InferenceExecution, InstancePrediction
from evaluation.benchmark_proofloop import run_benchmark


client = TestClient(app)


def _image_b64(seed: int, width: int = 160, height: int = 160) -> str:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[::2, ::2] = (255, seed % 255, 80)
    arr[1::2, 1::2] = (20, 120, 240)
    buffer = io.BytesIO()
    Image.fromarray(arr).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class _FakeMediaProvider:
    def __init__(self):
        self.calls = 0

    def predict_from_rgb(self, rgb_arr, threshold=0.5, raw_bytes=None):
        self.calls += 1
        x_offset = 10 if self.calls == 1 else max(100, rgb_arr.shape[1] - 60)
        y_offset = 10 if self.calls == 1 else 100
        mask = np.zeros(rgb_arr.shape[:2], dtype=bool)
        mask[y_offset:y_offset + 20, x_offset:x_offset + 20] = True
        prediction = InstancePrediction(
            instance_id=self.calls,
            label_id=39,
            label="bottle",
            score=0.91,
            mask=mask,
        )
        execution = InferenceExecution(
            provider="test-fake-server-provider",
            model_identifier="test-mask2former",
            model_revision="test-revision",
            inference_mode="UNPROMPTED_INSTANCE_SEGMENTATION",
            device="cpu",
            input_sha256="a" * 64,
            input_width=rgb_arr.shape[1],
            input_height=rgb_arr.shape[0],
            prompt=None,
            checkpoint_sha256="b" * 64,
            checkpoint_size_bytes=1,
            output_sha256=f"{self.calls}" * 64,
            latency_ms=1.0,
            execution_status="SUCCEEDED",
            runtime={"python": "test"},
        )
        return [prediction], execution


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


def test_media_api_derives_measurements_and_binds_both_image_hashes(monkeypatch):
    from backend.api import routes

    fake_provider = _FakeMediaProvider()
    monkeypatch.setattr(routes, "get_local_mask2former_provider", lambda: fake_provider)

    initial = client.post(
        "/api/proofloop/media/start",
        json={
            "task_id": "MEDIA_TEST_001",
            "primary_object": "bottle",
            "target_region": _box(90, 90, 140, 140),
            "image_base64": _image_b64(1),
            "image_source_type": "USER_UPLOADED_IMAGE",
        },
    )
    assert initial.status_code == 200, initial.text
    initial_data = initial.json()
    assert initial_data["provenance_status"] == "SERVER_DERIVED_MEDIA"
    assert initial_data["semantic_verdict"] == "INSUFFICIENT_EVIDENCE"
    assert initial_data["media_evidence"]["client_measurements_accepted"] is False
    assert initial_data["media_evidence"]["initial"]["matching_prediction"]["bbox"] == _box(10, 10, 30, 30)
    initial_hash = initial_data["media_evidence"]["initial"]["image_sha256"]
    assert initial_hash != ""
    assert initial_data["evidence_receipt"]["provenance"]["initial"]["source_image_hash"] == initial_hash

    followup = client.post(
        "/api/proofloop/media/reverify",
        json={
            "session_id": initial_data["session_id"],
            "image_base64": _image_b64(2, width=320, height=160),
            "image_source_type": "USER_UPLOADED_IMAGE",
        },
    )
    assert followup.status_code == 200, followup.text
    followup_data = followup.json()
    assert followup_data["semantic_verdict"] == "SUPPORTED"
    assert followup_data["state_trace"][-3:] == ["NEW_EVIDENCE", "REVERIFIED", "FINAL_VERDICT"]
    assert followup_data["media_evidence"]["client_measurements_accepted"] is False
    assert followup_data["media_evidence"]["followup"]["matching_prediction"]["bbox"] == _box(260, 100, 280, 120)
    assert followup_data["media_evidence"]["followup"]["coordinate_transform"]["normalized_server_bbox"] == _box(130, 100, 140, 120)
    assert followup_data["media_evidence"]["input_image_hashes_bound_to_receipt"] is True
    assert followup_data["evidence_receipt"]["provenance"]["followup"]["source_image_hash"] == followup_data["media_evidence"]["followup"]["image_sha256"]
    assert initial_hash != followup_data["media_evidence"]["followup"]["image_sha256"]
    assert fake_provider.calls == 2
    receipt_response = client.get(f"/api/receipts/{followup_data['evidence_receipt_id']}")
    assert receipt_response.status_code == 200
    assert receipt_response.json()["cryptographic_integrity_verified"] is True


def test_media_api_rejects_client_measurement_fields():
    response = client.post(
        "/api/proofloop/media/start",
        json={
            "task_id": "MEDIA_EXTRA_FIELDS",
            "primary_object": "bottle",
            "target_region": _box(90, 90, 140, 140),
            "image_base64": _image_b64(1),
            "image_source_type": "USER_UPLOADED_IMAGE",
            "before_bbox": _box(1, 1, 2, 2),
            "laplacian_variance": 999999.0,
            "model_confidence": 1.0,
        },
    )
    assert response.status_code == 422
    assert "extra_forbidden" in response.text


def test_media_api_rejects_manifest_hash_mismatch(monkeypatch):
    from backend.api import routes

    monkeypatch.setattr(routes, "get_local_mask2former_provider", lambda: _FakeMediaProvider())
    response = client.post(
        "/api/proofloop/media/start",
        json={
            "task_id": "MEDIA_BAD_MANIFEST",
            "primary_object": "bottle",
            "target_region": _box(0, 0, 160, 160),
            "image_base64": _image_b64(1),
            "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
            "manifest_image_id": 397133,
        },
    )
    assert response.status_code == 422
    assert "provenance" in response.json()["detail"].lower()


def test_closed_loop_benchmark_covers_resolution_and_abstention():
    report = run_benchmark()
    by_id = {case["case_id"]: case for case in report["results"]}

    assert report["case_count"] == 10
    assert report["initial_uncertain_cases"] == 7
    assert report["resolved_uncertainty_cases"] == 6
    assert by_id["class_mismatch"]["initial"]["semantic_verdict"] == "CONTRADICTED"
    assert by_id["unresolved_after_recapture"]["followup"]["resolved"] is False
    assert by_id["successfully_resolved_uncertainty"]["followup"]["resolved"] is True
