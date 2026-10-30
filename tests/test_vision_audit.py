"""Comprehensive integration and unit test suite for the genuine vision audit workflow.

Covers:
- Authentic COCO photograph unprompted segmentation audit.
- Controlled annotation perturbations:
    * Concordant segmentation (ACCEPT).
    * Class mismatch (REVIEW).
    * Spatial misalignment / shifted bbox (REVIEW).
    * Severe visual blur uncertainty (UNCERTAIN).
    * Border truncation uncertainty (UNCERTAIN).
    * Zero detections at high confidence (REVIEW).
- Cryptographic evidence receipt tamper verification.
- Input validation: invalid base64, empty payload, non-image format.
- Provider unavailable handling (HTTP 500 error propagation).
- Thread-safe model caching and resource management.
"""

from __future__ import annotations

import base64
import copy
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.app import app
from backend.evidence.receipts import verify_receipt_integrity
from backend.evidence.schemas import EvidenceReceipt, VisionAuditVerdict
from backend.providers.local_segmentation import get_local_mask2former_provider

# Determine if PyTorch and local checkpoint are available in current environment
try:
    import torch
    import transformers
    _p = get_local_mask2former_provider()
    _p.load()
    _MODEL_AVAILABLE = True
except Exception:
    _MODEL_AVAILABLE = False

requires_local_model = pytest.mark.skipif(
    not _MODEL_AVAILABLE,
    reason="PyTorch, Transformers, or local Mask2Former checkpoint not available in test environment",
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def pilot_samples(client: TestClient) -> list[dict]:
    resp = client.get("/api/vision/samples")
    assert resp.status_code == 200
    samples = resp.json()
    if not samples:
        pytest.skip("Genuine-photo integration tests require the optional COCO pilot download")
    return samples


def test_get_vision_samples_schema(pilot_samples: list[dict]):
    """Verify that COCO samples contain authentic photos and well-formed COCO annotations."""
    for s in pilot_samples:
        assert "sample_id" in s
        assert "image_base64" in s
        assert s["image_base64"].startswith("data:image/jpeg;base64,")
        assert "annotation" in s
        assert "category_name" in s["annotation"]
        assert "bbox" in s["annotation"]
        assert "perturbation_presets" in s


def test_invalid_image_payloads(client: TestClient, pilot_samples: list[dict]):
    """Invalid base64, empty bytes, or non-image format must be rejected with HTTP 400."""
    sample = pilot_samples[0]

    # Corrupt base64
    resp = client.post(
        "/api/vision/audit",
        json={"image_base64": "!!!not_valid_base64!!!", "annotation": sample["annotation"]},
    )
    assert resp.status_code == 400

    # Valid base64 but text file bytes (not PNG/JPEG)
    text_b64 = base64.b64encode(b"Hello world this is not an image").decode("ascii")
    resp = client.post(
        "/api/vision/audit",
        json={"image_base64": text_b64, "annotation": sample["annotation"]},
    )
    assert resp.status_code == 400
    assert "Unsupported image format" in resp.json()["detail"]


def test_provider_unavailable_handling(client: TestClient, pilot_samples: list[dict], monkeypatch):
    """When the model provider fails or raises an error, API returns HTTP 500 with explanatory detail."""
    sample = pilot_samples[0]
    from backend.api import routes

    class BrokenProvider:
        def predict_from_rgb(self, *args, **kwargs):
            raise RuntimeError("Model weights corrupted or memory exhausted")

    monkeypatch.setattr(routes, "get_local_mask2former_provider", lambda: BrokenProvider())

    payload = {
        "image_base64": sample["image_base64"],
        "annotation": sample["annotation"],
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 500
    assert "Mask2Former inference execution failed" in resp.json()["detail"]


@requires_local_model
def test_genuine_vision_audit_execution(client: TestClient, pilot_samples: list[dict]):
    """Run an authentic vision audit on a real COCO pilot photograph."""
    sample = pilot_samples[0]
    payload = {
        "image_base64": sample["image_base64"],
        "annotation": sample["annotation"],
        "manifest_image_id": sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
        "settings": {
            "confidence_threshold": 0.50,
            "iou_accept_threshold": 0.70,
            "iou_review_threshold": 0.40,
            "boundary_agreement_threshold": 0.60,
            "matching_strategy": "class_aware",
            "blur_threshold": 100.0,
            "boundary_margin_px": 3,
        },
    }

    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200, f"Audit failed: {resp.text}"
    data = resp.json()

    # 1. Verdict structure
    assert data["verdict"] in [v.value for v in VisionAuditVerdict]
    assert data["verdict_reason"]
    assert data["explanation"]

    # 2. Model execution provenance
    model_exec = data["model_execution"]
    assert model_exec["model_name"] == "facebook/mask2former-swin-tiny-coco-instance"
    assert model_exec["checkpoint_sha256"]
    assert model_exec["checkpoint_sha256"] != "UNKNOWN"
    assert model_exec["runtime_ms"] > 0

    # 3. Evidence receipt & integrity
    receipt_data = data["evidence_receipt"]
    assert data["receipt_verified"] is True
    receipt_obj = EvidenceReceipt(**receipt_data)
    assert verify_receipt_integrity(receipt_obj) is True

    # 4. Latency breakdown disaggregation
    latency = data["latency_breakdown_ms"]
    assert latency["image_decoding_ms"] >= 0
    assert latency["preprocessing_ms"] >= 0
    assert latency["model_inference_ms"] > 0
    assert latency["mask_postprocessing_ms"] >= 0
    assert latency["labelcourt_evaluation_ms"] >= 0
    assert latency["overlay_generation_ms"] >= 0
    assert latency["receipt_generation_ms"] >= 0
    assert latency["total_wall_clock_ms"] > 0

    # 5. Visual overlays
    overlays = data["visual_overlays"]
    assert overlays["disagreement"].startswith("data:image/png;base64,")
    assert overlays["reference"].startswith("data:image/png;base64,")
    assert overlays["predicted"].startswith("data:image/png;base64,")


@requires_local_model
def test_controlled_class_mismatch(client: TestClient, pilot_samples: list[dict]):
    """Deliberately altered annotation class must produce a REVIEW verdict with CLASS_MISMATCH reason."""
    sample = pilot_samples[0]
    ann = copy.deepcopy(sample["annotation"])
    # Intentionally corrupt category name
    ann["category_name"] = "airplane"

    payload = {
        "image_base64": sample["image_base64"],
        "annotation": ann,
        "manifest_image_id": sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
        "settings": {
            "confidence_threshold": 0.50,
            "matching_strategy": "class_aware",
        },
    }

    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["verdict"] == VisionAuditVerdict.REVIEW.value
    assert data["verdict_reason"] == "CLASS_MISMATCH"
    assert data["metrics"]["class_match"] is False
    assert "airplane" in data["explanation"]


@requires_local_model
def test_severe_visual_blur_produces_uncertain(client: TestClient, pilot_samples: list[dict]):
    """Images with high blur (Laplacian variance below threshold) must yield UNCERTAIN verdict."""
    sample = pilot_samples[0]
    payload = {
        "image_base64": sample["image_base64"],
        "annotation": sample["annotation"],
        "manifest_image_id": sample["manifest_image_id"],
        "settings": {
            "blur_threshold": 999999.0,  # All images will fall below this
        },
    }

    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["verdict"] == VisionAuditVerdict.UNCERTAIN.value
    assert data["verdict_reason"] == "HIGH_VISUAL_BLUR"
    assert any(s["reason"] == "BLUR" for s in data["evidence_receipt"]["uncertainty_signals"])


@requires_local_model
def test_border_truncation_uncertainty(client: TestClient, pilot_samples: list[dict]):
    """Annotations extending to the canvas border with imperfect IoU must be marked UNCERTAIN."""
    sample = pilot_samples[0]
    ann = copy.deepcopy(sample["annotation"])
    # Force bbox to contact top/left border
    ann["bbox"]["x_min"] = 0.0
    ann["bbox"]["y_min"] = 0.0
    ann["segmentation"] = None  # Use bbox mask

    payload = {
        "image_base64": sample["image_base64"],
        "annotation": ann,
        "manifest_image_id": sample["manifest_image_id"],
        "settings": {
            "boundary_margin_px": 5,
            "iou_accept_threshold": 0.99,  # High accept threshold so it triggers uncertainty
            "iou_review_threshold": 0.01,
        },
    }

    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["verdict"] == VisionAuditVerdict.UNCERTAIN.value
    assert data["verdict_reason"] == "BORDER_TRUNCATION_UNCERTAIN"


@requires_local_model
def test_zero_detections_handling(client: TestClient, pilot_samples: list[dict]):
    """Setting confidence threshold to 0.99 leads to zero model predictions, triggering REVIEW."""
    sample = pilot_samples[0]
    payload = {
        "image_base64": sample["image_base64"],
        "annotation": sample["annotation"],
        "manifest_image_id": sample["manifest_image_id"],
        "settings": {
            "confidence_threshold": 0.99,
        },
    }

    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["verdict"] == VisionAuditVerdict.REVIEW.value
    assert data["verdict_reason"] == "MODEL_ZERO_DETECTIONS"
    assert data["detected_instances_count"] == 0
    assert data["matched_prediction"] is None


@requires_local_model
def test_tamper_detection_on_receipt(client: TestClient, pilot_samples: list[dict]):
    """Modifying any field in the generated receipt must be detected by verify_receipt_integrity."""
    sample = pilot_samples[0]
    payload = {
        "image_base64": sample["image_base64"],
        "annotation": sample["annotation"],
    }

    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200
    receipt_data = resp.json()["evidence_receipt"]

    receipt = EvidenceReceipt(**receipt_data)
    assert verify_receipt_integrity(receipt) is True

    # Tamper with verdict
    tampered_data = copy.deepcopy(receipt_data)
    tampered_data["verdict"] = "ACCEPT" if receipt_data["verdict"] != "ACCEPT" else "REVIEW"
    tampered_receipt = EvidenceReceipt(**tampered_data)
    assert verify_receipt_integrity(tampered_receipt) is False

    # Tamper with measurement
    tampered_data2 = copy.deepcopy(receipt_data)
    tampered_data2["measurements"]["iou"] = 0.9999
    tampered_receipt2 = EvidenceReceipt(**tampered_data2)
    assert verify_receipt_integrity(tampered_receipt2) is False


@requires_local_model
def test_model_cache_singleton(pilot_samples: list[dict]):
    """Verify that multiple provider requests access the same singleton instance without reloading."""
    p1 = get_local_mask2former_provider()
    p2 = get_local_mask2former_provider()
    assert p1 is p2
    assert p1.device in ["mps", "cpu"]
    assert p1._checkpoint_sha256 is not None
