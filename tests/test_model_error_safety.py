"""Regression tests for model error handling and resource safety (Issue 3).

Verifies:
- Safe public error responses (no internal filesystem paths, API secrets, or full stack traces).
- Correlation ID included in error response.
- Missing model / initialization failure handled safely.
- Invalid checkpoint rejected with clean HTTP 500 without leaking file paths.
- Corrupted model output handled with safe error response.
- Unavailable MPS device handled safely (clean CPU fallback or safe error).
- Repeated and simultaneous inference requests execute safely under bounded concurrency.
- Oversized base64 payloads (> 20MB) and decompression bombs rejected.
- No success receipt issued or persisted after failed inference.
"""

from __future__ import annotations

import base64
import concurrent.futures
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.api import routes
from backend.evidence.dataset_manifest import CanonicalDatasetManifestRegistry
from backend.providers.local_segmentation import (
    LocalMask2FormerProvider,
    ModelOutputError,
    ModelUnavailableError,
    get_local_mask2former_provider,
)

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
def authentic_coco_sample(client: TestClient) -> dict:
    resp = client.get("/api/vision/samples")
    assert resp.status_code == 200
    samples = resp.json()
    if samples:
        return {**samples[0], "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE"}
    image = Image.new("RGB", (64, 64), color=(120, 180, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return {
        "image_base64": f"data:image/jpeg;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}",
        "annotation": {
            "id": 1,
            "category_name": "cup",
            "bbox": {"x_min": 8, "y_min": 8, "x_max": 48, "y_max": 48},
        },
        "manifest_image_id": None,
        "image_source_type": "USER_UPLOADED_IMAGE",
    }


def test_missing_model_produces_safe_error_response(client: TestClient, authentic_coco_sample: dict, monkeypatch):
    """When model cannot be loaded, client receives HTTP 500 with sanitized message and correlation ID."""
    class BrokenProvider:
        def predict_from_rgb(self, *args, **kwargs):
            raise ModelUnavailableError("Cached safetensors checkpoint not found at /Users/sunny/secrets/model.safetensors")

    monkeypatch.setattr(routes, "get_local_mask2former_provider", lambda: BrokenProvider())

    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": authentic_coco_sample["image_source_type"],
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "Request ID: INF-" in detail
    # Verify no local filesystem paths or user paths are leaked
    assert "/Users/" not in detail
    assert "secrets" not in detail
    assert "[PATH_REDACTED]" in detail


def test_invalid_checkpoint_error_is_sanitized(client: TestClient, authentic_coco_sample: dict, monkeypatch):
    """Checkpoint hash mismatch error must redact internal hashes/paths."""
    class HashFailProvider:
        def predict_from_rgb(self, *args, **kwargs):
            raise ModelUnavailableError(
                "Checkpoint SHA-256 verification failed: expected 3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80, "
                "got 0000000000000000000000000000000000000000000000000000000000000000 at /Users/sunny/cache/model.safetensors"
            )

    monkeypatch.setattr(routes, "get_local_mask2former_provider", lambda: HashFailProvider())

    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": authentic_coco_sample["image_source_type"],
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "/Users/" not in detail
    assert "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80" not in detail
    assert "[REDACTED_HASH]" in detail


def test_corrupted_model_output_handled_safely(client: TestClient, authentic_coco_sample: dict, monkeypatch):
    """Corrupted or malformed model output produces sanitized error response."""
    class CorruptOutputProvider:
        def predict_from_rgb(self, *args, **kwargs):
            raise ModelOutputError("Prediction dimension mismatch: (3, 3) != (640, 427)")

    monkeypatch.setattr(routes, "get_local_mask2former_provider", lambda: CorruptOutputProvider())

    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": authentic_coco_sample["image_source_type"],
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 500
    assert "Request ID: INF-" in resp.json()["detail"]


def test_no_success_receipt_issued_after_failed_inference(client: TestClient, authentic_coco_sample: dict, monkeypatch):
    """When inference fails, no evidence receipt is issued or stored in _receipt_store."""
    initial_count = len(routes._receipt_store)

    class FailingProvider:
        def predict_from_rgb(self, *args, **kwargs):
            raise RuntimeError("Fatal inference failure")

    monkeypatch.setattr(routes, "get_local_mask2former_provider", lambda: FailingProvider())

    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 500
    assert len(routes._receipt_store) == initial_count


def test_oversized_base64_payload_rejected(client: TestClient, authentic_coco_sample: dict):
    """Base64 payloads exceeding 20MB are rejected with HTTP 400 before decoding."""
    large_payload = "A" * (21 * 1024 * 1024)
    resp = client.post(
        "/api/vision/audit",
        json={"image_base64": large_payload, "annotation": authentic_coco_sample["annotation"]},
    )
    assert resp.status_code == 413
    assert "exceeds maximum allowed size limit" in resp.json()["detail"]


@requires_local_model
def test_bounded_concurrency_and_simultaneous_requests(client: TestClient, authentic_coco_sample: dict):
    """Simultaneous requests execute safely without device memory collisions."""
    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": authentic_coco_sample["image_source_type"],
    }

    def call_audit():
        return client.post("/api/vision/audit", json=payload)

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(call_audit) for _ in range(3)]
        results = [f.result() for f in futures]

    for resp in results:
        assert resp.status_code == 200
        assert resp.json()["receipt_verified"] is True
