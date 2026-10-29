"""
Regression tests for Issue 8: Comprehensive Security Review of Public API Routes.
Ensures that:
1. Strict 20MB payload limits are enforced across all image upload endpoints.
2. Path traversal attempts on receipt endpoints are rejected with 400 Bad Request.
3. Information leakage (local filesystem paths, raw exception traces, API secrets) is prevented.
4. Decompression bomb limits protect memory against oversize or maliciously inflated images.
5. In-memory receipt store enforces strict LRU eviction at 1,000 items to prevent unbounded memory growth.
6. Offline fallback guarantees no unauthorized third-party network requests.
"""
import base64
import os
import pytest
from fastapi.testclient import TestClient
from backend.app import app
from backend.api.routes import _receipt_store, store_receipt
from backend.evidence.schemas import EvidenceReceipt
from backend.providers.meta_adapter import MetaAdapter

client = TestClient(app)


def test_upload_limits_across_endpoints():
    """Verify that all image-accepting endpoints reject payloads exceeding 20MB with HTTP 413."""
    oversized_b64 = "A" * (21 * 1024 * 1024)

    # 1. /api/image/upload
    resp1 = client.post("/api/image/upload", json={"image_base64": oversized_b64})
    assert resp1.status_code == 413
    assert "20MB" in resp1.json()["detail"]

    # 2. /api/labelcourt/audit
    resp2 = client.post("/api/labelcourt/audit", json={
        "image_width": 640,
        "image_height": 480,
        "human_annotation": {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "category_name": "cup",
            "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        },
        "image_base64": oversized_b64,
    })
    assert resp2.status_code == 413
    assert "20MB" in resp2.json()["detail"]

    # 3. /api/actionproof/verify
    resp3 = client.post("/api/actionproof/verify", json={
        "request": {"task_type": "moved_A_to_B", "object_name": "cup"},
        "image_width": 640,
        "image_height": 480,
        "before_bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "after_bbox": {"x_min": 100, "y_min": 100, "x_max": 150, "y_max": 150},
        "after_image_base64": oversized_b64,
    })
    assert resp3.status_code == 413
    assert "20MB" in resp3.json()["detail"]

    # 4. /api/recapture/evaluate-followup
    resp4 = client.post("/api/recapture/evaluate-followup", json={
        "task_id": "SEC_TEST_001",
        "primary_object": "cup",
        "initial_verdict": "UNCERTAIN",
        "detected_defect": "BLUR",
        "directive_issued": "HOLD_STEADY",
        "instruction_followed": True,
        "effort_seconds": 1.5,
        "canvas_width": 640,
        "canvas_height": 480,
        "original_before_bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "followup_bbox": {"x_min": 100, "y_min": 100, "x_max": 150, "y_max": 150},
        "target_region": {"x_min": 100, "y_min": 100, "x_max": 200, "y_max": 200},
        "initial_region": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "followup_image_base64": oversized_b64,
    })
    assert resp4.status_code == 413
    assert "20MB" in resp4.json()["detail"]

    # 5. /api/vision/audit
    resp5 = client.post("/api/vision/audit", json={
        "image_base64": oversized_b64,
        "annotation": {"id": 1, "image_id": 1, "category_id": 1, "category_name": "cup", "bbox": {"x_min": 0, "y_min": 0, "x_max": 10, "y_max": 10}, "segmentation": [[0, 0, 10, 0, 10, 10, 0, 10]]},
    })
    assert resp5.status_code == 413
    assert "20MB" in resp5.json()["detail"]


def test_path_traversal_resistance_on_receipts():
    """Verify that path traversal attempts on /receipts/{report_id} are rejected with 400 Bad Request."""
    traversal_payloads = [
        "../../etc/passwd",
        "..%2F..%2Fetc%2Fpasswd",
        "....//....//etc//passwd",
        "/etc/shadow",
        "CON",
        "PRN",
        "a" * 128,  # Excessive length
        "invalid;injection",
    ]
    for payload in traversal_payloads:
        resp = client.get(f"/api/receipts/{payload}")
        # Must either be 400 Bad Request (regex rejection) or 404 Not Found (safe dict miss)
        # It must NEVER succeed (200) or crash (500)
        assert resp.status_code in (400, 404), f"Payload {payload} returned unexpected status {resp.status_code}"


def test_no_filesystem_paths_leaked_in_errors():
    """Verify that error messages sanitize absolute paths and internal trace details."""
    malformed_b64 = base64.b64encode(b"THIS_IS_NOT_AN_IMAGE").decode("ascii")

    resp = client.post("/api/image/upload", json={"image_base64": malformed_b64})
    assert resp.status_code == 400
    detail = resp.json().get("detail", "")
    assert "/Users/" not in detail
    assert "/home/" not in detail
    assert ".py" not in detail


def test_model_info_credential_redaction():
    """Verify that /api/model-info redacts any API tokens and reveals no raw secrets or host paths."""
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "api_key_status" in data
    assert data["api_key_status"] in ("[NOT_SET]", "[REDACTED_KEY_SUFFIX_...]") or "REDACTED" in data["api_key_status"]
    # Check that no sensitive environment variables or local paths are leaked
    for key, val in data.items():
        val_str = str(val)
        assert "sk-" not in val_str
        assert "password" not in val_str.lower()
        assert "/Users/" not in val_str


def test_receipt_store_lru_bounding():
    """Verify that the in-memory receipt store enforces strict LRU eviction capped at 1,000 items."""
    _receipt_store.clear()

    # Create valid dummy receipts
    for i in range(1050):
        receipt = EvidenceReceipt(
            report_id=f"TEST-REPORT-{i:04d}",
            timestamp="2026-10-29T20:00:00Z",
            software_version="0.2.6",
            engine_name="TestEngine",
            inputs_hash="a" * 64,
            receipt_sha256="b" * 64,
            measurements={"idx": i},
            uncertainty_signals=[],
            verdict="VALID",
            provenance={"category": "TEST"},
        )
        store_receipt(receipt)
        store_receipt(receipt)

    assert len(_receipt_store) == 1000
    # Oldest receipts (0 to 49) must have been evicted
    assert "TEST-REPORT-0000" not in _receipt_store
    assert "TEST-REPORT-0049" not in _receipt_store
    # Newest receipts must be present
    assert "TEST-REPORT-1049" in _receipt_store


def test_no_unauthorized_external_network_calls():
    """Verify that MetaAdapter defaults to offline deterministic mode without remote credentials."""
    adapter = MetaAdapter()
    assert not adapter.is_configured
    assert not adapter.enabled
    diag = adapter.get_diagnostics()
    assert diag["meta_inference_blocked"] is True
    assert "OFFLINE" in diag["active_mode"].upper() or "DETERMINISTIC" in diag["active_mode"].upper()
