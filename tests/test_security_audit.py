"""Security, privacy, and integrity verification tests (Phase 3 Final)."""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app
from backend.evidence.receipts import generate_receipt, verify_receipt_integrity
from backend.evidence.schemas import UncertaintyReason, UncertaintySignal
from backend.providers.meta_adapter import redact_token, sanitize_error_detail

client = TestClient(app)


def test_api_key_redaction_and_sanitization():
    # Never leaks full token
    raw_key = "meta_sec_live_9876543210abcdef"
    redacted = redact_token(raw_key)
    assert "9876543210" not in redacted
    assert redacted.endswith("cdef]")
    assert redact_token("") == "[NOT_SET]"
    assert redact_token(None) == "[NOT_SET]"

    # Error message sanitization scrubs Bearer headers and keys
    err_text = "HTTP 401: Invalid Bearer meta_sec_live_9876543210abcdef credential"
    sanitized = sanitize_error_detail(err_text)
    assert "meta_sec_live" not in sanitized
    assert "[REDACTED_SECRET]" in sanitized


def test_tamper_evident_receipt_verification():
    sig = UncertaintySignal(
        reason=UncertaintyReason.BLUR,
        severity="HIGH",
        metric_name="laplacian_var",
        metric_value=42.0,
        threshold=100.0,
        description="Blur test",
    )
    receipt = generate_receipt(
        report_id="TEST-RECEIPT-01",
        engine_name="LabelCourt",
        software_version="0.2.2",
        inputs_hash="abc123hash",
        measurements={"iou": 0.92},
        signals=[sig],
        verdict="ACCEPT",
    )

    # Valid receipt verifies
    assert verify_receipt_integrity(receipt) is True

    # Tampered verdict fails cryptographic verification
    tampered_receipt = receipt.model_copy(update={"verdict": "SUCCESS"})
    assert verify_receipt_integrity(tampered_receipt) is False

    # Tampered measurements fail verification
    tampered_measurements = receipt.model_copy(update={"measurements": {"iou": 0.99}})
    assert verify_receipt_integrity(tampered_measurements) is False


def test_path_traversal_rejection_in_receipt_lookup():
    # Attempt directory traversal in report ID
    resp = client.get("/api/receipts/../../etc/passwd")
    assert resp.status_code in (404, 400, 422)

    resp2 = client.get("/api/receipts/%2e%2e%2f%2e%2e%2fetc%2fpasswd")
    assert resp2.status_code in (404, 400, 422)


def test_invalid_base64_rejection():
    # Corrupted base64
    resp = client.post("/api/image/upload", json={"image_base64": "not_a_valid_base64_string!!!"})
    assert resp.status_code in (400, 422)
