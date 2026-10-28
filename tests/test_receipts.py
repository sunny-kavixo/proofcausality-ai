"""Comprehensive regression tests for Evidence Receipt SHA-256 integrity verification."""

import copy
import hashlib
import json
from backend.evidence.receipts import canonical_json_dump, generate_receipt, verify_receipt_integrity
from backend.evidence.schemas import EvidenceReceipt, UncertaintyReason, UncertaintySignal


def _build_sample_receipt() -> EvidenceReceipt:
    signal = UncertaintySignal(
        reason=UncertaintyReason.OUT_OF_FRAME,
        severity="HIGH",
        metric_name="border_touching_pixels",
        metric_value=45.0,
        threshold=1.0,
        description="Touches border",
    )
    return generate_receipt(
        report_id="TEST-001",
        engine_name="ActionProof",
        software_version="0.2.5",
        inputs_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        measurements={"displacement": 120.5, "max_velocity": 45.2},
        signals=[signal],
        verdict="UNCERTAIN",
        recapture_directive="Move camera back 20cm",
        provenance={
            "category": "VERIFIED_IMAGE_FOLLOWUP",
            "source_id": "sha256_mock_source_12345",
            "verification_method": "DECODED_RGB_PIXELS",
        },
    )


def test_original_receipt_verifies():
    """Verify that a freshly generated receipt passes integrity verification."""
    receipt = _build_sample_receipt()
    assert receipt.receipt_sha256 is not None
    assert len(receipt.receipt_sha256) == 64
    assert verify_receipt_integrity(receipt) is True


def test_serialized_deserialized_receipt_verifies():
    """Verify that a receipt survives JSON serialization and deserialization without breaking verification."""
    receipt = _build_sample_receipt()
    raw_json = receipt.model_dump_json()
    reconstructed = EvidenceReceipt.model_validate_json(raw_json)
    assert verify_receipt_integrity(reconstructed) is True


def test_tamper_measurements_invalidates_verification():
    """Verify that mutating recorded measurements invalidates receipt verification."""
    receipt = _build_sample_receipt()
    receipt.measurements["displacement"] = 120.6
    assert verify_receipt_integrity(receipt) is False


def test_tamper_provenance_invalidates_verification():
    """Verify that modifying evidence provenance invalidates receipt verification."""
    receipt = _build_sample_receipt()
    assert receipt.provenance is not None
    receipt.provenance["source_id"] = "different_source_sha"
    assert verify_receipt_integrity(receipt) is False


def test_tamper_verdict_invalidates_verification():
    """Verify that modifying the decision verdict invalidates receipt verification."""
    receipt = _build_sample_receipt()
    receipt.verdict = "SUCCESS"
    assert verify_receipt_integrity(receipt) is False


def test_tamper_source_inputs_hash_invalidates_verification():
    """Verify that modifying inputs_hash invalidates receipt verification."""
    receipt = _build_sample_receipt()
    receipt.inputs_hash = "0000000000000000000000000000000000000000000000000000000000000000"
    assert verify_receipt_integrity(receipt) is False


def test_tamper_timestamp_invalidates_verification():
    """Verify that modifying timestamp invalidates receipt verification."""
    receipt = _build_sample_receipt()
    receipt.timestamp = "2026-01-01T00:00:00Z"
    assert verify_receipt_integrity(receipt) is False


def test_tamper_software_version_invalidates_verification():
    """Verify that modifying software_version invalidates receipt verification."""
    receipt = _build_sample_receipt()
    receipt.software_version = "0.9.9"
    assert verify_receipt_integrity(receipt) is False


def test_tamper_uncertainty_signals_invalidates_verification():
    """Verify that modifying uncertainty signals invalidates receipt verification."""
    receipt = _build_sample_receipt()
    receipt.uncertainty_signals = []
    assert verify_receipt_integrity(receipt) is False


def test_tamper_recapture_directive_invalidates_verification():
    """Verify that modifying recapture_directive invalidates receipt verification."""
    receipt = _build_sample_receipt()
    receipt.recapture_directive = "Different directive"
    assert verify_receipt_integrity(receipt) is False


def test_receipt_is_sha256_digest_not_keyed_hmac():
    """Verify that receipt checksum is standard reproducible SHA-256 over canonical JSON, NOT keyed HMAC."""
    receipt = _build_sample_receipt()

    # Manually compute SHA-256 over canonical JSON without any key or HMAC
    payload_for_hashing = {
        "report_id": receipt.report_id,
        "timestamp": receipt.timestamp,
        "software_version": receipt.software_version,
        "engine_name": receipt.engine_name,
        "inputs_hash": receipt.inputs_hash,
        "measurements": receipt.measurements,
        "uncertainty_signals": receipt.uncertainty_signals,
        "verdict": receipt.verdict,
        "recapture_directive": receipt.recapture_directive,
        "provenance": receipt.provenance,
    }
    canonical_str = canonical_json_dump(payload_for_hashing)
    expected_sha256 = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

    # The receipt hash must match the raw standard SHA-256 digest identically
    assert receipt.receipt_sha256 == expected_sha256
