from backend.evidence.receipts import generate_receipt, verify_receipt_integrity
from backend.evidence.schemas import UncertaintyReason, UncertaintySignal


def test_receipt_generation_and_verification():
    signal = UncertaintySignal(
        reason=UncertaintyReason.OUT_OF_FRAME,
        severity="HIGH",
        metric_name="border_touching_pixels",
        metric_value=45.0,
        threshold=1.0,
        description="Touches border",
    )
    receipt = generate_receipt(
        report_id="TEST-001",
        engine_name="ActionProof",
        software_version="0.1.0",
        inputs_hash="abcdef123456",
        measurements={"displacement": 120.5},
        signals=[signal],
        verdict="UNCERTAIN",
        recapture_directive="Move camera back",
    )
    assert receipt.receipt_sha256 is not None
    assert len(receipt.receipt_sha256) == 64
    assert verify_receipt_integrity(receipt) is True

    # Tampering test: altering verdict invalidates hash check
    receipt.verdict = "SUCCESS"
    assert verify_receipt_integrity(receipt) is False
