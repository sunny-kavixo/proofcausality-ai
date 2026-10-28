"""Deterministic JSON evidence receipts with reproducible SHA-256 payload integrity verification.

Scientific Disclosure:
- Evidence Receipts provide a reproducible SHA-256 integrity checksum over a canonicalized JSON payload.
- This mechanism enables any party to detect post-hoc tampering of recorded measurements, verdicts, signals, or provenance.
- This is NOT an HMAC (no shared secret key), NOT a digital signature (no private key), and NOT a Merkle tree.
- A publicly recomputable hash guarantees internal payload consistency; it does not prove physical truth or authentic author identity.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, List, Optional

from backend.evidence.schemas import EvidenceReceipt, UncertaintySignal


def canonical_json_dump(obj: Any) -> str:
    """Deterministic canonical JSON serialization for SHA-256 payload hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def calculate_sha256(data: bytes | str) -> str:
    """Compute standard unkeyed SHA-256 hex digest of string or raw bytes."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def generate_receipt(
    report_id: str,
    engine_name: str,
    software_version: str,
    inputs_hash: str,
    measurements: dict[str, Any],
    signals: List[UncertaintySignal],
    verdict: str,
    recapture_directive: Optional[str] = None,
    provenance: Optional[dict[str, Any]] = None,
) -> EvidenceReceipt:
    """Generate a tamper-evident Evidence Receipt with a canonical SHA-256 integrity checksum."""
    timestamp = datetime.now(timezone.utc).isoformat()
    signals_dict = [s.model_dump() for s in signals]

    payload_for_hashing = {
        "report_id": report_id,
        "timestamp": timestamp,
        "software_version": software_version,
        "engine_name": engine_name,
        "inputs_hash": inputs_hash,
        "measurements": measurements,
        "uncertainty_signals": signals_dict,
        "verdict": verdict,
        "recapture_directive": recapture_directive,
        "provenance": provenance,
    }

    serialized = canonical_json_dump(payload_for_hashing)
    receipt_sha256 = calculate_sha256(serialized)

    return EvidenceReceipt(
        report_id=report_id,
        timestamp=timestamp,
        software_version=software_version,
        engine_name=engine_name,
        inputs_hash=inputs_hash,
        measurements=measurements,
        uncertainty_signals=signals_dict,
        verdict=verdict,
        recapture_directive=recapture_directive,
        provenance=provenance,
        receipt_sha256=receipt_sha256,
    )


def verify_receipt_integrity(receipt: EvidenceReceipt) -> bool:
    """Recompute receipt hash from payload fields and verify cryptographic match."""
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
    recalculated_hash = calculate_sha256(canonical_json_dump(payload_for_hashing))
    return recalculated_hash == receipt.receipt_sha256
