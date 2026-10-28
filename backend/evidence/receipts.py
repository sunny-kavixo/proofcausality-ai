"""Deterministic JSON evidence receipts with cryptographic SHA-256 integrity verification."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, List, Optional

from backend.evidence.schemas import EvidenceReceipt, UncertaintySignal


def canonical_json_dump(obj: Any) -> str:
    """Deterministic canonical JSON serialization for cryptographic hashing."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def calculate_sha256(data: bytes | str) -> str:
    """Compute SHA-256 hex digest of string or raw bytes."""
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
) -> EvidenceReceipt:
    """Generate a tamper-evident cryptographic Evidence Receipt."""
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
    }
    recalculated_hash = calculate_sha256(canonical_json_dump(payload_for_hashing))
    return recalculated_hash == receipt.receipt_sha256
