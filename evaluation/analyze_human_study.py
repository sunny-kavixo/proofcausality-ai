"""Analyze the pre-registered ProofLoop human-study CSV.

This runner intentionally requires participant data supplied by the study
owner. It never creates annotations or fills missing responses.

Expected columns:
``case_id,condition,initial_insufficient,system_resolved,annotator_a_resolved,annotator_b_resolved,seconds,retakes``
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "case_id",
    "condition",
    "initial_insufficient",
    "system_resolved",
    "annotator_a_resolved",
    "annotator_b_resolved",
}


def as_bool(value: str, field: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false", "1", "0", "yes", "no"}:
        raise ValueError(f"row {row_number}: {field} must be boolean, got {value!r}")
    return normalized in {"true", "1", "yes"}


def cohen_kappa(first: list[bool], second: list[bool]) -> float | None:
    if not first or len(first) != len(second):
        return None
    agreement = sum(a == b for a, b in zip(first, second)) / len(first)
    first_true = sum(first) / len(first)
    second_true = sum(second) / len(second)
    expected = first_true * second_true + (1.0 - first_true) * (1.0 - second_true)
    if expected == 1.0:
        return 1.0 if agreement == 1.0 else 0.0
    return (agreement - expected) / (1.0 - expected)


def analyze(path: Path) -> dict[str, Any]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or [])
        missing = sorted(REQUIRED_COLUMNS - columns)
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")

        rows: list[dict[str, Any]] = []
        for row_number, raw in enumerate(reader, start=2):
            if not raw.get("case_id", "").strip():
                raise ValueError(f"row {row_number}: case_id is required")
            a = as_bool(raw["annotator_a_resolved"], "annotator_a_resolved", row_number)
            b = as_bool(raw["annotator_b_resolved"], "annotator_b_resolved", row_number)
            rows.append({
                "case_id": raw["case_id"].strip(),
                "condition": raw["condition"].strip(),
                "initial_insufficient": as_bool(raw["initial_insufficient"], "initial_insufficient", row_number),
                "system_resolved": as_bool(raw["system_resolved"], "system_resolved", row_number),
                "annotator_a_resolved": a,
                "annotator_b_resolved": b,
                "adjudicated_resolved": a if a == b else None,
            })

    if not rows:
        return {
            "status": "NO_DATA",
            "note": "No participant rows supplied; independent human validation is not completed.",
            "row_count": 0,
        }

    adjudicated = [row for row in rows if row["adjudicated_resolved"] is not None]
    resolved_by_condition: dict[str, dict[str, int]] = defaultdict(lambda: {"resolved": 0, "total": 0})
    for row in adjudicated:
        bucket = resolved_by_condition[row["condition"]]
        bucket["total"] += 1
        bucket["resolved"] += int(row["adjudicated_resolved"])

    system_true = [row["system_resolved"] for row in adjudicated]
    human_true = [row["adjudicated_resolved"] for row in adjudicated]
    true_positive = sum(s and h for s, h in zip(system_true, human_true))
    false_positive = sum(s and not h for s, h in zip(system_true, human_true))
    true_negative = sum(not s and not h for s, h in zip(system_true, human_true))
    false_negative = sum(not s and h for s, h in zip(system_true, human_true))

    return {
        "status": "ANALYZED",
        "row_count": len(rows),
        "adjudicated_row_count": len(adjudicated),
        "annotator_disagreements": len(rows) - len(adjudicated),
        "inter_rater_cohen_kappa": cohen_kappa(
            [row["annotator_a_resolved"] for row in rows],
            [row["annotator_b_resolved"] for row in rows],
        ),
        "resolution_by_condition": dict(resolved_by_condition),
        "system_vs_adjudicated_confusion": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "true_negative": true_negative,
            "false_negative": false_negative,
        },
        "disclosure": "Results depend on authentic participant responses and blinded adjudication; no rows are synthesized.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
