"""Deterministic ProofLoop closed-loop benchmark.

This benchmark exercises the state machine with measured feature fixtures.  It
is a software-policy benchmark, not a human study, not a perception accuracy
benchmark, and not evidence of physical causality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.evidence.proofloop import (
    ProofLoopObservation,
    ProofLoopVerdict,
    ProofLoopVerifier,
)


def box(x_min: float, y_min: float, x_max: float, y_max: float) -> dict[str, float]:
    return {"x_min": x_min, "y_min": y_min, "x_max": x_max, "y_max": y_max}


def cases() -> list[dict[str, Any]]:
    clean_before = box(50, 50, 100, 100)
    clean_after = box(310, 110, 390, 190)
    clean_target = box(300, 100, 400, 200)
    clean_initial = box(20, 20, 120, 120)

    return [
        {
            "case_id": "missing_final_state",
            "description": "No post-action object observation is available.",
            "initial": dict(
                after_bbox=None,
                has_final_state=False,
                laplacian_variance=250.0,
            ),
            "followup": dict(after_bbox=clean_after, has_final_state=True, laplacian_variance=250.0),
        },
        {
            "case_id": "blurred_object",
            "description": "Final geometry is present but below the sharpness threshold.",
            "initial": dict(after_bbox=clean_after, laplacian_variance=20.0),
            "followup": dict(after_bbox=clean_after, laplacian_variance=250.0),
        },
        {
            "case_id": "border_clipped_object",
            "description": "The target is visible at the frame edge, then recaptured with margin.",
            "initial": dict(
                before_bbox=box(300, 50, 350, 100),
                initial_region=box(280, 30, 370, 120),
                target_region=box(0, 100, 100, 200),
                after_bbox=box(0, 110, 80, 190),
            ),
            "followup": dict(after_bbox=box(20, 110, 80, 190)),
        },
        {
            "case_id": "class_mismatch",
            "description": "Geometry agrees, but the supplied claim class conflicts with the observation.",
            "initial": dict(after_bbox=clean_after, claim_class_agreement=False),
            "followup": None,
        },
        {
            "case_id": "partial_occlusion",
            "description": "A large measured area drop makes the final surface unobservable.",
            "initial": dict(after_bbox=clean_after, area_drop_ratio=0.60),
            "followup": dict(after_bbox=clean_after, area_drop_ratio=0.0),
        },
        {
            "case_id": "incorrect_annotation",
            "description": "The claimed destination does not contain the observed object.",
            "initial": dict(after_bbox=box(450, 300, 520, 380)),
            "followup": None,
        },
        {
            "case_id": "ambiguous_model_output",
            "description": "All geometry is adequate, but upstream detector confidence is below policy threshold.",
            "initial": dict(after_bbox=clean_after, model_confidence=0.20),
            "followup": dict(after_bbox=clean_after, model_confidence=0.95),
        },
        {
            "case_id": "successfully_resolved_uncertainty",
            "description": "A blurred observation is resolved by a sharp follow-up.",
            "initial": dict(after_bbox=clean_after, laplacian_variance=15.0),
            "followup": dict(after_bbox=clean_after, laplacian_variance=250.0),
        },
        {
            "case_id": "unresolved_after_recapture",
            "description": "A follow-up is supplied but remains too blurred to decide.",
            "initial": dict(after_bbox=clean_after, laplacian_variance=15.0),
            "followup": dict(after_bbox=clean_after, laplacian_variance=15.0),
        },
        {
            "case_id": "clean_success",
            "description": "All deterministic evidence predicates are supported on the first observation.",
            "initial": dict(after_bbox=clean_after),
            "followup": None,
        },
    ]


def make_observation(case_id: str, overrides: dict[str, Any]) -> ProofLoopObservation:
    values: dict[str, Any] = {
        "task_id": f"PROOFLOOP_BENCH_{case_id.upper()}",
        "primary_object": "cup",
        "before_bbox": box(50, 50, 100, 100),
        "after_bbox": box(310, 110, 390, 190),
        "target_region": box(300, 100, 400, 200),
        "initial_region": box(20, 20, 120, 120),
        "canvas_width": 640,
        "canvas_height": 480,
        "laplacian_variance": 250.0,
        "area_drop_ratio": 0.0,
        "has_final_state": True,
    }
    values.update(overrides)
    return ProofLoopObservation(**values)


def run_benchmark() -> dict[str, Any]:
    verifier = ProofLoopVerifier()
    results: list[dict[str, Any]] = []
    initial_uncertain = 0
    resolved = 0

    for case in cases():
        initial = verifier.evaluate(make_observation(case["case_id"], case["initial"]))
        followup_result = None
        if initial.semantic_verdict == ProofLoopVerdict.INSUFFICIENT_EVIDENCE:
            initial_uncertain += 1
        if case["followup"] is not None:
            followup_values = make_observation(case["case_id"], case["initial"]).model_dump()
            followup_values.update(case["followup"])
            followup_observation = ProofLoopObservation.model_validate(followup_values)
            followup_result = verifier.evaluate(
                followup_observation,
                phase="followup",
                previous_trace=initial.state_trace,
            )
            if (
                initial.semantic_verdict == ProofLoopVerdict.INSUFFICIENT_EVIDENCE
                and followup_result.semantic_verdict != ProofLoopVerdict.INSUFFICIENT_EVIDENCE
            ):
                resolved += 1

        result = {
            "case_id": case["case_id"],
            "description": case["description"],
            "initial": {
                "state": initial.state.value,
                "state_trace": [state.value for state in initial.state_trace],
                "legacy_action_verdict": initial.legacy_action_verdict.value,
                "semantic_verdict": initial.semantic_verdict.value,
                "decision_basis": initial.decision_basis,
                "evidence_debt": initial.debt_report.total_debt,
                "observation_request": initial.observation_request,
            },
            "followup": None,
        }
        if followup_result is not None:
            result["followup"] = {
                "state": followup_result.state.value,
                "state_trace": [state.value for state in followup_result.state_trace],
                "legacy_action_verdict": followup_result.legacy_action_verdict.value,
                "semantic_verdict": followup_result.semantic_verdict.value,
                "decision_basis": followup_result.decision_basis,
                "evidence_debt": followup_result.debt_report.total_debt,
                "resolved": followup_result.semantic_verdict != ProofLoopVerdict.INSUFFICIENT_EVIDENCE,
            }
        results.append(result)

    return {
        "benchmark": "ProofLoop Closed-Loop Evidence Verifier",
        "benchmark_version": "proofloop-v1",
        "scope": "Deterministic measured-feature fixtures; no human or causal validation.",
        "case_count": len(results),
        "initial_uncertain_cases": initial_uncertain,
        "resolved_uncertainty_cases": resolved,
        "resolution_rate": (resolved / initial_uncertain) if initial_uncertain else None,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/benchmarks/proofloop_closed_loop_report.json"),
    )
    args = parser.parse_args()
    report = run_benchmark()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "results"}, indent=2))


if __name__ == "__main__":
    main()
