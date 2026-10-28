"""Scientific evaluation harness computing False-Success Rate and Re-Capture Resolution Rate."""

from __future__ import annotations
import numpy as np

import csv
import json
import os
import sys

from backend.actionproof.verifier import ActionProofVerifier
from backend.evidence.masks import bbox_to_mask
from backend.evidence.schemas import ActionVerdict, ActionVerificationRequest
from evaluation.cases.action_dataset import get_standard_action_benchmark_cases


DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "benchmarks")

def run_benchmarks(output_dir: str = DEFAULT_OUTPUT_DIR) -> dict[str, Any]:
    """Execute complete scientific evaluation over all standardized test cases."""
    os.makedirs(output_dir, exist_ok=True)
    cases = get_standard_action_benchmark_cases()
    verifier = ActionProofVerifier(blur_threshold=100.0, boundary_margin_px=3, occlusion_drop_threshold=0.35)

    total_cases = len(cases)
    total_non_success_cases = sum(1 for c in cases if c.ground_truth in ("FAILED", "UNCERTAIN"))
    total_uncertain_cases = sum(1 for c in cases if c.ground_truth == "UNCERTAIN")

    # Metrics accumulators
    proofcausality_false_successes = 0
    naive_baseline_false_successes = 0

    proofcausality_resolved_uncertain = 0
    generic_baseline_resolved_uncertain = 0

    records = []

    for c in cases:
        before_mask, after_mask, image_gray = c.get_masks()
        req = ActionVerificationRequest(
            task_type=c.task_type,
            object_name="cube",
            region_a=c.region_a,
            region_b=c.region_b,
            distance_threshold_px=60.0,
        )

        res, receipt = verifier.verify_action(
            request=req,
            before_mask=before_mask,
            after_mask=after_mask,
            after_image_gray=image_gray,
            has_final_state=c.has_final_state,
        )

        # Naive Baseline: Ignores blur/occlusion/framing and only checks bounding box coordinates
        c_before = c.before_bbox.centroid
        c_after = c.after_bbox.centroid
        disp = ((c_before[0] - c_after[0])**2 + (c_before[1] - c_after[1])**2)**0.5
        naive_verdict = "SUCCESS" if disp >= 60.0 else "FAILED"

        # Check false-success on non-success ground-truth cases
        if c.ground_truth != "SUCCESS":
            if res.verdict == ActionVerdict.SUCCESS:
                proofcausality_false_successes += 1
            if naive_verdict == "SUCCESS":
                naive_baseline_false_successes += 1

        # ReCapture simulation on uncertain cases
        recapture_resolved = False
        generic_resolved = False
        if c.ground_truth == "UNCERTAIN":
            # ReCapture case following targeted instruction:
            recaptured_after_mask = bbox_to_mask(c.recaptured_after_bbox, 400, 600)
            # Re-captured frame is sharp, unobstructed, and within frame
            sharp_gray = np.full((400, 600), 128, dtype=np.uint8)
            sharp_gray[::4, ::4] = 240
            sharp_gray[1::4, 1::4] = 10

            res_recap, _ = verifier.verify_action(
                request=req,
                before_mask=before_mask,
                after_mask=recaptured_after_mask,
                after_image_gray=sharp_gray,
                has_final_state=True,
            )
            if res_recap.verdict.value == c.recaptured_ground_truth:
                proofcausality_resolved_uncertain += 1
                recapture_resolved = True

            if c.generic_recaptured_ground_truth != "UNCERTAIN":
                generic_baseline_resolved_uncertain += 1
                generic_resolved = True

        records.append({
            "case_id": c.case_id,
            "name": c.name,
            "ground_truth": c.ground_truth,
            "proofcausality_verdict": res.verdict.value,
            "naive_baseline_verdict": naive_verdict,
            "signals_detected": [s.reason.value for s in res.uncertainty_signals],
            "recapture_directive": res.recapture_plan.directive if res.recapture_plan else None,
            "recapture_resolved": recapture_resolved if c.ground_truth == "UNCERTAIN" else None,
        })

    # Mathematical metric calculations
    fsr_proofcausality = float(proofcausality_false_successes) / float(total_non_success_cases)
    fsr_naive = float(naive_baseline_false_successes) / float(total_non_success_cases)

    rrr_proofcausality = float(proofcausality_resolved_uncertain) / float(total_uncertain_cases)
    rrr_generic = float(generic_baseline_resolved_uncertain) / float(total_uncertain_cases)

    benchmark_summary = {
        "evaluation_name": "ProofCausality AI Physical Action & Uncertainty Benchmark",
        "sample_size": total_cases,
        "ground_truth_composition": {
            "SUCCESS": sum(1 for c in cases if c.ground_truth == "SUCCESS"),
            "FAILED": sum(1 for c in cases if c.ground_truth == "FAILED"),
            "UNCERTAIN": total_uncertain_cases,
        },
        "false_success_rate": {
            "description": "False Success Count / Total Non-Success Cases",
            "denominator": total_non_success_cases,
            "proofcausality_fsr": round(fsr_proofcausality, 4),
            "proofcausality_percentage": f"{fsr_proofcausality * 100:.1f}%",
            "naive_baseline_fsr": round(fsr_naive, 4),
            "naive_baseline_percentage": f"{fsr_naive * 100:.1f}%",
            "relative_reduction": f"{(fsr_naive - fsr_proofcausality) / max(0.001, fsr_naive) * 100:.1f}%",
        },
        "recapture_resolution_rate": {
            "description": "Initially Uncertain Cases Decided Correctly Post-Recapture / Total Uncertain Cases",
            "denominator": total_uncertain_cases,
            "proofcausality_rrr": round(rrr_proofcausality, 4),
            "proofcausality_percentage": f"{rrr_proofcausality * 100:.1f}%",
            "generic_baseline_rrr": round(rrr_generic, 4),
            "generic_baseline_percentage": f"{rrr_generic * 100:.1f}%",
            "relative_gain": f"{(rrr_proofcausality - rrr_generic) * 100:.1f}% improvement",
        },
        "detailed_records": records,
    }

    # Write JSON results
    json_path = os.path.join(output_dir, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)
    syn_path = os.path.join(output_dir, "benchmark_results_synthetic.json")
    with open(syn_path, "w") as f:
        json.dump(benchmark_summary, f, indent=2)

    # Write CSV summary
    csv_path = os.path.join(output_dir, "benchmark_cases.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "case_id", "name", "ground_truth", "proofcausality_verdict", "naive_baseline_verdict", "signals_detected", "recapture_directive", "recapture_resolved"
        ])
        writer.writeheader()
        for r in records:
            writer.writerow({
                "case_id": r["case_id"],
                "name": r["name"],
                "ground_truth": r["ground_truth"],
                "proofcausality_verdict": r["proofcausality_verdict"],
                "naive_baseline_verdict": r["naive_baseline_verdict"],
                "signals_detected": "; ".join(r["signals_detected"]),
                "recapture_directive": r["recapture_directive"] or "N/A",
                "recapture_resolved": str(r["recapture_resolved"]),
            })

    print(f"Benchmark completed successfully over {total_cases} cases.")
    print(f"ProofCausality False-Success Rate: {fsr_proofcausality * 100:.1f}% vs Naive: {fsr_naive * 100:.1f}%")
    print(f"ProofCausality ReCapture Resolution Rate: {rrr_proofcausality * 100:.1f}% vs Generic: {rrr_generic * 100:.1f}%")
    return benchmark_summary


if __name__ == "__main__":
    import numpy as np
    run_benchmarks()
