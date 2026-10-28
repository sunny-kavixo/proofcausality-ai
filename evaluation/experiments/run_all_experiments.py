"""Runner for all Phase 2 scientific research experiments."""

import json
from pathlib import Path

from evaluation.experiments.exp_h01_evidence_debt import run_experiment_h01
from evaluation.experiments.exp_h02_dependency_graph import run_experiment_h02
from evaluation.experiments.exp_h03_budget_planner import run_experiment_h03
from evaluation.experiments.exp_h04_detector_independence import run_experiment_h04

if __name__ == "__main__":
    print("Running Experiment H01: Physics-Constrained Evidence Debt...")
    res_h01 = run_experiment_h01()

    print("Running Experiment H02: Evidence Dependency Graph...")
    res_h02 = run_experiment_h02()

    print("Running Experiment H03: Evidence Budget Planner...")
    res_h03 = run_experiment_h03()

    print("Running Experiment H04: Detector Independence Audit...")
    res_h04 = run_experiment_h04()

    consolidated = {
        "h01_evidence_debt": res_h01,
        "h02_dependency_graph": res_h02,
        "h03_budget_planner": res_h03,
        "h04_detector_independence": res_h04,
    }

    out_path = Path("/Users/sunny/Projects/proofcausality-ai/evaluation/experiments/EXPERIMENT_RESULTS.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)

    print(f"\nAll experiments completed successfully. Results saved to {out_path.name}")
    print(json.dumps(consolidated, indent=2))
