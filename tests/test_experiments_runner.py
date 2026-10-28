"""Tests for the 4 scientific research experiments."""

from evaluation.experiments.exp_h01_evidence_debt import run_experiment_h01
from evaluation.experiments.exp_h02_dependency_graph import run_experiment_h02
from evaluation.experiments.exp_h03_budget_planner import run_experiment_h03
from evaluation.experiments.exp_h04_detector_independence import run_experiment_h04


def test_experiment_h01_evidence_debt_execution():
    res = run_experiment_h01()
    assert res["experiment"] == "H01_EVIDENCE_DEBT_FAIR_COMPARISON"
    assert res["sample_size"] == 37
    # Method 5 (Evidence Calculus Full) must achieve 0.0 false-success rate
    assert res["results"]["method_5_evidence_calculus_full"]["false_success_rate"] == 0.0
    # Baseline 1 naive heuristic false success rate must be positive
    assert res["results"]["baseline_1_naive_center_distance"]["false_success_rate"] > 0.30
    # Must report 95% Wilson confidence intervals
    assert len(res["results"]["method_5_evidence_calculus_full"]["false_success_rate_wilson_ci_95"]) == 2


def test_experiment_h02_dependency_graph_parity():
    res = run_experiment_h02()
    assert res["experiment"] == "H02_DEPENDENCY_GRAPH"
    # Strict mathematical parity must hold
    assert res["mathematical_parity_rate"] == 1.0
    assert res["results"]["method_c_incremental_recomputation"]["mean_skipped_nodes"] > 0


def test_experiment_h03_budget_planner_voi():
    res = run_experiment_h03()
    assert res["experiment"] == "H03_BUDGET_PLANNER_CONTROLLED_SIMULATION"
    assert res["live_human_resolution_rate_status"] == "NOT MEASURED"
    sim_res = res["simulation_results"]
    assert sim_res["method_d_voi_budget_planner"]["simulated_resolution_rate"] > sim_res["method_a_generic_advice"]["simulated_resolution_rate"]


def test_experiment_h04_detector_independence():
    res = run_experiment_h04()
    assert res["experiment"] == "H04_DETECTOR_INDEPENDENCE_SIMULATION"
    assert res["empirical_real_world_validation_status"] == "NOT YET MEASURED ON LIVE MODELS"
    sim_res = res["simulation_results"]
    assert sim_res["dependence_aware_false_success_rate"] < sim_res["majority_vote_false_success_rate"]
