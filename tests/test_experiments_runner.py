"""Tests for the 4 scientific research experiments."""

from evaluation.experiments.exp_h01_evidence_debt import run_experiment_h01
from evaluation.experiments.exp_h02_dependency_graph import run_experiment_h02
from evaluation.experiments.exp_h03_budget_planner import run_experiment_h03
from evaluation.experiments.exp_h04_detector_independence import run_experiment_h04


def test_experiment_h01_evidence_debt_execution():
    res = run_experiment_h01()
    assert res["experiment"] == "H01_EVIDENCE_DEBT"
    assert res["sample_size"] == 37
    # Method D must achieve 0.0 false-success rate
    assert res["results"]["method_d_evidence_debt_plus_contradiction"]["false_success_rate"] == 0.0
    # Method A confidence baseline must fail
    assert res["results"]["method_a_confidence_only"]["false_success_rate"] == 1.0


def test_experiment_h02_dependency_graph_parity():
    res = run_experiment_h02()
    assert res["experiment"] == "H02_DEPENDENCY_GRAPH"
    # Strict mathematical parity must hold
    assert res["mathematical_parity_rate"] == 1.0
    assert res["results"]["method_c_incremental_recomputation"]["mean_skipped_nodes"] > 0


def test_experiment_h03_budget_planner_voi():
    res = run_experiment_h03()
    assert res["experiment"] == "H03_BUDGET_PLANNER"
    # VOI planner must beat generic advice
    assert res["results"]["method_d_voi_budget_planner"]["resolution_rate"] > res["results"]["method_a_generic_advice"]["resolution_rate"]


def test_experiment_h04_detector_independence():
    res = run_experiment_h04()
    assert res["experiment"] == "H04_DETECTOR_INDEPENDENCE"
    # Dependence-aware fusion must reduce false-success rate over majority vote
    assert res["results"]["dependence_aware_false_success_rate"] < res["results"]["majority_vote_false_success_rate"]
