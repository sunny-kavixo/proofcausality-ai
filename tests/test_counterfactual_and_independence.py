"""Tests for Counterfactual Alternative Worlds and Detector Independence."""

from backend.evidence.calculus.counterfactual import CounterfactualConsistencyChecker
from backend.evidence.calculus.independence import DetectorIndependenceAuditor


def test_counterfactual_temporal_gap_analysis():
    checker = CounterfactualConsistencyChecker(max_occlusion_gap_frames=5)

    # Case A: Short gap with smooth motion trail -> genuine transfer plausible
    hyps_a = checker.analyze_temporal_gap(
        observed_gap_frames=2,
        bounding_box_displacement_px=50.0,
        has_continuous_motion_trail=True,
    )
    assert not checker.is_decision_underdetermined(hyps_a)

    # Case B: Large gap without motion trail -> object swap consistent & plausible
    hyps_b = checker.analyze_temporal_gap(
        observed_gap_frames=12,
        bounding_box_displacement_px=200.0,
        has_continuous_motion_trail=False,
    )
    assert checker.is_decision_underdetermined(hyps_b)


def test_detector_independence_correlation_and_fusion():
    auditor = DetectorIndependenceAuditor(correlation_discount=0.5)

    errs_det1 = [False, False, True, False, True, False, False, True, False, False]
    errs_det2 = [False, False, True, False, True, False, False, True, False, False]  # Identical errors
    errs_det3 = [False, True, False, False, False, False, True, False, False, False]  # Independent errors

    corr_12 = auditor.compute_empirical_error_correlation(errs_det1, errs_det2)
    corr_13 = auditor.compute_empirical_error_correlation(errs_det1, errs_det3)

    assert corr_12 == 1.0
    assert corr_13 < 0.2

    # Arbitration test where model 1 and model 2 share identical correlated positive predictions
    predictions = {"det1": True, "det2": True, "det3": False}
    confidences = {"det1": 0.90, "det2": 0.88, "det3": 0.95}
    pairwise = {("det1", "det2"): 0.95, ("det1", "det3"): 0.05, ("det2", "det3"): 0.05}

    results = auditor.arbitrate_multi_model_verdicts(predictions, confidences, pairwise)
    assert "dependence_aware" in results
    assert "majority_vote" in results
    assert results["majority_vote"]["verdict"] is True
    # Dependence-aware discounts redundant correlated models 1 and 2
    assert results["dependence_aware"]["calibrated_score"] < results["confidence_weighted"]["weighted_score"]
