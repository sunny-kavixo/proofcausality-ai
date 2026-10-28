"""Tests for the Cost-Aware Evidence Budget Planner (H03)."""

from backend.evidence.calculus.budget_planner import EvidenceBudgetPlanner
from backend.evidence.calculus.schemas import (
    EvidenceDebtReport,
    EvidenceState,
    ObservablePredicate,
    PredicateType,
)


def test_planner_selects_optimal_recovery_action():
    planner = EvidenceBudgetPlanner(cost_lambda=0.05)

    pred_blur = ObservablePredicate(
        predicate_id="P_BLUR",
        predicate_type=PredicateType.OBSERVATION_SHARPNESS,
        description="Final frame blur variance below 100.0",
        state=EvidenceState.UNOBSERVABLE,
    )
    pred_clip = ObservablePredicate(
        predicate_id="P_CLIP",
        predicate_type=PredicateType.CANVAS_CONTAINMENT,
        description="Object clipped by image edge",
        state=EvidenceState.UNOBSERVABLE,
    )

    debt_report = EvidenceDebtReport(
        total_debt=2.0,
        missing_evidence_debt=2.0,
        contradictory_evidence_debt=0.0,
        unreliability_debt=0.0,
        is_acceptable=False,
        critical_unresolved_predicates=["BLUR", "CLIP"],
        contradictions=[],
    )

    action = planner.plan_recovery_action(
        debt_report=debt_report,
        predicates=[pred_blur, pred_clip],
    )

    assert action is not None
    assert "Hold Camera Steady" in action.name or "Step Back" in action.name
    assert action.net_utility_score > 0.0
    assert action.expected_risk_reduction > 0.70


def test_budget_limit_filtering():
    planner = EvidenceBudgetPlanner(cost_lambda=0.05)
    pred_obj = ObservablePredicate(
        predicate_id="P_OBJ",
        predicate_type=PredicateType.OBJECT_IDENTIFICATION,
        description="Cup identity unverified",
        state=EvidenceState.UNOBSERVABLE,
    )

    debt_report = EvidenceDebtReport(
        total_debt=1.0,
        missing_evidence_debt=1.0,
        contradictory_evidence_debt=0.0,
        unreliability_debt=0.0,
        is_acceptable=False,
    )

    # Human verification effort is 12 seconds; tight 5 second budget should filter it out
    action = planner.plan_recovery_action(
        debt_report=debt_report,
        predicates=[pred_obj],
        budget_limit_sec=5.0,
    )

    assert action is None
