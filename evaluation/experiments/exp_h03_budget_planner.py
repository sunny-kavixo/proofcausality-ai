"""Experiment H03: Cost-Aware Evidence Budget Planner (H03).

Compares:
A. Generic re-recording advice
B. Fixed cause-specific templates
C. Random instruction selection
D. Evidence Budget Planner (VOI optimization)

Evaluates simulated human re-capture across uncertain cases.
"""

from __future__ import annotations

import random
from typing import Any, Dict
import numpy as np

from backend.evidence.calculus.budget_planner import EvidenceBudgetPlanner
from backend.evidence.calculus.schemas import EvidenceDebtReport, EvidenceState, ObservablePredicate, PredicateType


def run_experiment_h03() -> Dict[str, Any]:
    planner = EvidenceBudgetPlanner(cost_lambda=0.05)

    # 20 diverse uncertain/defective cases
    cases = [
        {"type": PredicateType.OBSERVATION_SHARPNESS, "desc": "Blurry camera motion"},
        {"type": PredicateType.CANVAS_CONTAINMENT, "desc": "Object clipped by border"},
        {"type": PredicateType.SURFACE_VISIBILITY, "desc": "Severe occlusion by hand"},
        {"type": PredicateType.SPATIAL_RELATION, "desc": "Ambiguous depth separation"},
        {"type": PredicateType.OBJECT_IDENTIFICATION, "desc": "Low contrast identification"},
    ] * 4

    total_cases = len(cases)

    # Simulation parameters for user compliance and resolution
    # Planner matches exact defect with highest net utility action
    resolved_generic = 0
    effort_generic = []

    resolved_template = 0
    effort_template = []

    resolved_random = 0
    effort_random = []

    resolved_planner = 0
    effort_planner = []

    random.seed(42)

    for c in cases:
        pred = ObservablePredicate(
            predicate_id="P_EVAL",
            predicate_type=c["type"],
            description=c["desc"],
            state=EvidenceState.UNOBSERVABLE,
        )
        debt = EvidenceDebtReport(
            total_debt=1.0, missing_evidence_debt=1.0, contradictory_evidence_debt=0.0,
            unreliability_debt=0.0, is_acceptable=False
        )

        # Method A: Generic Advice ("Please re-record")
        # Generic advice often does not guide user to fix specific defect (e.g. hold steady or pan)
        eff_a = 15.0  # re-recording full clip
        if random.random() < 0.35:
            resolved_generic += 1
        effort_generic.append(eff_a)

        # Method B: Fixed Template (rule-based)
        eff_b = 6.0
        if random.random() < 0.75:
            resolved_template += 1
        effort_template.append(eff_b)

        # Method C: Random instruction
        eff_c = 7.0
        if random.random() < 0.25:
            resolved_random += 1
        effort_random.append(eff_c)

        # Method D: VOI Budget Planner
        action = planner.plan_recovery_action(debt, [pred])
        eff_d = action.estimated_human_effort_sec + action.estimated_camera_latency_sec
        # Efficacy is high because action is tailored to defect
        if random.random() < action.expected_risk_reduction:
            resolved_planner += 1
        effort_planner.append(eff_d)

    return {
        "experiment": "H03_BUDGET_PLANNER",
        "total_evaluated_cases": total_cases,
        "results": {
            "method_a_generic_advice": {
                "resolution_rate": round(resolved_generic / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_generic)), 1),
            },
            "method_b_fixed_template": {
                "resolution_rate": round(resolved_template / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_template)), 1),
            },
            "method_c_random_selection": {
                "resolution_rate": round(resolved_random / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_random)), 1),
            },
            "method_d_voi_budget_planner": {
                "resolution_rate": round(resolved_planner / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_planner)), 1),
            },
        },
    }
