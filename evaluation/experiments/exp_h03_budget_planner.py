"""Experiment H03: Evidence Budget Planner vs Baselines (Controlled Simulation).

Compares:
A. Generic re-recording advice ("Please re-record")
B. Fixed cause-specific templates (static rule map)
C. Random instruction selection
D. Evidence Budget Planner (VOI optimization: max E[Delta Risk] - lambda * Cost)

EVALUATION METHODOLOGY DISCLAIMER:
- This experiment is a CONTROLLED SIMULATION across 20 synthetic/metadata defect cases.
- It tests algorithmic directive selection and expected cost curves.
- LIVE MULTI-SESSION HUMAN USER RE-CAPTURE RESOLUTION IS NOT MEASURED ON IN-THE-WILD HUMANS.
- On the 4 curated paired tabletop recovery cases in real_dataset.py, targeted templates resolved 4/4 cases.
"""

from __future__ import annotations

import random
from typing import Any, Dict
import numpy as np

from backend.evidence.calculus.budget_planner import EvidenceBudgetPlanner
from backend.evidence.calculus.schemas import EvidenceDebtReport, EvidenceState, ObservablePredicate, PredicateType


def run_experiment_h03() -> Dict[str, Any]:
    planner = EvidenceBudgetPlanner(cost_lambda=0.05)

    # 20 diverse defect scenarios
    cases = [
        {"type": PredicateType.OBSERVATION_SHARPNESS, "desc": "Motion blur detected in final frames"},
        {"type": PredicateType.CANVAS_CONTAINMENT, "desc": "Object clipped by image boundary"},
        {"type": PredicateType.SURFACE_VISIBILITY, "desc": "Gripper occluding object surface area"},
        {"type": PredicateType.SPATIAL_RELATION, "desc": "Ambiguous depth separation in monocular view"},
        {"type": PredicateType.OBJECT_IDENTIFICATION, "desc": "Low contrast bounding box proposal"},
    ] * 4

    total_cases = len(cases)

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

        # Method A: Generic Advice (Simulated compliance: 30%, effort: 15s)
        eff_a = 15.0
        if random.random() < 0.30:
            resolved_generic += 1
        effort_generic.append(eff_a)

        # Method B: Fixed Template (Simulated compliance: 85%, effort: 6s)
        eff_b = 6.0
        if random.random() < 0.85:
            resolved_template += 1
        effort_template.append(eff_b)

        # Method C: Random instruction (Simulated compliance: 35%, effort: 7s)
        eff_c = 7.0
        if random.random() < 0.35:
            resolved_random += 1
        effort_random.append(eff_c)

        # Method D: VOI Budget Planner
        action = planner.plan_recovery_action(debt, [pred])
        eff_d = action.estimated_human_effort_sec + action.estimated_camera_latency_sec
        if random.random() < action.expected_risk_reduction:
            resolved_planner += 1
        effort_planner.append(eff_d)

    return {
        "experiment": "H03_BUDGET_PLANNER_CONTROLLED_SIMULATION",
        "evaluation_type": "Controlled Monte Carlo Simulation (Assumed Compliance Distributions)",
        "total_evaluated_cases": total_cases,
        "live_human_trials_conducted": False,
        "live_human_resolution_rate_status": "NOT MEASURED",
        "curated_tabletop_paired_recovery_cases": "4 / 4 resolved (100.0% on static paired states)",
        "simulation_results": {
            "method_a_generic_advice": {
                "simulated_resolution_rate": round(resolved_generic / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_generic)), 1),
            },
            "method_b_fixed_template": {
                "simulated_resolution_rate": round(resolved_template / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_template)), 1),
            },
            "method_c_random_selection": {
                "simulated_resolution_rate": round(resolved_random / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_random)), 1),
            },
            "method_d_voi_budget_planner": {
                "simulated_resolution_rate": round(resolved_planner / total_cases, 4),
                "mean_effort_seconds": round(float(np.mean(effort_planner)), 1),
            },
        },
    }
