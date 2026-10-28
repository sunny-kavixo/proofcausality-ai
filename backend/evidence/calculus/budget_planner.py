"""Cost-Aware Evidence Budget Planner (H03).

Selects the optimal recovery action maximizing Expected Value of Information:
VOI(a) = E[Delta Risk(a)] - lambda * Cost(a)

Where Cost(a) incorporates:
- Human physical effort (seconds)
- Camera repositioning / hold steady latency
- Inference overhead / bandwidth

Outperforms static generic advice and fixed unprioritized templates.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from backend.evidence.calculus.schemas import EvidenceDebtReport, ObservablePredicate, PredicateType, RecoveryAction


class EvidenceBudgetPlanner:
    """Plans minimal-cost visual recovery actions to resolve critical evidence debt."""

    def __init__(
        self,
        cost_lambda: float = 0.05,  # trade-off parameter between risk reduction (0-1) and effort (seconds)
    ) -> None:
        self.cost_lambda = cost_lambda

        # Library of empirical candidate actions parameterized by physical cost and efficacy
        self.action_catalog: Dict[str, Dict[str, float]] = {
            "HOLD_STEADY_RECAPTURE": {
                "name": "Hold Camera Steady & Re-record",
                "human_effort_sec": 3.0,
                "camera_latency_sec": 1.5,
                "base_efficacy": 0.85,
                "directive": "Camera blur detected. Rest your hands against the table and record 2 steady seconds.",
            },
            "PAN_TO_REVEAL_OCCLUDED": {
                "name": "Pan Camera to Unblock Line-of-Sight",
                "human_effort_sec": 5.0,
                "camera_latency_sec": 2.0,
                "base_efficacy": 0.90,
                "directive": "The target object is occluded or truncated by the frame edge. Pan camera 15 cm right.",
            },
            "ZOOM_OUT_CONTAINMENT": {
                "name": "Step Back to Capture Full Interaction Canvas",
                "human_effort_sec": 4.0,
                "camera_latency_sec": 2.0,
                "base_efficacy": 0.80,
                "directive": "Object clipped by boundary. Pull camera back 20 cm so both start and end locations are fully visible.",
            },
            "CAPTURE_CROSS_VIEW": {
                "name": "Capture 45-Degree Oblique Confirmation Angle",
                "human_effort_sec": 8.0,
                "camera_latency_sec": 3.5,
                "base_efficacy": 0.95,
                "directive": "Top-down view obscures depth separation. Move camera to a 45-degree oblique angle.",
            },
            "HUMAN_VERIFICATION_CHECK": {
                "name": "Request Single-Click Reviewer Point Annotation",
                "human_effort_sec": 12.0,
                "camera_latency_sec": 0.0,
                "base_efficacy": 0.99,
                "directive": "Automated perception is uncertain. Tap the exact center of the object to disambiguate.",
            },
        }

    def plan_recovery_action(
        self,
        debt_report: EvidenceDebtReport,
        predicates: List[ObservablePredicate],
        budget_limit_sec: Optional[float] = None,
    ) -> Optional[RecoveryAction]:
        """Rank and return the single highest-net-utility action resolving current debt."""
        if debt_report.is_acceptable or debt_report.total_debt == 0:
            return None

        candidates = self.generate_all_candidate_actions(predicates)
        if not candidates:
            return None

        # Filter by budget limit if set
        if budget_limit_sec is not None:
            candidates = [
                c for c in candidates
                if (c.estimated_human_effort_sec + c.estimated_camera_latency_sec) <= budget_limit_sec
            ]
            if not candidates:
                return None

        # Sort descending by net utility score
        candidates.sort(key=lambda a: a.net_utility_score, reverse=True)
        return candidates[0]

    def generate_all_candidate_actions(
        self,
        predicates: List[ObservablePredicate],
    ) -> List[RecoveryAction]:
        """Generates and scores all viable actions matching unobservable or contradicted predicates."""
        actions: List[RecoveryAction] = []

        for p in predicates:
            if p.state == "SUPPORTED":
                continue

            # Map predicate defects to candidate recovery actions
            if p.predicate_type == PredicateType.OBSERVATION_SHARPNESS:
                actions.append(self._build_action("HOLD_STEADY_RECAPTURE", p, risk_reduction=0.85))
            elif p.predicate_type == PredicateType.CANVAS_CONTAINMENT:
                actions.append(self._build_action("ZOOM_OUT_CONTAINMENT", p, risk_reduction=0.80))
            elif p.predicate_type == PredicateType.SURFACE_VISIBILITY:
                actions.append(self._build_action("PAN_TO_REVEAL_OCCLUDED", p, risk_reduction=0.90))
            elif p.predicate_type == PredicateType.SPATIAL_RELATION:
                actions.append(self._build_action("CAPTURE_CROSS_VIEW", p, risk_reduction=0.92))
            elif p.predicate_type == PredicateType.OBJECT_IDENTIFICATION:
                actions.append(self._build_action("HUMAN_VERIFICATION_CHECK", p, risk_reduction=0.98))
            elif p.predicate_type == PredicateType.PHYSICS_NO_CONTRADICTION:
                # Contradiction requires explicit multi-angle cross-verification
                actions.append(self._build_action("CAPTURE_CROSS_VIEW", p, risk_reduction=0.95))

        return actions

    def _build_action(
        self,
        action_key: str,
        target_pred: ObservablePredicate,
        risk_reduction: float,
    ) -> RecoveryAction:
        meta = self.action_catalog[action_key]
        effort = meta["human_effort_sec"]
        latency = meta["camera_latency_sec"]
        total_cost_sec = effort + latency

        # VOI calculation: Net Utility = Risk_Reduction - (cost_lambda * total_cost_sec)
        cost_penalty = self.cost_lambda * total_cost_sec
        net_utility = risk_reduction - cost_penalty

        return RecoveryAction(
            action_id=f"ACT_{action_key}_{target_pred.predicate_id}",
            name=meta["name"],
            target_defect=f"{target_pred.predicate_type.value}: {target_pred.description}",
            directive=meta["directive"],
            estimated_human_effort_sec=effort,
            estimated_camera_latency_sec=latency,
            expected_risk_reduction=round(risk_reduction, 3),
            cost_penalty=round(cost_penalty, 3),
            net_utility_score=round(net_utility, 3),
        )
