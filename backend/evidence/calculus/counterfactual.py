"""Minimal Consistent-World Counterexamples (High-Risk Research).

Explores alternative physical hypotheses consistent with observed pixels.
If an unobserved temporal gap exists (e.g. hand occludes object for 15 frames),
can we distinguish genuine object transfer from an object swap or tracking collapse?

Formulation:
If multiple distinct physical trajectories {H_0: genuine_transfer, H_1: object_swap, H_2: tracking_lost}
are all strictly consistent with visible evidence E, then the task is underdetermined.
The system MUST abstain (UNCERTAIN) rather than asserting SUCCESS.
"""

from __future__ import annotations

from typing import List
from backend.evidence.calculus.schemas import AlternativeWorldHypothesis


class CounterfactualConsistencyChecker:
    """Evaluates whether alternative non-completion worlds are consistent with evidence."""

    def __init__(self, max_occlusion_gap_frames: int = 5) -> None:
        self.max_occlusion_gap = max_occlusion_gap_frames

    def analyze_temporal_gap(
        self,
        observed_gap_frames: int,
        bounding_box_displacement_px: float,
        has_continuous_motion_trail: bool,
    ) -> List[AlternativeWorldHypothesis]:
        """Returns candidate world hypotheses consistent with the observed visual evidence."""
        hypotheses: List[AlternativeWorldHypothesis] = []

        # Hypothesis 0: Intended genuine physical transfer
        hypotheses.append(
            AlternativeWorldHypothesis(
                hypothesis_id="H_GENUINE_COMPLETION",
                description="Object was moved along an unobserved trajectory to the target location.",
                is_consistent_with_visible_evidence=True,
                task_completed_under_hypothesis=True,
                plausibility_score=0.90 if has_continuous_motion_trail else 0.50,
            )
        )

        # Hypothesis 1: Object Swap / Replacement during occlusion
        swap_consistent = observed_gap_frames > self.max_occlusion_gap and not has_continuous_motion_trail
        hypotheses.append(
            AlternativeWorldHypothesis(
                hypothesis_id="H_OBJECT_SWAP",
                description="Primary object was removed and a second object was placed into the scene during occlusion.",
                is_consistent_with_visible_evidence=swap_consistent,
                task_completed_under_hypothesis=False,
                plausibility_score=0.40 if swap_consistent else 0.05,
            )
        )

        # Hypothesis 2: Teleportation / False Association (Tracker Failure)
        teleport_consistent = bounding_box_displacement_px > 300.0 and observed_gap_frames <= 2
        hypotheses.append(
            AlternativeWorldHypothesis(
                hypothesis_id="H_TRACKER_DRIFT",
                description="Perception identity switched to a background distracter with similar appearance.",
                is_consistent_with_visible_evidence=teleport_consistent,
                task_completed_under_hypothesis=False,
                plausibility_score=0.85 if teleport_consistent else 0.02,
            )
        )

        return hypotheses

    def is_decision_underdetermined(self, hypotheses: List[AlternativeWorldHypothesis]) -> bool:
        """True if at least one failure hypothesis is consistent and reasonably plausible."""
        for h in hypotheses:
            if not h.task_completed_under_hypothesis and h.is_consistent_with_visible_evidence:
                if h.plausibility_score >= 0.30:
                    return True
        return False
