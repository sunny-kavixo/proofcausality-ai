"""Typed closed-loop evidence verification built on Evidence Calculus.

ProofLoop is deliberately a policy layer, not a second perception model.  It
turns the existing ActionProof/Evidence Calculus result into an explicit
observation-and-recovery state machine and preserves the legacy verdict
contract for callers that still consume ``SUCCESS``, ``FAILED`` or
``UNCERTAIN``.

The implementation accepts measured evidence features (coordinates, blur,
visibility and optional model confidence).  It does not claim that metadata
alone is an independently verified photograph or video observation; the API
records that provenance boundary in its receipt.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from backend.evidence.calculus.budget_planner import EvidenceBudgetPlanner
from backend.evidence.calculus.engine import EvidenceCalculusEngine
from backend.evidence.calculus.schemas import EvidenceDebtReport, ObservablePredicate
from backend.evidence.schemas import ActionVerdict, BoundingBox


class ProofLoopState(str, Enum):
    """Finite states exposed by the closed-loop verifier."""

    OBSERVED = "OBSERVED"
    ANALYZED = "ANALYZED"
    UNCERTAIN = "UNCERTAIN"
    OBSERVATION_REQUESTED = "OBSERVATION_REQUESTED"
    NEW_EVIDENCE = "NEW_EVIDENCE"
    REVERIFIED = "REVERIFIED"
    FINAL_VERDICT = "FINAL_VERDICT"


class ProofLoopVerdict(str, Enum):
    """Provider-independent semantic verdict for a claim."""

    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class ProofLoopObservation(BaseModel):
    """Measured evidence supplied to one ProofLoop evaluation step."""

    task_id: str = Field(..., min_length=1, max_length=128)
    primary_object: str = Field(..., min_length=1, max_length=128)
    before_bbox: BoundingBox
    after_bbox: Optional[BoundingBox] = None
    target_region: BoundingBox
    initial_region: Optional[BoundingBox] = None
    canvas_width: int = Field(640, ge=1, le=10000)
    canvas_height: int = Field(480, ge=1, le=10000)
    laplacian_variance: float = Field(250.0, ge=0.0)
    area_drop_ratio: float = Field(0.0, ge=0.0, le=1.0)
    has_final_state: bool = True
    require_rest_proof: bool = False
    # Optional independent inputs from an upstream detector or claim parser.
    # They are never silently synthesized by ProofLoop.
    claim_class_agreement: Optional[bool] = None
    model_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_geometry_and_finiteness(self) -> "ProofLoopObservation":
        boxes = {
            "before_bbox": self.before_bbox,
            "after_bbox": self.after_bbox,
            "target_region": self.target_region,
            "initial_region": self.initial_region,
        }
        for name, box in boxes.items():
            if box is None:
                continue
            values = (box.x_min, box.y_min, box.x_max, box.y_max)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} coordinates must be finite")
            if box.x_max <= box.x_min or box.y_max <= box.y_min:
                raise ValueError(f"{name} must have positive width and height")
        for name in ("laplacian_variance", "area_drop_ratio", "model_confidence"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        return self


class ProofLoopEvaluation(BaseModel):
    """Auditable result of one initial or follow-up evaluation."""

    phase: str
    state: ProofLoopState
    state_trace: list[ProofLoopState]
    legacy_action_verdict: ActionVerdict
    semantic_verdict: ProofLoopVerdict
    decision_basis: str
    observation_request: Optional[dict[str, Any]] = None
    debt_report: EvidenceDebtReport
    predicates: list[ObservablePredicate]
    measurements: dict[str, Any]


def semantic_verdict_for(action_verdict: ActionVerdict) -> ProofLoopVerdict:
    """Map the legacy ActionProof contract to explicit claim semantics."""

    return {
        ActionVerdict.SUCCESS: ProofLoopVerdict.SUPPORTED,
        ActionVerdict.FAILED: ProofLoopVerdict.CONTRADICTED,
        ActionVerdict.UNCERTAIN: ProofLoopVerdict.INSUFFICIENT_EVIDENCE,
    }[action_verdict]


class ProofLoopVerifier:
    """Closed-loop policy over the existing deterministic evidence engine."""

    def __init__(
        self,
        evidence_engine: Optional[EvidenceCalculusEngine] = None,
        recovery_planner: Optional[EvidenceBudgetPlanner] = None,
        model_confidence_threshold: float = 0.50,
    ) -> None:
        if not 0.0 <= model_confidence_threshold <= 1.0:
            raise ValueError("model_confidence_threshold must be between 0 and 1")
        self.engine = evidence_engine or EvidenceCalculusEngine()
        self.planner = recovery_planner or EvidenceBudgetPlanner()
        self.model_confidence_threshold = model_confidence_threshold

    def evaluate(
        self,
        observation: ProofLoopObservation,
        *,
        phase: str = "initial",
        previous_trace: Optional[list[ProofLoopState]] = None,
    ) -> ProofLoopEvaluation:
        """Evaluate one observation and produce the next state-machine step.

        ``phase`` is either ``initial`` or ``followup``.  A follow-up carries
        forward the initial trace and is always marked ``NEW_EVIDENCE`` before
        re-evaluation.  If the new evidence remains insufficient, the final
        state is still ``FINAL_VERDICT`` with an insufficient semantic verdict;
        this makes unresolved recovery an explicit, testable outcome.
        """

        if phase not in {"initial", "followup"}:
            raise ValueError("phase must be 'initial' or 'followup'")

        initial_region = observation.initial_region or observation.before_bbox
        spec = self.engine.build_standard_relocation_spec(
            task_id=observation.task_id,
            primary_object=observation.primary_object,
            initial_region=initial_region,
            target_region=observation.target_region,
            require_rest_proof=observation.require_rest_proof,
        )
        action_verdict, debt_report, updated_spec = self.engine.evaluate_task(
            spec=spec,
            before_bbox=observation.before_bbox,
            after_bbox=observation.after_bbox,
            canvas_width=observation.canvas_width,
            canvas_height=observation.canvas_height,
            laplacian_variance=observation.laplacian_variance,
            area_drop_ratio=observation.area_drop_ratio,
            has_final_state=observation.has_final_state,
        )

        decision_basis = "EVIDENCE_CALCULUS"
        if observation.claim_class_agreement is False:
            # A supplied class conflict falsifies the claim, even if geometry
            # happens to agree.  This is an explicit semantic override rather
            # than a change to the legacy ActionProof engine.
            action_verdict = ActionVerdict.FAILED
            decision_basis = "CLAIM_CLASS_MISMATCH"
        elif (
            observation.model_confidence is not None
            and observation.model_confidence < self.model_confidence_threshold
            and action_verdict == ActionVerdict.SUCCESS
        ):
            # Low detector confidence creates epistemic uncertainty, not a
            # contradiction.  The deterministic baseline remains untouched.
            action_verdict = ActionVerdict.UNCERTAIN
            decision_basis = "LOW_MODEL_CONFIDENCE"

        semantic_verdict = semantic_verdict_for(action_verdict)
        recovery = self.planner.plan_recovery_action(
            debt_report,
            updated_spec.predicates,
        ) if semantic_verdict == ProofLoopVerdict.INSUFFICIENT_EVIDENCE else None

        if phase == "initial":
            trace = [ProofLoopState.OBSERVED, ProofLoopState.ANALYZED]
            if semantic_verdict == ProofLoopVerdict.INSUFFICIENT_EVIDENCE:
                trace.extend([ProofLoopState.UNCERTAIN, ProofLoopState.OBSERVATION_REQUESTED])
            else:
                trace.append(ProofLoopState.FINAL_VERDICT)
        else:
            trace = list(previous_trace or [ProofLoopState.OBSERVED, ProofLoopState.ANALYZED])
            trace.extend([ProofLoopState.NEW_EVIDENCE, ProofLoopState.REVERIFIED, ProofLoopState.FINAL_VERDICT])

        measurements = {
            "task_id": observation.task_id,
            "primary_object": observation.primary_object,
            "phase": phase,
            "legacy_action_verdict": action_verdict.value,
            "semantic_verdict": semantic_verdict.value,
            "decision_basis": decision_basis,
            "laplacian_variance": round(observation.laplacian_variance, 3),
            "area_drop_ratio": round(observation.area_drop_ratio, 3),
            "has_final_state": observation.has_final_state,
            "claim_class_agreement": observation.claim_class_agreement,
            "model_confidence": observation.model_confidence,
            "evidence_debt": debt_report.total_debt,
            "structured_finding": updated_spec.structured_finding,
        }
        return ProofLoopEvaluation(
            phase=phase,
            state=trace[-1],
            state_trace=trace,
            legacy_action_verdict=action_verdict,
            semantic_verdict=semantic_verdict,
            decision_basis=decision_basis,
            observation_request=recovery.model_dump() if recovery else None,
            debt_report=debt_report,
            predicates=updated_spec.predicates,
            measurements=measurements,
        )
