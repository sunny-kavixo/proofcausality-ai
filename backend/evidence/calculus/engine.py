"""Evidence Calculus Engine for ProofCausality AI.

Implements:
- Explicit decomposition of physical tasks into observable predicates.
- Physics-constrained Evidence Debt calculation: D(E, T).
- Three-state condition evaluation: SUPPORTED, CONTRADICTED, UNOBSERVABLE.
- Physical plausibility and non-teleportation constraints.
- Grounded tri-state decision arbitration (SUCCESS, FAILED, UNCERTAIN).
"""

from __future__ import annotations

import math
from typing import Any, List, Optional
import numpy as np

from backend.actionproof.relations import compute_centroid_distance, is_contained_in_region
from backend.evidence.calculus.schemas import (
    EvidenceDebtReport,
    EvidenceState,
    ObservablePredicate,
    PredicateType,
    TaskSpecification,
)
from backend.evidence.masks import mask_to_bbox
from backend.evidence.schemas import ActionVerdict, BoundingBox


class EvidenceCalculusEngine:
    """Evaluates task-specific observational predicates and calculates Evidence Debt."""

    def __init__(
        self,
        blur_threshold: float = 100.0,
        boundary_margin_px: int = 3,
        occlusion_area_drop_threshold: float = 0.35,
        max_physical_speed_px_per_frame: float = 500.0,
    ) -> None:
        self.blur_threshold = blur_threshold
        self.boundary_margin_px = boundary_margin_px
        self.occlusion_drop_threshold = occlusion_area_drop_threshold
        self.max_physical_speed = max_physical_speed_px_per_frame

    def build_standard_relocation_spec(
        self,
        task_id: str,
        primary_object: str,
        initial_region: BoundingBox,
        target_region: BoundingBox,
        distance_threshold_px: float = 60.0,
    ) -> TaskSpecification:
        """Create standard typed specification for object relocation tasks (moved A to B)."""
        predicates = [
            ObservablePredicate(
                predicate_id=f"{task_id}_P_IDENT",
                predicate_type=PredicateType.OBJECT_IDENTIFICATION,
                description=f"Primary object '{primary_object}' observed with valid coordinates",
                is_critical=True,
                weight=1.0,
            ),
            ObservablePredicate(
                predicate_id=f"{task_id}_P_INIT",
                predicate_type=PredicateType.INITIAL_STATE_OBSERVED,
                description=f"Initial state observed within start region A",
                is_critical=True,
                weight=1.0,
            ),
            ObservablePredicate(
                predicate_id=f"{task_id}_P_FINAL",
                predicate_type=PredicateType.FINAL_STATE_OBSERVED,
                description=f"Final resting state observable after action conclusion",
                is_critical=True,
                weight=1.5,
            ),
            ObservablePredicate(
                predicate_id=f"{task_id}_P_SPATIAL",
                predicate_type=PredicateType.SPATIAL_RELATION,
                description=f"Final state satisfies target region B containment and minimum displacement >= {distance_threshold_px}px",
                is_critical=True,
                weight=2.0,
            ),
            ObservablePredicate(
                predicate_id=f"{task_id}_P_SHARP",
                predicate_type=PredicateType.OBSERVATION_SHARPNESS,
                description=f"Visual sharpness above blur threshold ({self.blur_threshold} var)",
                is_critical=True,
                weight=1.0,
            ),
            ObservablePredicate(
                predicate_id=f"{task_id}_P_FRAME",
                predicate_type=PredicateType.CANVAS_CONTAINMENT,
                description=f"Object maintains >= {self.boundary_margin_px}px margin from canvas borders",
                is_critical=True,
                weight=1.0,
            ),
            ObservablePredicate(
                predicate_id=f"{task_id}_P_OCCL",
                predicate_type=PredicateType.SURFACE_VISIBILITY,
                description=f"Surface area retention within unoccluded bounds (< {self.occlusion_drop_threshold:.0%} drop)",
                is_critical=True,
                weight=1.0,
            ),
            ObservablePredicate(
                predicate_id=f"{task_id}_P_PHYS",
                predicate_type=PredicateType.PHYSICS_NO_CONTRADICTION,
                description=f"Transition satisfies physical plausibility (no instantaneous teleportation)",
                is_critical=True,
                weight=2.0,
            ),
        ]

        return TaskSpecification(
            task_id=task_id,
            task_type="moved_A_to_B",
            primary_object=primary_object,
            initial_region=initial_region,
            target_region=target_region,
            predicates=predicates,
            max_tolerable_debt=0.0,
        )

    def evaluate_task(
        self,
        spec: TaskSpecification,
        before_bbox: Optional[BoundingBox],
        after_bbox: Optional[BoundingBox],
        canvas_width: int,
        canvas_height: int,
        laplacian_variance: float = 250.0,
        area_drop_ratio: float = 0.0,
        has_final_state: bool = True,
    ) -> tuple[ActionVerdict, EvidenceDebtReport, TaskSpecification]:
        """Evaluate each observable predicate, compute Evidence Debt, and arbitrate verdict."""
        evaluated_predicates: List[ObservablePredicate] = []

        # 1. P_IDENT: Object identification
        p_ident = self._find_predicate(spec, PredicateType.OBJECT_IDENTIFICATION)
        if before_bbox is not None and after_bbox is not None:
            p_ident.state = EvidenceState.SUPPORTED
            p_ident.rationale = "Coordinates successfully tracked in both temporal frames."
        else:
            p_ident.state = EvidenceState.UNOBSERVABLE
            p_ident.rationale = "Object coordinates missing in one or both frames."
        evaluated_predicates.append(p_ident)

        # 2. P_INIT: Initial containment in Region A
        p_init = self._find_predicate(spec, PredicateType.INITIAL_STATE_OBSERVED)
        if before_bbox and spec.initial_region:
            in_a = is_contained_in_region(before_bbox, spec.initial_region)
            p_init.state = EvidenceState.SUPPORTED if in_a else EvidenceState.CONTRADICTED
            p_init.rationale = "Initial state inside Region A." if in_a else "Initial state outside start Region A."
        else:
            p_init.state = EvidenceState.UNOBSERVABLE
            p_init.rationale = "Initial region or bounding box unobservable."
        evaluated_predicates.append(p_init)

        # 3. P_FINAL: Final rest state observation
        p_final = self._find_predicate(spec, PredicateType.FINAL_STATE_OBSERVED)
        if has_final_state and after_bbox is not None:
            p_final.state = EvidenceState.SUPPORTED
            p_final.rationale = "Final resting state observed."
        else:
            p_final.state = EvidenceState.UNOBSERVABLE
            p_final.rationale = "Recording terminated before final rest state was established."
        evaluated_predicates.append(p_final)

        # 4. P_SPATIAL: Target region containment and minimum displacement
        p_spatial = self._find_predicate(spec, PredicateType.SPATIAL_RELATION)
        if before_bbox and after_bbox and spec.target_region:
            dist = compute_centroid_distance(before_bbox, after_bbox)
            in_b = is_contained_in_region(after_bbox, spec.target_region)
            p_spatial.measured_value = round(dist, 2)
            if in_b and dist >= 30.0:
                p_spatial.state = EvidenceState.SUPPORTED
                p_spatial.rationale = f"Contained in Region B with {dist:.1f}px displacement."
            elif not in_b:
                p_spatial.state = EvidenceState.CONTRADICTED
                p_spatial.rationale = f"Object placed outside Region B (displacement: {dist:.1f}px)."
            else:
                p_spatial.state = EvidenceState.CONTRADICTED
                p_spatial.rationale = f"Displacement {dist:.1f}px insufficient for relocation."
        else:
            p_spatial.state = EvidenceState.UNOBSERVABLE
            p_spatial.rationale = "Target region or object bounds unobservable."
        evaluated_predicates.append(p_spatial)

        # 5. P_SHARP: Observation sharpness
        p_sharp = self._find_predicate(spec, PredicateType.OBSERVATION_SHARPNESS)
        p_sharp.measured_value = round(laplacian_variance, 2)
        p_sharp.threshold = self.blur_threshold
        if laplacian_variance >= self.blur_threshold:
            p_sharp.state = EvidenceState.SUPPORTED
            p_sharp.rationale = f"Laplacian variance {laplacian_variance:.1f} >= {self.blur_threshold} (sharp)."
        else:
            p_sharp.state = EvidenceState.UNOBSERVABLE
            p_sharp.rationale = f"Motion blur detected: variance {laplacian_variance:.1f} < {self.blur_threshold}."
        evaluated_predicates.append(p_sharp)

        # 6. P_FRAME: Canvas containment (not clipped by borders)
        p_frame = self._find_predicate(spec, PredicateType.CANVAS_CONTAINMENT)
        if after_bbox:
            touches = (
                after_bbox.x_min <= self.boundary_margin_px
                or after_bbox.y_min <= self.boundary_margin_px
                or after_bbox.x_max >= (canvas_width - self.boundary_margin_px)
                or after_bbox.y_max >= (canvas_height - self.boundary_margin_px)
            )
            if touches:
                p_frame.state = EvidenceState.UNOBSERVABLE
                p_frame.rationale = "Object contacts frame edge: partial boundary unobservable."
            else:
                p_frame.state = EvidenceState.SUPPORTED
                p_frame.rationale = "Full object boundary contained within canvas margins."
        else:
            p_frame.state = EvidenceState.UNOBSERVABLE
            p_frame.rationale = "Bounding box unobservable."
        evaluated_predicates.append(p_frame)

        # 7. P_OCCL: Surface visibility
        p_occl = self._find_predicate(spec, PredicateType.SURFACE_VISIBILITY)
        p_occl.measured_value = round(area_drop_ratio, 3)
        p_occl.threshold = self.occlusion_drop_threshold
        if area_drop_ratio >= self.occlusion_drop_threshold:
            p_occl.state = EvidenceState.UNOBSERVABLE
            p_occl.rationale = f"Severe occlusion: {area_drop_ratio:.0%} area drop exceeds {self.occlusion_drop_threshold:.0%} threshold."
        else:
            p_occl.state = EvidenceState.SUPPORTED
            p_occl.rationale = f"Surface area intact ({area_drop_ratio:.1%} occlusion drop)."
        evaluated_predicates.append(p_occl)

        # 8. P_PHYS: Physics contradiction constraint
        p_phys = self._find_predicate(spec, PredicateType.PHYSICS_NO_CONTRADICTION)
        if before_bbox and after_bbox:
            dist = compute_centroid_distance(before_bbox, after_bbox)
            if dist > self.max_physical_speed:
                p_phys.state = EvidenceState.CONTRADICTED
                p_phys.rationale = f"Physically impossible jump: {dist:.1f}px exceeds speed limit {self.max_physical_speed}px."
            else:
                p_phys.state = EvidenceState.SUPPORTED
                p_phys.rationale = f"Physical motion magnitude {dist:.1f}px is physically plausible."
        else:
            p_phys.state = EvidenceState.UNOBSERVABLE
            p_phys.rationale = "Temporal trajectory unobservable."
        evaluated_predicates.append(p_phys)

        # Compute Evidence Debt: D(E, T)
        debt_report = self._calculate_evidence_debt(evaluated_predicates, spec.max_tolerable_debt)

        # Arbitrate final verdict
        verdict = self._arbitrate_verdict(evaluated_predicates, debt_report)

        updated_spec = spec.model_copy(update={"predicates": evaluated_predicates})
        return verdict, debt_report, updated_spec

    def _find_predicate(self, spec: TaskSpecification, pred_type: PredicateType) -> ObservablePredicate:
        for p in spec.predicates:
            if p.predicate_type == pred_type:
                return p.model_copy()
        return ObservablePredicate(
            predicate_id=f"PRED_{pred_type.value}",
            predicate_type=pred_type,
            description=f"Auto-generated {pred_type.value}",
        )

    def _calculate_evidence_debt(
        self,
        predicates: List[ObservablePredicate],
        max_tolerable_debt: float,
    ) -> EvidenceDebtReport:
        missing_debt = 0.0
        contra_debt = 0.0
        critical_unresolved: List[str] = []
        contradictions: List[str] = []

        for p in predicates:
            if not p.is_critical:
                continue

            if p.state == EvidenceState.UNOBSERVABLE:
                deficit = p.weight * 1.0
                missing_debt += deficit
                critical_unresolved.append(f"{p.predicate_type.value}: {p.rationale}")
            elif p.state == EvidenceState.CONTRADICTED:
                deficit = p.weight * 2.0  # Contradictions carry double penalty weight
                contra_debt += deficit
                contradictions.append(f"{p.predicate_type.value}: {p.rationale}")

        total_debt = missing_debt + contra_debt
        is_acceptable = (total_debt <= max_tolerable_debt) and (len(contradictions) == 0)

        return EvidenceDebtReport(
            total_debt=round(total_debt, 2),
            missing_evidence_debt=round(missing_debt, 2),
            contradictory_evidence_debt=round(contra_debt, 2),
            unreliability_debt=0.0,
            is_acceptable=is_acceptable,
            critical_unresolved_predicates=critical_unresolved,
            contradictions=contradictions,
        )

    def _arbitrate_verdict(
        self,
        predicates: List[ObservablePredicate],
        debt_report: EvidenceDebtReport,
    ) -> ActionVerdict:
        # Falsification priority: Any contradiction yields FAILED
        if len(debt_report.contradictions) > 0:
            return ActionVerdict.FAILED

        # Incompleteness: Any unresolved critical condition yields UNCERTAIN
        if debt_report.total_debt > 0.0 or not debt_report.is_acceptable:
            return ActionVerdict.UNCERTAIN

        # Full verifiable proof
        return ActionVerdict.SUCCESS
