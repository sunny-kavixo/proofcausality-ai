"""Typed Pydantic schemas for the Evidence Calculus Engine.

Formalizes:
- Tri-state predicate evidence values: SUPPORTED, CONTRADICTED, UNOBSERVABLE.
- Typed observable predicates with physical and perceptual preconditions.
- Physics-constrained Evidence Debt formulations: D(E, T).
- Directed Evidence Dependency Graph node specifications.
- Cost-aware Value-of-Information (VOI) recovery actions.
- Minimal consistent-world alternative hypothesis representation.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field

from backend.evidence.schemas import ActionVerdict, BoundingBox


class EvidenceState(str, Enum):
    """Tri-state evidence condition evaluation outcome."""
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNOBSERVABLE = "UNOBSERVABLE"


class PredicateType(str, Enum):
    """Categorical classification of observable conditions in physical verification."""
    OBJECT_IDENTIFICATION = "OBJECT_IDENTIFICATION"
    INITIAL_STATE_OBSERVED = "INITIAL_STATE_OBSERVED"
    FINAL_STATE_OBSERVED = "FINAL_STATE_OBSERVED"
    SPATIAL_RELATION = "SPATIAL_RELATION"
    OBSERVATION_SHARPNESS = "OBSERVATION_SHARPNESS"
    CANVAS_CONTAINMENT = "CANVAS_CONTAINMENT"
    SURFACE_VISIBILITY = "SURFACE_VISIBILITY"
    TEMPORAL_CONTINUITY = "TEMPORAL_CONTINUITY"
    PHYSICS_NO_CONTRADICTION = "PHYSICS_NO_CONTRADICTION"


class ObservablePredicate(BaseModel):
    """An individual verifiable condition with evidence state and diagnostic rationale."""
    predicate_id: str
    predicate_type: PredicateType
    description: str
    is_critical: bool = True
    weight: float = 1.0
    state: EvidenceState = EvidenceState.UNOBSERVABLE
    measured_value: Optional[float] = None
    threshold: Optional[float] = None
    rationale: str = ""
    evidence_source: str = "OBSERVED_PIXELS"
    dependencies: List[str] = Field(default_factory=list)


class TaskSpecification(BaseModel):
    """High-level physical task specification decomposed into observable predicates."""
    task_id: str
    task_type: str
    primary_object: str
    reference_object: Optional[str] = None
    target_region: Optional[BoundingBox] = None
    initial_region: Optional[BoundingBox] = None
    predicates: List[ObservablePredicate] = Field(default_factory=list)
    max_tolerable_debt: float = 0.0


class EvidenceDebtReport(BaseModel):
    """Quantified breakdown of unresolved, contradictory, and unobservable evidence."""
    total_debt: float
    missing_evidence_debt: float
    contradictory_evidence_debt: float
    unreliability_debt: float
    is_acceptable: bool
    critical_unresolved_predicates: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    formula: str = "D(E,T) = sum_{c in critical} w_c * deficit(c)"


class GraphNode(BaseModel):
    """Node in the Evidence Dependency Graph."""
    node_id: str
    node_type: str  # "OBSERVATION", "DETECTION", "MASK", "MEASUREMENT", "PREDICATE", "VERDICT"
    label: str
    parents: List[str] = Field(default_factory=list)
    children: List[str] = Field(default_factory=list)
    is_stale: bool = False
    value: Any = None
    cached_timestamp: Optional[float] = None


class RecoveryAction(BaseModel):
    """Candidate corrective action evaluated under the Evidence Budget Planner."""
    action_id: str
    name: str
    target_defect: str
    directive: str
    estimated_human_effort_sec: float
    estimated_camera_latency_sec: float
    expected_risk_reduction: float
    net_utility_score: float  # VOI = Expected_Risk_Reduction - lambda * Cost
    cost_penalty: float


class AlternativeWorldHypothesis(BaseModel):
    """Candidate alternative explanation consistent with visible evidence."""
    hypothesis_id: str
    description: str
    is_consistent_with_visible_evidence: bool
    task_completed_under_hypothesis: bool
    plausibility_score: float
