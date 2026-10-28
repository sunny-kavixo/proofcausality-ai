"""Evidence Calculus Engine package.

Provides typed schemas, predicate decomposition, Evidence Debt scoring,
directed dependency graphs, and cost-aware budget planning.
"""

from backend.evidence.calculus.schemas import (
    EvidenceDebtReport,
    EvidenceState,
    GraphNode,
    ObservablePredicate,
    PredicateType,
    RecoveryAction,
    TaskSpecification,
    AlternativeWorldHypothesis,
)
from backend.evidence.calculus.engine import EvidenceCalculusEngine
from backend.evidence.calculus.graph import EvidenceDependencyGraph, build_canonical_task_graph
from backend.evidence.calculus.budget_planner import EvidenceBudgetPlanner
from backend.evidence.calculus.counterfactual import CounterfactualConsistencyChecker
from backend.evidence.calculus.independence import DetectorIndependenceAuditor

__all__ = [
    "EvidenceDebtReport",
    "EvidenceState",
    "GraphNode",
    "ObservablePredicate",
    "PredicateType",
    "RecoveryAction",
    "TaskSpecification",
    "AlternativeWorldHypothesis",
    "EvidenceCalculusEngine",
    "EvidenceDependencyGraph",
    "build_canonical_task_graph",
    "EvidenceBudgetPlanner",
    "CounterfactualConsistencyChecker",
    "DetectorIndependenceAuditor",
]
