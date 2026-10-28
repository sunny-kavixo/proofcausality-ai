# Formal Hypothesis Register & Research Gate Log

**Project**: ProofCausality AI — Phase 2  
**Date**: October 2026  
**Status**: Final Audit Record  

---

## 1. Primary Hypotheses (Gates 1 to 4)

| ID | Hypothesis Title | Primary Claim | Falsification Metric | Empirical Result | Gate Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **H01** | **Physics-Constrained Evidence Debt** | Representing missing ($w=1.0$) and contradictory ($w=2.0$) evidence predicts false-success errors better than confidence scoring. | False-Success Rate $\ge$ 5% or over-abstention on true successes. | **FS Rate: 0.0%** (vs 100% confidence baseline). Zero over-abstention. 0.026 ms latency. | **PASS (GO)** |
| **H02** | **Evidence Dependency Graph** | Directed graph enables incremental re-evaluation matching full pipeline output. | Parity match $< 100\%$ or non-explained overhead. | **Parity: 100.0%**. Skipped 4/14 nodes. Pure Python overhead 0.011 ms. | **PASS as Explainability / Parity (REVISE on pure speed)** |
| **H03** | **Cost-Aware Evidence Budget Planner** | Value-of-Information (VOI) optimization resolves uncertain cases with higher success and lower effort than static advice. | Resolution rate $\le$ fixed templates or excessive human effort. | **Resolution: 95.0%** (vs 85% templates, 30% generic) with 8.2s mean effort. | **PASS (GO)** |
| **H04** | **Detector-Independence Audit** | Dependence-aware discounting of correlated detectors prevents false consensus under distribution shifts. | False-success rate matches simple majority voting. | **FS Rate: 0.0%** (vs 100% majority vote on correlated shift). | **PASS (GO)** |

---

## 2. Experimental Backlog Decisions

| ID | Mechanism | Closest Prior Art | Estimated Cost | Feasibility / Failure Mode | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **B01** | **Minimal Consistent-World Counterexamples** | Symbolic Action Models (Fikes et al.) | Low (Analytical) | High risk of false certainty in unobserved physical state spaces. | **PROTOTYPE TESTED** |
| **B02** | **Temporal Logic Action Ordering** | LTL Action Monitors (Pnueli) | Moderate | High specification engineering burden for ad-hoc user tasks. | **REVISE** |
| **B03** | **Cross-View Multi-Camera Alignment** | Epipolar Geometry (Hartley & Zisserman) | High (Multi-cam hardware) | Requires synchronized multi-camera hardware not present on mobile/monocular setups. | **NO-GO (Hardware blocked)** |
| **B04** | **Annotation Influence Ranking** | Influence Functions (Koh & Liang, 2017) | Very High GPU compute | Computing inverse Hessian over foundation vision models exceeds hackathon resource budget. | **NO-GO (Resource limit)** |
| **B05** | **Temporal Proof Compression** | Video Keyframe Extraction | Low | Redundant with state-transition bounding box diffs. | **MERGED into EvidenceCore** |
