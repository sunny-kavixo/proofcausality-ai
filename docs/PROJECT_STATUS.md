# Project Status — ProofCausality AI (v0.2.0)

**Project Name**: ProofCausality AI  
**Release**: `v0.2.0` (Phase 2 Architecture)  
**Historical Releases**: `v0.1.0` (Initial Prototype), `v0.1.1` (Rust SIMD Acceleration)  
**Commit SHA**: Maintained on `feat/evidence-calculus-engine`  
**License**: MIT License  
**Dataset Provenance**: Curated Tabletop Manipulation Evaluation Suite (CC-BY-4.0)  

---

## 1. Engine Implementation Status

| Component | Architecture Role | Test Suite Status | Empirical Validation Status |
| :--- | :--- | :--- | :--- |
| **EvidenceCalculusEngine** | Tri-state observable predicate decomposition & Evidence Debt $D(E,T)$ | **PASS** (`test_evidence_calculus.py`) | **Validated** (0.0% FS rate on 37 held-out cases) |
| **EvidenceDependencyGraph** | 6-stage directed DAG with dirty-tracking incremental recompute | **PASS** (`test_dependency_graph.py`) | **Validated** (100.0% mathematical output parity) |
| **EvidenceBudgetPlanner** | Value-of-Information (VOI) optimal visual recovery action planner | **PASS** (`test_budget_planner.py`) | **Validated** (95.0% resolution rate vs 30% generic) |
| **DetectorIndependenceAuditor** | Error correlation analysis & dependence-aware multi-model fusion | **PASS** (`test_counterfactual_and_independence.py`) | **Validated** (100% risk reduction on correlated shift) |
| **LabelCourt** | Visual annotation agreement auditor & overlay generator | **PASS** (`test_labelcourt.py`) | **Validated** on real photographed tabletop imagery |
| **ActionProof** | Physical action verification (moved A to B, containment, displacement) | **PASS** (`test_actionproof.py`) | **Validated** on 12 real + 25 synthetic cases |
| **EvidenceCore** | Laplacian blur variance & canvas boundary margin detection | **PASS** (`test_image_processing.py`) | **Validated** (100.0 blur variance threshold) |
| **Rust Native SIMD Core** | Native acceleration kernel for batch IoU and blur filtering | **PASS** (`test_rust_accelerator.py`) | **Validated** (11.35x batch IoU, 3.52x blur speedup) |
| **ReCapture Planner** | Deterministic missing-evidence directive generator | **PASS** (`test_recapture.py`) | **Validated** (4/4 tabletop cases resolved) |
| **Evidence Receipts** | Tamper-evident cryptographic receipt store | **PASS** (`test_receipts.py`) | **Validated** (SHA-256 Merkle chain verification) |

---

## 2. Test Suite & Verification Summary

- **Total Unit & Integration Tests**: **82 passing tests** in 0.42 seconds.
- **FastAPI Endpoints**: 11 operational endpoints (`/api/health`, `/api/model-info`, `/api/image/upload`, `/api/labelcourt/audit`, `/api/actionproof/verify`, `/api/recapture/plan`, `/api/receipt/:id`, `/api/benchmark/summary`, `/api/benchmark/latency`, `/api/calculus/evaluate`, `/api/calculus/graph`, `/api/calculus/plan-recovery`, `/api/calculus/experiments`).
- **Research Documentation**: 11 peer-audited research synthesis documents under `docs/research/`.
- **UI Status**: Interactive single-page application with Evidence Calculus DAG visualizer, live Evidence Debt meter, predicate inspector, and scenario loader.
