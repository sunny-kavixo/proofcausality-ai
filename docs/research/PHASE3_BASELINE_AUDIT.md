# Phase 3 Baseline Audit & Integrity Investigation

**Project**: ProofCausality AI  
**Audit Date**: October 2026  
**Auditor**: Independent Principal AI Research Scientist & Software Auditor  
**Baseline Release**: `v0.2.0` (`commit 681bb47`)  
**Development Branch**: `fix/scientific-validation-v021`  

---

## 1. Baseline Git State & Remote Integrity

- **Main Commit SHA**: `681bb475ec2504620f3d61a0b3365fa375cb47be`
- **Tags Preserved**:
  - `v0.1.0`: Initial prototype release
  - `v0.1.1`: Rust native acceleration release
  - `v0.2.0`: Phase 2 Evidence Calculus architecture release
- **Remotes**: `origin` points to `https://github.com/sunny-kavixo/proofcausality-ai.git` (push & fetch active)
- **CI Workflow Status**: GitHub Actions CI workflow `ProofCausality AI CI` is passing on branch `main`.

---

## 2. Test Suite & Benchmark Execution Audit

- **Executed Command**: `python3 -m pytest`
- **Result**: 82 passed, 1 warning (Starlette deprecation notice for httpx testclient) in 0.44 seconds.
- **Test Composition**:
  - Unit tests for ActionProof, LabelCourt, EvidenceCore, ReCapture: 35 tests
  - API endpoints and image upload: 11 tests
  - Rust SIMD and image processing: 9 tests
  - Evidence Calculus schemas, engine, graph, and planner: 23 tests
  - Experiment execution pipeline: 4 tests

---

## 3. Forensic Code & Experiment Investigation

A forensic code audit of `backend/evidence/calculus/` and `evaluation/experiments/` revealed four critical engineering defects and three methodological evaluation biases:

### A. Critical Engineering Defects in `backend/evidence/calculus/engine.py`

1. **Configurable Displacement Threshold Ignored (Issue A)**:
   - `build_standard_relocation_spec` accepts `distance_threshold_px` (default 60.0px), but `evaluate_task` hardcoded `dist >= 30.0` at line 173.
   - Any configured threshold other than 30.0px was silently discarded during spatial predicate evaluation.
2. **Misleading "Proof of Physical Impossibility" Claim (Issue B)**:
   - `P_PHYS` evaluated `dist > self.max_physical_speed` (500px) and reported `"Physically impossible jump"`.
   - Without camera focal length, extrinsic calibration, 3D metric depth, or timestamp delta ($\Delta t$), a 2D pixel shift cannot mathematically prove physical impossibility. It is merely a 2D heuristic anomaly.
3. **Unsubstantiated Object Identity Assertion (Issue C)**:
   - `P_IDENT` checked only that `before_bbox` and `after_bbox` were non-null, declaring `"Coordinates successfully tracked in both temporal frames"`.
   - Coordinates establish bounding box presence, not semantic re-identification or object instance identity.
4. **Unqualified Final Resting State Assertion (Issue D)**:
   - `P_FINAL` accepted any final frame where `after_bbox` was present, ignoring whether the object was still moving or in-flight (motion blurred).

### B. Methodological Biases in Historical Experiments

1. **H01 Confidence-Only Baseline Bias**:
   - `exp_h01_evidence_debt.py` manually hardcoded `"mock_detector_confidence": 0.92` (real cases) and `0.94` (synthetic cases) across all cases, guaranteeing that the confidence baseline would fail on every true negative.
   - This fabricated confidence score created an artificially disadvantaged baseline rather than evaluating genuine confidence scores or calibrated geometric heuristics.
2. **H03 Simulated ReCapture Assumptions**:
   - `exp_h03_budget_planner.py` used `random.random() < 0.35` and `random.random() < 0.90` draws based on assumed probability distributions.
   - This tested a Monte Carlo simulation of assumed human behavior, not empirically observed human re-capture.
3. **H04 Synthetic Correlation in Multi-Model Arbitration**:
   - `exp_h04_detector_independence.py` manually supplied synthetic error vectors and predetermined correlation matrix ($ho = 0.88$).
   - This verified that the mathematical formula for dependence-aware discounting functions as written, but did not measure actual error correlation across real vision foundation models on real images.
4. **Dataset Media vs Metadata Provenance**:
   - The 12 tabletop manipulation cases in `evaluation/cases/real_dataset.py` are curated geometric metadata specifications (bounding boxes, blur flags, dimensions), not raw photographic JPEG/PNG files stored in the repository. Calling them "photographs" without bundled image files is inaccurate.

---

## 4. Planned Remediation Actions for Release v0.2.1

1. Correct `engine.py` to use dynamic, validated displacement thresholds, reframe physical claims as heuristic anomalies, qualify object identity, and require sharp final frames for rest state.
2. Implement fair baselines in H01 (Center-Distance Heuristic and Score-Thresholded Risk-Coverage Analysis) with 95% Wilson confidence intervals.
3. Label H03 and H04 as controlled simulations, publish the real re-capture evaluation protocol, and state that live multi-session human trials are not yet measured.
4. Bundle real sample image fixtures in `tests/fixtures/images/` and document dataset provenance honestly in `docs/research/DATASET_PROVENANCE_AUDIT.md`.
5. Update all documentation to remove overconfident claims of 100% categorical superiority.
