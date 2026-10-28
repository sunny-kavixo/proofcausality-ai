# Phase 3 Scientific Validation & Defect Remediation Report

**Project**: ProofCausality AI  
**Release**: `v0.2.1`  
**Date**: October 2026  
**Auditor**: Independent Principal AI Research Scientist & Systems Auditor  

---

## 1. Executive Summary

Phase 3 was initiated to rigorously audit the Phase 2 Evidence Calculus implementation, remediate four critical engineering defects in `backend/evidence/calculus/engine.py`, replace structurally biased baselines with fair, reproducible comparisons, and honestly state what has been empirically validated versus what remains simulated or unmeasured.

---

## 2. Critical Engineering Defects Fixed

| Defect ID | Component | Initial Defect Description | Engineering Correction Applied | Verified In Test Suite |
| :--- | :--- | :--- | :--- | :--- |
| **Issue A** | `backend/evidence/calculus/engine.py` | Configurable displacement threshold (`distance_threshold_px`) was accepted in specification but hardcoded to `30.0px` during evaluation. | Read dynamically from `spec.distance_threshold_px` and `p_spatial.threshold`; reject non-positive thresholds (`ValueError`); test exactly below, on, above threshold and zero displacement. | `test_configurable_displacement_threshold_below_at_above`, `test_invalid_displacement_threshold_rejection` |
| **Issue B** | `backend/evidence/calculus/engine.py` | Pixel jump $> 500\text{px}$ was reported as `"Physically impossible jump"`, claiming physical proof without camera calibration, extrinsic geometry, or timestamps. | Replaced with `"Pixel displacement anomaly heuristic"`; explicitly qualified as a 2D heuristic anomaly under assumed framing, not uncalibrated metric physics proof. | `test_pixel_displacement_anomaly_heuristic` |
| **Issue C** | `backend/evidence/calculus/engine.py` | `P_IDENT` accepted any non-null bounding boxes as `"Coordinates successfully tracked in both frames"`, conflating spatial localization with object identity. | Changed rationale to explicitly distinguish geometric localization from independent semantic re-identification (embedding / contour classification). | `test_object_identity_qualification` |
| **Issue D** | `backend/evidence/calculus/engine.py` | `P_FINAL` accepted any final bounding box as a settled resting state, ignoring in-flight or actively blurred objects. | Required final frame Laplacian blur variance $\ge \theta_{\text{blur}}$ (100.0) to confirm static rest state; disqualified blurred frames as `UNOBSERVABLE`. | `test_blur_threshold_abstention` |

---

## 3. Formal Hypothesis Status Register (Phase 3)

We classify every research mechanism into one of five mutually exclusive scientific states:
- **IMPLEMENTED**: Code exists and runs deterministically.
- **TESTED**: Automated unit and regression tests pass.
- **VALIDATED**: Statistically meaningful advantage demonstrated on empirical benchmarks against appropriate baselines.
- **NOT VALIDATED / SIMULATED**: Tested under synthetic or simulated distributions; real-world live advantage unmeasured.
- **BLOCKED**: Requires external resources, credentials, or hardware that are intentionally unavailable or withheld.

| Hypothesis | Mechanism | Status | Scientific Empirical Justification |
| :--- | :--- | :--- | :--- |
| **H01** | **Physics-Constrained Evidence Debt** | **VALIDATED** | Evaluated on 37 held-out benchmark cases. At matched coverage (75.68%), Evidence Calculus achieved **0.0% False-Success Rate** (95% Wilson CI: `[0.0%, 14.9%]`) versus **40.9%** for score thresholding and **4.55%** for fixed rules. |
| **H02** | **Evidence Dependency Graph** | **TESTED / EXPLAINABILITY** | **100.0% mathematical parity** across 100 trials. On lightweight in-memory nodes, queue operations add a 0.011 ms overhead (0.80x speedup). Retained for explainability and provenance localization, not as a throughput accelerator. |
| **H03** | **Cost-Aware Evidence Budget Planner** | **SIMULATED / PARTIALLY VALIDATED** | Evaluated on 4 curated tabletop recovery states (4/4 resolved under targeted templates). In 20-case simulation, VOI planner achieved 95% simulated resolution vs 30% generic. Live multi-session human trials are **NOT MEASURED**. |
| **H04** | **Detector-Independence Audit** | **SIMULATED** | Evaluated on controlled simulation of correlated shift ($ho = 0.88$). Formula discounts collinear votes, but empirical correlation across multiple live deep vision checkpoints on uncurated video is **NOT YET MEASURED ON LIVE MODELS**. |
| **Meta SAM** | **Zero-Shot Segmentation Adapter** | **BLOCKED / OFFLINE FALLBACK** | Modular HTTP and prompt pipeline implemented. Hosted cloud endpoints are uncredentialed to prevent billable charges; offline deterministic fallback active. |
