# Consolidated Empirical Research Results (Phase 3 Verified)

**Project**: ProofCausality AI  
**Release**: `v0.2.1`  
**Date**: October 2026  
**Environment**: Apple M4 (macOS 15, ARM64), Python 3.14.5, Rust 1.99.0  

---

## 1. Experiment H01: Evidence Debt vs Fair Baselines (37 Cases)

- **Dataset**: 12 curated tabletop metadata pairs + 25 synthetic benchmark cases (37 total).
- **Ground Truth**: 15 True Successes, 22 True Negatives (14 Uncertain, 8 Failed).

| Verification Strategy | Decided Cases | Coverage | False-Successes (FS) | False-Success Rate | 95% Wilson Score CI | Selective Risk | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1. Naive Center-Distance Heuristic** | 37 / 37 | **100.0%** | 10 / 22 | **45.45%** | `[26.92%, 65.34%]` | 0.3514 | 0.001 ms |
| **B2. Score-Thresholded Proximity Predictor** | 36 / 37 | **97.30%** | 9 / 22 | **40.91%** | `[23.26%, 61.27%]` | 0.3333 | 0.001 ms |
| **B3. Fixed Cause-Specific Rules** | 28 / 37 | **75.68%** | 1 / 22 | **4.55%** | `[0.81%, 21.80%]` | 0.1429 | 0.000 ms |
| **M4. Evidence Debt (Unit Weight Only)** | 15 / 37 | **40.54%** | 0 / 22 | **0.00%** | `[0.00%, 14.87%]` | **0.0000** | 0.037 ms |
| **M5. Evidence Calculus (v0.2.1 Full Engine)**| 28 / 37 | **75.68%** | **0 / 22** | **0.00%** | **`[0.00%, 14.87%]`**| **0.1429** | **0.028 ms** |

### Matched-Coverage Analysis
- At matched coverage level ($C = 75.68\%$, 28 cases decided):
  - Score Thresholding False-Success Rate: **40.91% (9/22)**
  - Fixed Rules False-Success Rate: **4.55% (1/22)**
  - Evidence Calculus False-Success Rate: **0.00% (0/22)**
  - **Net Reduction**: Elimination of 9 false-success errors over score thresholding and elimination of the velocity anomaly error present in fixed rules.

---

## 2. Experiment H02: Evidence Dependency Graph (100 Iterations)

- **Mathematical Output Equivalence**: **100.0% (100/100 matching verdicts)**.
- **Node Evaluation**:
  - Full Recomputation: 12.0 nodes evaluated per pass.
  - Incremental Dirty Evaluation: 10.0 nodes recomputed, 4.0 nodes skipped.
- **Mean Execution Time**:
  - Full Recomputation: **0.040 ms**
  - Incremental Evaluation: **0.051 ms**
  - Overhead: 0.011 ms (BFS queue and dirty-set bookkeeping in pure Python).
- **Scientific Finding**: Retained as an explainability and provenance localization architecture; not claimed as a throughput accelerator for lightweight in-memory nodes.

---

## 3. Experiment H03: Evidence Budget Planner (Controlled Simulation)

- **Evaluation Type**: Controlled Monte Carlo Simulation across 20 synthetic defect scenarios.
- **Curated Tabletop Paired Recovery States**: 4 / 4 resolved (100.0% on static paired states).
- **Live Human In-the-Wild Trials**: **NOT MEASURED**.
- **Simulation Results**:
  - Generic Advice: 5/20 resolved (**25.0%**), Mean effort: 15.0s.
  - Fixed Templates: 18/20 resolved (**90.0%**), Mean effort: 6.0s.
  - Random Instruction: 10/20 resolved (**50.0%**), Mean effort: 7.0s.
  - VOI Budget Planner: 19/20 resolved (**95.0%**), Mean effort: 8.2s.

---

## 4. Experiment H04: Detector Independence (Controlled Simulation)

- **Evaluation Type**: Controlled Simulation of Correlated Distribution Shift ($ho = 0.88$).
- **Live Multi-Model Deep Checkpoints Executed**: **NOT YET MEASURED ON LIVE MODELS**.
- **Simulation Results**:
  - Single Model False-Success Rate: **100.0%**
  - Simple Majority Vote False-Success Rate: **100.0%**
  - Confidence-Weighted False-Success Rate: **100.0%**
  - Dependence-Aware Fusion False-Success Rate: **0.0%** (**100% Risk Reduction over Majority Vote**)
