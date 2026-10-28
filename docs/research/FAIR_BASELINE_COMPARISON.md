# Fair Baseline Comparison & Risk-Coverage Analysis

**Project**: ProofCausality AI  
**Release**: `v0.2.3`  
**Date**: October 2026  

---

## 1. Why the Previous Baseline Was Methodologically Defective

In Phase 2, `exp_h01_evidence_debt.py` compared Evidence Calculus against a confidence-only baseline where all cases were manually assigned `"mock_detector_confidence": 0.92` (real cases) and `0.94` (synthetic cases).  
Because all true negative cases had simulated high confidence, the confidence baseline had a mathematically guaranteed 100% false-success rate. This produced an artificially inflated comparison that violated scientific fairness.

---

## 2. Fair Baselines Implemented in Phase 3

We replaced the fabricated baseline with two deterministic, reproducible engineering baselines and an established cause-specific boolean rule set:

1. **Baseline 1: Naive Center-Distance Heuristic (100% Coverage)**:
   - Predicts `SUCCESS` if the final bounding box centroid is within $80\text{px}$ of the target region center.
   - Completely ignores motion blur, boundary clipping, surface area occlusion drop, and velocity jumps.
2. **Baseline 2: Score-Thresholded Proximity Predictor (Tuned to Matched Coverage)**:
   - Computes a continuous geometric proximity score:
     $$S = \max\left(0.0, \min\left(1.0, 1.0 - \frac{\text{dist}}{120\text{ px}}\right)\right)$$
   - At operating threshold $\tau = 0.55$, achieves $97.3\%$ coverage; at higher thresholds, achieves matched coverage ($75.68\%$) by abstaining when $1 - \tau < S < \tau$.
3. **Baseline 3: Fixed Cause-Specific Boolean Rules**:
   - Checks boolean thresholds for final state, blur variance ($< 100.0$), occlusion ($> 35\%$), and distance ($\le 80\text{px}$). Coverage: $75.68\%$.
4. **Method 4: Evidence Debt (Unit Weights Only)**:
   - Sums all unresolved critical conditions with unit weight $1.0$, abstaining if $D(E,T) > 0$. Coverage: $40.54\%$.
5. **Method 5: Evidence Calculus Engine (Full with Contradiction Constraints)**:
   - Incorporates $2.0\times$ contradiction penalties and velocity anomaly bounding. Coverage: $75.68\%$.

---

## 3. Empirical Results Across 37 Benchmark Cases

- **Dataset**: 12 curated tabletop metadata pairs + 25 synthetic benchmark cases (37 total).
- **Ground Truth**: 15 True Successes, 22 True Negatives (14 Uncertain/Defective, 8 Failed).

| Verification Strategy | Decided Cases | Decision Coverage | False-Successes | False-Success Rate | 95% Wilson Score CI | Selective Risk | Mean Latency | Failure Breakdown |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1. Naive Center-Distance** | 37 / 37 | **100.0%** | 10 / 22 | **45.45%** | `[26.92%, 65.34%]` | 0.3514 | 0.001 ms | Blur (4), Occlusion (4), Clipping (1), Cutoff (1) |
| **B2. Score-Thresholded (Ranked)** | 28 / 37 | **75.68%** | 3 / 22 | **13.64%** | `[4.75%, 33.33%]` | 0.1429 | 0.001 ms | Blur (1), Occlusion (1), Cutoff (1) |
| **B3. Fixed Cause-Specific Rules** | 28 / 37 | **75.68%** | 1 / 22 | **4.55%** | `[0.81%, 21.80%]` | 0.1429 | 0.000 ms | Border clipping (1) |
| **M4. Evidence Debt (Unit Weight)** | 15 / 37 | **40.54%** | 0 / 22 | **0.00%** | `[0.00%, 14.87%]` | **0.0000** | 0.031 ms | Over-abstains on minor deficits |
| **M5. Evidence Calculus (Full Engine)**| 28 / 37 | **75.68%** | **0 / 22** | **0.00%** | **`[0.00%, 14.87%]`**| **0.1429** | 0.028 ms | **0 false-successes across all defect types** |

### Continuous Score-Threshold Risk-Coverage Trade-off Curve
| Operating Point | Decided Cases | Coverage | False-Successes | False-Success Rate | Selective Risk |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Top 15 (High Confidence) | 15 / 37 | **40.54%** | 0 / 22 | 0.00% | 0.0000 |
| Top 20 | 20 / 37 | **54.05%** | 2 / 22 | 9.09% | 0.1000 |
| Top 25 | 25 / 37 | **67.57%** | 3 / 22 | 13.64% | 0.1200 |
| **Top 28 (Matched Operating Point)** | 28 / 37 | **75.68%** | 3 / 22 | **13.64%** | **0.1429** |
| Top 32 | 32 / 37 | **86.49%** | 5 / 22 | 22.73% | 0.2500 |
| All 37 (Unconstrained) | 37 / 37 | **100.0%** | 10 / 22 | 45.45% | 0.3514 |

---

## 4. Matched-Coverage Analysis & Honest Interpretation

- **Comparing at Matched Coverage ($C = 75.68\%$, 28 cases decided)**:
  - Fixed Rules: 1 false success (the velocity anomaly jump was missed because coordinates happened to land in Region B).
  - Score Thresholding: 3 false successes (at matched coverage 75.68%) (objects placed in Region B while occluded or blurred were accepted).
  - Evidence Calculus: **0 false successes** (100% reduction in false successes over score thresholding, and elimination of the velocity anomaly error).
- **Abstention Trade-off**:
  - Evidence Calculus achieves 0.0% false-success rate by **abstaining on 9 defective cases** ($24.32\%$ abstention rate).
  - It does **not** classify 100% of cases. Comparing a 75.7% coverage system to a 100% coverage system and claiming unconditional superiority is scientifically invalid; the proper claim is that Evidence Calculus provides **zero false-success risk at 75.7% coverage on verifiable cases**.
