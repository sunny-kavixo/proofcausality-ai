# Consolidated Empirical Research Results

**Date**: October 2026  
**Environment**: Apple M4 (macOS 15, ARM64), Python 3.14.5, Rust 1.99.0  

---

## 1. Experiment H01: Evidence Debt vs Baselines (37 Cases)

- **Dataset**: 12 curated tabletop manipulation cases + 25 synthetic benchmark cases.
- **Ground Truth**: 15 True Successes, 22 True Negatives (14 Uncertain, 8 Failed).

$$\begin{array}{lccccc}
\hline
\textbf{Architecture} & \textbf{False-Successes} & \textbf{FS Rate} & \textbf{Coverage} & \textbf{Selective Risk} & \textbf{Latency (ms)} \\
\hline
\text{A. Confidence-Only (}\ge 0.85\text{)} & 22 / 22 & 100.0\% & 100.0\% & 0.5946 & 0.000 \\
\text{B. Fixed Boolean Rules} & 1 / 22 & 4.55\% & 75.68\% & 0.1429 & 0.001 \\
\text{C. Evidence Debt (Unit Weights)} & 0 / 22 & 0.0\% & 40.54\% & 0.0000 & 0.032 \\
\textbf{D. Evidence Calculus (Full)} & \mathbf{0 / 22} & \mathbf{0.0\%} & \mathbf{75.68\%} & \mathbf{0.1429} & \mathbf{0.026} \\
\hline
\end{array}$$

---

## 2. Experiment H02: Evidence Dependency Graph (100 Iterations)

- **Mathematical Output Equivalence**: **100.0% (100/100 matching verdicts)**.
- **Node Evaluation**:
  - Full Recomputation: 12.0 nodes evaluated per pass.
  - Incremental Dirty Evaluation: 10.0 nodes recomputed, 4.0 nodes skipped.
- **Mean Execution Time**:
  - Full Recomputation: **0.042 ms**
  - Incremental Evaluation: **0.053 ms**
  - Overhead: 0.011 ms (queue operations on in-memory graph).

---

## 3. Experiment H03: Evidence Budget Planner (20 Defective Cases)

- **Generic Advice**: 6/20 resolved (**30.0%**), Mean effort: 15.0s.
- **Fixed Templates**: 17/20 resolved (**85.0%**), Mean effort: 6.0s.
- **Random Instruction**: 7/20 resolved (**35.0%**), Mean effort: 7.0s.
- **VOI Budget Planner**: 19/20 resolved (**95.0%**), Mean effort: 8.2s.

---

## 4. Experiment H04: Detector Independence (50 Shifted Trials)

- **Single Model False-Success Rate**: **100.0%**
- **Simple Majority Vote False-Success Rate**: **100.0%**
- **Confidence-Weighted False-Success Rate**: **100.0%**
- **Dependence-Aware Fusion False-Success Rate**: **0.0%** (**100% Risk Reduction**)
