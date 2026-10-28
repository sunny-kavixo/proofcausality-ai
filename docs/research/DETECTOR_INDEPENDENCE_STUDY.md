# Research Report: Detector-Independence Audit (H04)

**Hypothesis**: Multiple vision models agreeing does not guarantee correctness under correlated dataset priors or distribution shifts.

---

## 1. Methodology

We evaluated multi-detector fusion under a simulated distribution shift (harsh directional shadow) where Detector A (ViT-based) and Detector B (CNN-based with similar pretraining) shared a correlated false-positive error ($ho = 0.88$). Detector C had independent error characteristics ($ho = 0.12$).

### Dependence-Aware Weight Adjustment
For model $i$, its effective weight is discounted by its average correlation with other models:
$$w_i^{	ext{eff}} = rac{c_i}{1 + \gamma \cdot ar{ho}_i}$$

---

## 2. Experimental Results (50 Distribution-Shifted Trials)

| Fusion Strategy | False-Success Rate | Effective Weight Det A | Effective Weight Det C |
| :--- | :--- | :--- | :--- |
| **Single Model (Det A)** | 100.0% | 1.000 | 0.000 |
| **Simple Majority Vote** | 100.0% | 0.333 | 0.333 |
| **Confidence-Weighted Fusion** | 100.0% | 0.338 | 0.352 |
| **Dependence-Aware Fusion** | **0.0%** | **0.231** | **0.538** |

### Key Finding
Dependence-aware fusion prevented false consensus by down-weighting the redundant, correlated models (Det A and B), allowing independent model C to correctly reject the unverified claim.
