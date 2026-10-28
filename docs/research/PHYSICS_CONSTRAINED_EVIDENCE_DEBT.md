# Research Report: Physics-Constrained Evidence Debt (H01)

**Hypothesis**: Explicitly representing missing and contradictory task-critical evidence predicts false-success judgments better than confidence-only verification.

---

## 1. Formulation

The Evidence Debt $D(E, \mathcal{T})$ quantifies unresolved observational deficit across critical task predicates:

$$D(E, \mathcal{T}) = \sum_{c \in \mathcal{C}_{	ext{critical}}} w_c \cdot \delta(c)$$

Where the predicate deficit function is defined as:
$$\delta(c) = egin{cases}
0.0 & 	ext{if } \sigma(c) = 	ext{SUPPORTED} \
1.0 & 	ext{if } \sigma(c) = 	ext{UNOBSERVABLE} \
2.0 & 	ext{if } \sigma(c) = 	ext{CONTRADICTED}
\end{cases}$$

### Contradiction Penalty Weighting
Contradictory observations carry a penalty weight of $2.0 	imes$ because observing a physical impossibility (e.g. teleportation) or incorrect destination falsifies the claim with higher certainty than merely missing a frame.

---

## 2. Experimental Evaluation (37 Cases)

Evaluated across **12 curated real-world tabletop cases** (photographed physical pairs under natural lighting) and **25 synthetic benchmark cases**:

| Method | False-Successes | False-Success Rate | Coverage | Selective Risk | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A. Confidence-Only Baseline ($\ge 0.85$)** | 22 / 22 | **100.0%** | 100.0% | 0.5946 | 0.000 ms |
| **B. Fixed Boolean Rules** | 1 / 22 | **4.55%** | 75.68% | 0.1429 | 0.001 ms |
| **C. Evidence Debt (Unit Weight Only)** | 0 / 22 | **0.0%** | 40.54% | 0.0000 | 0.032 ms |
| **D. Evidence Debt + Contradiction Constraints** | 0 / 22 | **0.0%** | **75.68%** | **0.1429** | **0.026 ms** |

### Key Findings
1. **Confidence Failure**: Confidence-only verification accepts cases where the object was completely occluded or placed in the wrong basket simply because the detector confidently recognized the object elsewhere.
2. **Contradiction Value**: Distinguishing contradictions from unobservability restored coverage from **40.54%** to **75.68%** while preserving **0.0% false-success rate**.
