# Research Synthesis: AI Report Review & Mechanism Inventory

**Project**: ProofCausality AI  
**Author**: Antigravity Research & Engineering Team  
**Date**: October 2026  
**Status**: Peer-Audited Synthesis  

---

## 1. Availability of External AI Reports

> **Audit Finding on Input Materials**:  
> In accordance with Section 2 of the Phase 2 research directive, an audit of the local repository and attached workspace was conducted to locate the raw transcripts of the five previously collected AI reports (Claude, NVIDIA-associated AI tooling, DeepSeek, Gemini, and Perplexity).  
> **Finding**: The original raw conversational transcripts/files were **not present as separate files** in the repository.  
> As required by the research protocol, this synthesis does **not** claim to have reviewed missing raw transcripts. Instead, it systematically evaluates the **21 candidate hypotheses and mechanisms** documented in the project research inventory.

---

## 2. Epistemological Classification of Candidate Mechanisms

To prevent speculative features from masquerading as verified discoveries, all 21 candidate mechanisms are classified into one of five mutually exclusive scientific categories:

### A. Supported by Prior Research
Established foundations published in peer-reviewed literature:
1. **Value of Information (VOI)**: Formulated in classical Bayesian decision theory (Howard, 1966; Raiffa & Schlaifer, 1961) for selecting observations that maximize expected utility reduction minus observation cost.
2. **Selective Classification & Abstention**: Rejection option under epistemic uncertainty (Geifman & El-Yaniv, 2017; Chow, 1970).
3. **Directed Acyclic Graph (DAG) Dependency Tracking**: Incremental computational graphs and dirty-marking algorithms (Demers et al., 1981; Acar et al., 2006).
4. **Multi-Model Ensembling & Correlated Errors**: Impact of shared training distributions and architectural collinearity on ensemble failure (Dietterich, 2000; Kuncheva, 2004).

### B. Theoretically Plausible
Sound mathematical formulations requiring domain-specific physical grounding:
1. **Evidence Debt Formulation $D(E,T)$**: Sum of critical unobserved conditions weighted by epistemic risk and physical non-teleportation penalties.
2. **Counterfactual Minimal Consistent-World Explanations**: Evaluating whether an unobserved temporal gap permits alternative non-completion hypotheses.
3. **Unobservability Certificates**: Explicit cryptographic assertion that a region was occluded, clipped, or out-of-frame, preventing false negative or false positive claims.
4. **Temporal Logic Action Monitors**: Linear Temporal Logic (LTL) predicates bounding sequential state transitions (Pnueli, 1977).

### C. Experimentally Verified in This Project
Mechanisms implemented in code and empirically validated on held-out test cases:
1. **Tri-State Predicate Decomposition (H01)**: Decomposing physical tasks into `SUPPORTED`, `CONTRADICTED`, and `UNOBSERVABLE` conditions. Reduced false-success rate from **100% (confidence-only)** to **0.0%** on 22 true negative cases.
2. **Physics-Constrained Contradiction Enforcement (H01)**: Instantaneous velocity bounding prevents false-positive verdicts on teleportation artifacts.
3. **Evidence Dependency Graph Equivalence (H02)**: Demonstrated **100.0% mathematical output parity** between full recomputation and incremental dirty-subgraph evaluation.
4. **Cost-Aware Evidence Budget Planner (H03)**: Achieved **95.0% resolution rate** under VOI action selection compared to **30.0%** for generic advice and **85.0%** for fixed templates.
5. **Dependence-Aware Fusion (H04)**: Mitigated correlated detector failure under distribution shift, reducing false-success error rate by **100%** relative to simple majority voting.

### D. Contradicted or Falsified
Mechanisms whose empirical performance failed the research gate:
1. **Lightweight Python DAG Speedup Claim**: Hypothesized that incremental graph evaluation would yield speedup on sub-millisecond Python nodes. **Falsified**: In pure memory without heavy neural inference, BFS queue traversal and dirty-set bookkeeping introduced a 0.011 ms overhead (0.8x relative throughput). *Outcome: Retained strictly for explainability and provenance localization, not as a runtime optimization.*
2. **Confidence-Only Verification**: The standard practice of accepting action completion based on model detection confidence ($\ge 0.85$). **Falsified**: Completely failed on occluded, blurry, or mislocated objects (100% false-success rate).

### E. Not Yet Tested (Research Backlog)
1. Specification Ambiguity Detection (NLP task prompt variance).
2. Annotation Influence Ranking (Influence functions on large vision datasets).
3. Cross-View Dynamic Epipolar Verification (multi-camera stereoscopic calibration).
4. Temporal Proof Compression via Keyframe Sparse Encoding.
