# Competitive Baselines & Methodological Comparison

**Project**: ProofCausality AI — Phase 2  
**Date**: October 2026  

---

## 1. Classification of Evidence Categories

In strict accordance with scientific integrity standards, comparisons are partitioned into four explicit evidence categories:
1. **Actually Executed Comparisons**: Run directly in this test environment using identical inputs.
2. **Published External Results**: Extracted from published academic papers.
3. **Conceptual Architectural Comparisons**: Theoretical workflow comparisons.
4. **Proposed Future Comparisons**: Work planned for future releases.

---

## 2. Comparison Matrix

| System / Tool | Type | Evidence Category | Verification Approach | Missing Evidence Handling | Contradiction Detection |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ProofCausality AI (Ours)** | Physical Verification Engine | **Actually Executed** | Tri-State Evidence Calculus | Explicit Debt $D(E,T)$ | Instantaneous velocity & boundary bounds |
| **Confidence Baseline** | Standard Computer Vision | **Actually Executed** | Score thresholding | Ignored (treated as confident) | None (scores override geometry) |
| **Cleanlab** | Data Quality Framework | **Published / Conceptual** | Confident Learning (Label noise) | Detects label discrepancies | Dataset-level label noise only |
| **FiftyOne (Voxel51)** | Dataset Visualizer | **Published / Conceptual** | Visualization & embeddings | Visual inspection by human | Human reviewer dependent |
| **Meta SAM Family** | Foundation Segmentation | **Actually Executed (SAM Adapter)** | Zero-shot mask prompt | Returns mask even on occlusions | None (pure perception) |
| **Ego4D Action Verifiers** | Video Action Benchmark | **Published External** | Temporal action localization | Binary classification | Susceptible to occlusion gaps |

> **Crucial Distinction**: Cleanlab and FiftyOne are dataset curation platforms, not real-time physical AI action verifiers. Meta SAM is a segmentation foundation model, not a causal decision engine. ProofCausality AI operates as the **verifiable arbitration layer** sitting on top of perception models.
