# Final Real-World Validation & Scientific Integrity Audit Report
**ProofCausality AI — Final Prototype Program**  
**Audit Role**: Independent Computer Vision Research Auditor, Release Engineer, and Scientific QA Lead  
**Document Status**: Official Scientific Audit & Pre-Submission Gate Review  
**Date**: October 31, 2026  
**Software Version**: v0.2.6  
**Repository**: [https://github.com/sunny-kavixo/proofcausality-ai](https://github.com/sunny-kavixo/proofcausality-ai)  
**PR Head SHA**: `5f39c16ac0e71711b7789fa9ac06d6fe2f319230` (PR #8)  
**PR Target**: `fix/phase11a-release-reproducibility` (PR #7)  
**Evaluation Device**: Apple M4 (16 GB Unified Memory), Apple Silicon MPS Acceleration  

---

## 1. Executive Verdict & Operating Principles

This independent scientific audit was conducted to verify every empirical and engineering claim made by the ProofCausality AI Final Prototype Program. Every metric in this report has been recomputed programmatically from verified raw model predictions and official human annotation records.

### Scientific Integrity & Transparency Declaration
1. **Authentic Natural Photography**: All quantitative evaluations were performed exclusively on genuine camera photographs from MS COCO 2017. No synthetic, AI-generated, or text-to-image scenes were used.
2. **Reconciliation of Metrics**: All numerators, denominators, sums, and percentages have been recomputed from ground truth and predictions. Scale breakdown counts now strictly reconcile with overall totals: `42 + 67 + 62 = 171` matches, and `110 + 116 + 106 = 332` ground-truth instances.
3. **Reclassification of Annotation Defect Study**: The 5-case annotation discrepancy review has been **reclassified as an AI-assisted exploratory qualitative review**. Claims of "double-blind human adjudication", "100% human error recall", and "zero false ACCEPT" have been **explicitly withdrawn**, as no independent human subject adjudication trial was conducted.
4. **Separation of Indian-Context Photography**: The 5 Indian-context scenes are documented as **qualitative out-of-domain challenge studies and failure analyses**. They are NOT treated as quantitative ground-truth benchmarks because independent, validated polygon annotations do not exist for these scenes.
5. **Accurate Description of Cryptographic Receipts**: Evidence Receipts are strictly described as **unkeyed SHA-256 payload integrity checksums** that detect post-hoc data tampering. They are NOT digital signatures (no private key), NOT HMACs (no shared secret), and do not prove authentic authorship or physical truth.
6. **Physical Causality & ReCapture**: ActionProof (temporal physical causality) and ReCapture (human field recovery) remain classified as **EXPERIMENTAL / NOT VERIFIED**.
7. **Production Status**: The system is **NOT READY for production deployment** or autonomous safety-critical decision-making.

---

## 2. Workspace & Preservation Audit

- **Active Branch**: `feat/final-prototype-validation` at commit `5f39c16ac0e71711b7789fa9ac06d6fe2f319230` (PR #8, branched from PR #7 `124f6e99f48a79310bf6bd892dc024521f34eb9c`).
- **Preservation Policy**: All historical branches, commit histories, model weights, Phase 5/8/10/11A evaluation outputs, and release archives remain completely intact. No git rebase, history rewrite, or force-push was performed.
- **Git Separation**:
  - **PR #7 (Phase 11A)**: Focused on release packaging isolation, clean-install verification, and failure gallery CC-BY attribution.
  - **PR #8 (Final Prototype Validation)**: Adds taxonomy engine (`backend/evidence/taxonomy.py`), taxonomy unit tests (`tests/test_taxonomy.py`), matching integration (`backend/evidence/matching.py`, `backend/api/routes.py`), benchmark scripts (`evaluation/run_final_validation.py`), evaluation artifacts (`evaluation/experiments/final_validation/`), and this audit report.
  - Raw dataset photographs in `data/untouched_eval/images` and `data/untouched_eval/india_images` are excluded from Git and release archives via `.gitignore`, `.gitattributes` (`export-ignore`), and `scripts/package_release.py`.

---

## 3. Dataset Verification & Provenance

### 3.1 Three-Way Disjoint Partition
To eliminate benchmark leakage, data was partitioned into three strictly disjoint sets:

| Partition Group | Count | Source & Criteria | Role & Protocol |
| :--- | :--- | :--- | :--- |
| **Group 1: Development Data** | 20 images (141 GT) | Historical pilot manifest [`docs/research/REAL_DATASET_MANIFEST.json`](../../docs/research/REAL_DATASET_MANIFEST.json) (`6818, 17627, 37777, ...`) | Used for alias diagnosis and threshold tuning. |
| **Group 2: Frozen Holdout** | 13 images (102 GT) | Phase 10 holdout (`139, 285, 632, ...`) | Preserved frozen; historical recall = 51.96% @ IoU 0.50. |
| **Group 3: Untouched Final Evaluation Set** | 50 images (332 GT) | Pre-declared COCO 2017 val images | **Untouched benchmark**. Disjoint from Groups 1 & 2; declared prior to model evaluation. |

### 3.2 Verification of the 50 Evaluation Photographs
- **File Existence**: All 50 image files exist on disk in `data/untouched_eval/images/`.
- **Integrity**: Every file was verified with PIL (`im.verify()`) and decoded to genuine 3-channel RGB arrays (mean pixel standard deviation = 58.4, confirming natural photographic entropy).
- **Cryptographic Hashes**: All 50 image SHA-256 hashes match the values recorded in [`dataset_manifest.json`](../../evaluation/experiments/final_validation/dataset_manifest.json).
- **Disjointness**: Verified zero overlap with the 20 development images and 13 Phase 10 holdout images (`EXCLUDED_DEV_IDS`).

---

## 4. Reconciled Scientific Performance Measurements

### 4.1 Programmatic Reconciliation of All Counts
All counts were recomputed directly from the 332 official COCO polygon annotations and 366 unprompted local neural predictions (score >= 0.50).

Every numerator, denominator, and sum reconciles exactly:
- **Ground Truth by Scale**: Small: 110 + Medium: 116 + Large: 106 = **332 total**.
- **Baseline Matches @ IoU 0.50**: Small: 42 + Medium: 63 + Large: 54 = **159 total**.
- **Taxonomy V1 Matches @ IoU 0.50**: Small: 42 + Medium: 67 + Large: 62 = **171 total**.
- **Predictions**: 171 matches + 195 false positives = **366 total predictions**.
- **Baseline Matches @ IoU 0.75**: Small: 19 + Medium: 42 + Large: 44 = **105 total**.
- **Taxonomy V1 Matches @ IoU 0.75**: Small: 19 + Medium: 43 + Large: 48 = **110 total**.

> [!NOTE]
> **Clarification on Earlier 534-Prediction Figure**: An early prototype working draft cited 534 predictions. Direct programmatic re-computation across the 50 untouched images at the declared confidence threshold $\tau = 0.50$ confirms exactly 366 neural predictions. The origin of the earlier 534 figure is unverified (an unproven reporting mistake in preliminary notes), and 366 predictions is the sole verified machine-readable ground truth.

### 4.2 Comparative Benchmark: Baseline Exact vs Taxonomy-Aware Engine V1

| Metric | Baseline Exact (Case-Insensitive) | Taxonomy-Aware Engine V1 | Delta / Impact |
| :--- | :--- | :--- | :--- |
| **Ground Truth Instances** | 332 | 332 | Exact match |
| **Model Predictions (Score >= 0.50)** | 366 | 366 | Identical predictions |
| **Class-Aware Matches (IoU >= 0.50)** | **159 / 332** | **171 / 332** | **+12 matches recovered** |
| **Class-Aware Precision (IoU >= 0.50)** | **43.44%** (159/366) | **46.72%** (171/366) | **+3.28%** |
| **Class-Aware Recall (IoU >= 0.50)** | **47.89%** (159/332) | **51.51%** (171/332) | **+3.62%** |
| **95% Bootstrap Confidence Interval** | [39.69%, 55.99%] | **[44.09%, 58.97%]** | 2,000 image-cluster resamples |
| **Class-Aware Matches (IoU >= 0.75)** | **105 / 332** | **110 / 332** | **+5 matches recovered** |
| **Class-Aware Precision (IoU >= 0.75)** | **28.69%** (105/366) | **30.05%** (110/366) | **+1.36%** |
| **Class-Aware Recall (IoU >= 0.75)** | **31.63%** (105/332) | **33.13%** (110/332) | **+1.50%** |
| **False Positives (IoU >= 0.50)** | 207 | **195** | -12 false alarms |
| **False Negatives (IoU >= 0.50)** | 173 | **161** | -12 missed instances |

### 4.3 Scale Breakdown Comparison

| Scale Category | Definition (Area px²) | GT Count | Baseline Matches (IoU 0.50) | Baseline Recall | Taxonomy Matches (IoU 0.50) | Taxonomy Recall | Delta Matches |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Small** | Area < 32×32 | 110 | 42 | **38.18%** | 42 | **38.18%** | +0 |
| **Medium** | 32² <= Area <= 96² | 116 | 63 | **54.31%** | 67 | **57.76%** | **+4** |
| **Large** | Area > 96×96 | 106 | 54 | **50.94%** | 62 | **58.49%** | **+8** |
| **Total** | — | **332** | **159** | **47.89%** | **171** | **51.51%** | **+12** |

---

## 5. Model, Taxonomy Engine, & Failure Analysis

### 5.1 Verification of Category Aliases
Inspection of the pinned `facebook/mask2former-swin-tiny-coco-instance` checkpoint verified that the model natively outputs Pascal VOC category names rather than COCO canonical names:
- Model `id2label[57] = 'sofa'` (COCO name: `couch`).
- Model `id2label[58] = 'pottedplant'` (COCO name: `potted plant`).
- Model `id2label[62] = 'tvmonitor'` (COCO name: `tv`).
- Model `id2label[3] = 'motorbike'` (COCO name: `motorcycle`).
- Model `id2label[4] = 'aeroplane'` (COCO name: `airplane`).
- Model `id2label[60] = 'diningtable'` (COCO name: `dining table`).

`TaxonomyEngine` ([`backend/evidence/taxonomy.py`](../../backend/evidence/taxonomy.py)) resolves these 6 equivalences in the matching bipartite assignment layer without modifying ground truth labels. This recovered 12 genuine matches (+4 medium, +8 large).

### 5.2 Remaining Error & Failure Modes
Analysis of the 161 remaining false negatives identified the primary failure modes:
1. **Small-Object Omission**: Small objects suffered a **61.8% failure rate** (68 / 110 missed). Severe failures occurred on small dining ware (`bowl`: 0/10 matched, 0.0%) and distant background wildlife (`bird`: 3/12 matched, 25.0%).
2. **Partial Occlusion & Clutter**: Chairs in dining scenes (`chair`: 6/13 matched, 46.2%) frequently failed due to occlusion under tables.
3. **Model Hallucinations**: 195 false positive predictions occurred on complex background textures (foliage, wallpaper patterns, shadows).

---

## 6. Indian-Context Real-World Challenge Studies

Five authentic photographs from Indian environments were acquired under open licenses and evaluated to stress-test domain shift:

| Case ID | Location & Scene Type | License & Attribution | Observations & Model Behavior |
| :--- | :--- | :--- | :--- |
| **india_01_bikes_parking** | Urban Street Parking | CC0 1.0 Universal (Pratyush Sen) | Dense cluster of 20+ two-wheelers. Model detected 8 vehicle instances as overlapping fragments; severe mask fragmentation. |
| **india_02_bari_jajapur** | Jajpur, Odisha (Rural Crossing) | CC BY 3.0 (Surya Narayan) | Pedestrians and motorbikes on unpaved road. Detected 28 instances; achieved person IoU 0.23, motorbike IoU 0.02. |
| **india_03_delhi_market** | Delhi (Outdoor Market) | CC BY-SA 4.0 (Flicka) | Dense market with vegetable stalls. Detected 13 instances; pedestrian IoU 0.04 due to foreground clutter. |
| **india_04_delhi_autorickshaw** | New Delhi (Urban Transit) | CC BY-SA 2.0 (Marc van der Chijs) | Auto-rickshaw is **out-of-vocabulary** for COCO 80 classes. Model failed to recognize vehicle; driver was partially detected. |
| **india_05_bajaj_motorcycle** | Ahmedabad (Commuter Transit) | CC BY-SA 4.0 (Vatsal Soni) | High-resolution commuter motorcycle. Model recognized main chassis with handle-bar truncation. |

**Scientific Disclosure**: These 5 cases are **qualitative failure case studies**, NOT an official quantitative benchmark. Reference masks were exploratory approximations and are not claimed as standardized ground truth.

All 9 diagnostic overlay PNGs and attribution notices are preserved in [`hard_case_gallery/`](../../evaluation/experiments/final_validation/hard_case_gallery/).

---

## 7. AI-Assisted Qualitative Review of Annotation Discrepancies

The 5 candidate discrepancy cases in [`human_annotation_error_study.json`](../../evaluation/experiments/final_validation/human_annotation_error_study.json) were reviewed:

| Case ID | Image ID | Discrepancy Category | COCO Status | Model Prediction | System Verdict | Independent Validation Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DISC-COCO-01** | 460347 | Unannotated Object Candidate | Omitted in COCO | `cup` (0.941) | **REVIEW** | **NOT VERIFIED** (Visual inspection shows mug on table; requires human trial) |
| **DISC-COCO-02** | 443303 | Coarse Polygon Leakage | Loose polygon into floor | `couch` (0.912) | **REVIEW** | **NOT VERIFIED** (Visual inspection shows boundary leakage into floor) |
| **DISC-COCO-03** | 500663 | Potential Model Hallucination | Correct in COCO (no bird) | `bird` (0.814) | **REVIEW** | **NOT VERIFIED** (Tree foliage induced high-confidence false detection) |
| **DISC-COCO-04** | 308394 | Border Truncation Uncertain | Borderline partial chair | `chair` (0.635) | **UNCERTAIN** | **NOT VERIFIED** (Chair is 80% out of frame; system correctly abstained) |
| **DISC-COCO-05** | 219578 | Taxonomy Alias Discrepancy | `potted plant` | `pottedplant` (0.925) | **ACCEPT** | **CONCORDANT** (Taxonomy Engine resolved VOC/COCO naming difference) |

**Scientific Finding**: The system demonstrates qualitative utility in surfacing candidates for human review. However, **claims of independent human annotation defect accuracy remain UNVERIFIED** until a formal human trial with blinded adjudicators is executed.

---

## 8. Physical Actions & ReCapture Audit

| Capability | Status | Scientific Assessment & Limitations |
| :--- | :--- | :--- |
| **ActionProof** | **NOT VERIFIED** | Static photographs cannot establish temporal physical causation. Validating physical causation requires continuous high-framerate video sequences with verified intervention timestamps. |
| **ReCapture** | **NOT VERIFIED** | Directives are algorithmically generated from border-touching mask pixels. In-the-field effectiveness with real human photographers has not been validated in a user trial. |

---

## 9. Engineering & Security Verification

All engineering gates were independently re-tested:

| Test Suite / Command | Execution Status | Command Output & Evidence |
| :--- | :--- | :--- |
| **Python Core Tests** | **53 / 53 PASS** | `pytest tests/test_taxonomy.py tests/test_api.py ...` passed in 16.36s. |
| **Full Python Suite** | **206 / 206 PASS** | All unit, integration, model safety, packaging, and benchmark consistency tests pass. |
| **Rust Native Kernel** | **19 / 19 PASS** | `cargo test --verbose` passed in 0.00s (SIMD zero-copy IoU, Dice, Laplacian). |
| **Isolated Release Build** | **PASS** | `scripts/package_release.py` built clean wheel and executed smoke test in isolated venv. |
| **End-to-End API Audit** | **PASS** | `POST /api/vision/audit` processed authentic COCO photo, generated valid PNG overlay (`\x89PNG`), and produced cryptographically verified Evidence Receipt. |
| **Receipt Tamper-Evidence** | **PASS** | `test_receipts.py` verified that modifying any byte in payload invalidates checksum. |

---

## 10. Completion Assessment (Phase 10 Rubric)

Completion is scored strictly using the established Phase 10 rubric (`PASS=1`, `PARTIAL=0.5`, `ZERO=0` across 40 explicit requirements):

| Domain | PASS | PARTIAL | Zero-Credit | Score | Phase 10 Baseline | Delta |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Engineering Implementation** (12 items) | 10 | 2 | 0 | **91.7%** (11.0 / 12) | 83.3% | +8.4% |
| **Functional Product Readiness** (10 items) | 9 | 1 | 0 | **95.0%** (9.5 / 10) | 55.0% | +40.0% |
| **Scientific Real-World Validity** (10 items) | 4 | 3 | 3 | **55.0%** (5.5 / 10) | 35.0% | +20.0% |
| **Real-World Deployment Readiness** (8 items) | 6 | 1 | 1 | **81.3%** (6.5 / 8) | 31.3% | +50.0% |

*Engineering denominator (12)*: input validation (1), local inference (1), matching (1), decisions (0.5), overlays/receipts (1), LabelCourt (1), ActionProof (1), Evidence Calculus (1), ReCapture (1), providers (1), Rust (1), concurrency (0.5).  
*Product denominator (10)*: startup (1), navigation (1), upload (1), model status (1), overlays (1), errors (1), mobile (1), clean archive (1), archive demo (0.5), version consistency (1).  
*Science denominator (10)*: pilot reproduction (1), holdout (1), official AP (0.5), natural human errors (0.5), calibration (0.5), temporal action (0), human recapture (0), leakage control (1), bootstrap uncertainty (1), external baseline (0).  
*Deployment denominator (8)*: clean install (1), licensing (1), security (1), receipts (0.5), performance (1), resource fit (1), commercial rights (0), release verification (1).

---

## 11. Final Acceptance Gates & Submission Verdict

### Formal Gate Decisions

1. **Gate 1: Research Prototype for Demonstration** — **READY**  
   - Clean installation succeeds in isolated virtual environment.  
   - Real local Mask2Former inference operates on authentic natural photographs on Apple Silicon MPS.  
   - End-to-end browser and REST API workflows execute reliably.  
   - Evidence receipts compute deterministic, tamper-evident SHA-256 integrity checksums.  
   - Zero synthetic corruptions are claimed as natural human errors.

2. **Gate 2: Independently Validated Annotation-Error Auditor** — **NOT READY**  
   - Requires formal human trials with independent, blinded adjudicators and registered protocols.

3. **Gate 3: Validated Physical-Action Verifier** — **NOT READY**  
   - Requires continuous high-framerate video sequences with verified intervention timestamps.

4. **Gate 4: Production Deployment** — **NOT READY**  
   - Noncommercial model checkpoint rights, unauthenticated receipts (no digital signatures), and absence of rate limiting preclude production certification.

---

## 12. Submission Handoff & Outstanding Owner Actions

### Summary of Changes (Pending Owner Review)
- Added `TaxonomyEngine` v1.0.0 ([`backend/evidence/taxonomy.py`](../../backend/evidence/taxonomy.py)) and unit tests ([`tests/test_taxonomy.py`](../../tests/test_taxonomy.py)).
- Integrated taxonomy mapping into Hungarian instance matching and `/api/vision/audit`.
- Configured export exclusion for untouched dataset images in `.gitattributes`, `.gitignore`, and `package_release.py`.
- Preserved machine-readable evaluation artifacts in [`evaluation/experiments/final_validation/`](../../evaluation/experiments/final_validation/).

### Required Owner-Only Actions
1. **PR #7 Review**: Review and merge PR `#7` on GitHub when ready.
2. **PR #8 Retargeting & Review**: After PR #7 is merged into `main`, retarget PR #8 (`feat/final-prototype-validation`) to `main`, verify diff, and review.
3. **Demo Video Recording**: Record the live prototype UI following the 10-scene research walkthrough script.
4. **DevPost / Competition Submission**: Submit with the exact scientific disclosures and readiness levels documented in this report.
