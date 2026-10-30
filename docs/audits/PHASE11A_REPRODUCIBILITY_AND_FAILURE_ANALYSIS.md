# ProofCausality AI — Phase 11A Reproducibility and Failure Analysis

**Date:** 2026-10-30  
**Branch:** `fix/phase11a-release-reproducibility`  
**Baseline:** `main` at `4e2503a360411a70214e71a45fe8014846e8780a`  
**Release under test:** exact v0.2.6 source archive, SHA-256 `856c968b9b1ac23a49c404b9ff04555c7c8a1cd7f7dd963be7b7bad7bbd19f76`

## Executive result

Phase 11A repairs the source-package discovery failure, declares the missing SciPy runtime dependency, bounds dependency compatibility, packages frontend assets, makes excluded-photo prerequisites explicit, replaces the unsafe downloader, and adds artifact-level release validation. The enriched checkout now passes **191 tests** after the three new downloader regressions are included. A dependency-only environment installs successfully, serves `/api/health`, and runs the non-model suite with explicit model-only skips.

This work does **not** make ProofCausality production-ready or scientifically validated. The 13-image holdout remains a small developer-controlled diagnostic set. No natural annotation-error adjudication, real temporal action benchmark, human ReCapture study, official COCO AP evaluation, or authenticated receipt signing was added.

## 1. Exact v0.2.6 reproduction

Two environments were intentionally separated.

| Environment | Result | Interpretation |
|---|---:|---|
| Fresh venv, install exact extracted archive with `.[test]` | **INSTALL FAILED** | Setuptools refused ambiguous flat-layout discovery: `data`, `native`, `backend`, `frontend`, and `evaluation` were all top-level candidates. Tests could not start. |
| Existing enriched project venv against exact extracted archive | **160 passed, 4 failed, 24 errors** | Reproduced the Phase 10 signature exactly. |

The four failures were:

1. two dataset-license tests assumed 33 local COCO JPEGs that are deliberately excluded from release archives;
2. one test required `.gitattributes`, itself excluded from the archive;
3. one test ran `git archive HEAD` inside an export with no `.git` directory.

All 24 setup errors came from three module fixtures asking `/api/vision/samples` for omitted COCO photographs and asserting the returned list was non-empty. The affected tests never reached their intended model-error, provenance, or vision assertions. The complete per-test inventory is `evaluation/experiments/phase11a/before/failure_inventory.json`; the original JUnit and console evidence are preserved beside it.

## 2. Root causes and repairs

| Root cause | Repair | Verification |
|---|---|---|
| No build-system or explicit package discovery | Added setuptools build metadata and explicit `backend`, `evaluation`, `frontend`, and `scripts` discovery | Fresh `pip install '.[test]'` succeeds on Python 3.14 |
| Frontend files not guaranteed in a wheel | Made `frontend` a package and included `static/*` as package data | Archive validator installs wheel outside source tree and asserts `static/index.html` exists |
| SciPy imported by production matching code but undeclared | Added bounded core requirement `scipy>=1.11,<2` | Fresh health import succeeds with declared dependencies only |
| Unbounded dependency drift | Added conservative major-version ceilings to core, test, and inference extras | Resolver output preserved in `after_clean_install.log` |
| Optional photos treated as mandatory test fixtures | Tests now issue narrow, explicit skips only when the external COCO photographs are absent | Enriched environment still executes all tests; archive environment no longer errors at setup |
| Checkout-only release tests ran in exports | Tests identify absence of `.git`/`.gitattributes` and skip only those maintainer checks | No broad suite skip was introduced |
| Release smoke test imported source without installing it | Validator now builds a wheel, installs it in a throwaway venv, imports outside the source directory, checks frontend data, and runs Cargo check | Protects against the exact flat-layout regression |
| Downloader wrote on import, depended on `/tmp`, used HTTP, invented license defaults, and overwrote canonical docs | Replaced with explicit CLI requiring official annotations, dry-run default, `--write` opt-in, HTTPS, atomic downloads, exact metadata, observed hashes, and overwrite refusal | Three offline safety regressions cover missing IDs, unknown-license handling, and pre-network overwrite refusal |

The downloader records hashes it observes; it does not misrepresent those values as publisher-signed checksums. Third-party image and model licenses remain separate from the repository MIT license.

## 3. Before/after test evidence

| Check | Before | After repair |
|---|---:|---:|
| Exact release, enriched environment | 160 passed / 4 failed / 24 errors | 188 passed before adding downloader tests |
| Current enriched checkout | — | 191 passed |
| Fresh declared-dependency venv | install failure | 177 passed / 14 model-only skips after new tests |
| Health endpoint in fresh venv | unavailable | HTTP 200, version 0.2.6 |
| Wheel build/install from extracted archive | failed discovery | pass after repair commit |
| Full tests in committed clean source archive | 160 passed / 4 failed / 24 errors | 163 passed / 28 explicit optional-prerequisite skips |

Model-only skips are tied to the absence of the optional `inference` extra/checkpoint. Genuine-photo skips are tied to excluded COCO files and state the acquisition command. Core deterministic behavior, API error handling that does not require the model, downloader safety, packaging metadata, and health loading still execute.

## 4. Natural-photo holdout failure analysis

The Phase 10 split was not changed: 13 predeclared local COCO val2017 images, 102 polygon ground truths, and 135 unprompted model predictions. No holdout thresholds were tuned.

| Diagnostic | Result |
|---|---:|
| Class-aware true positives at IoU 0.50 | 53 |
| Unmatched predictions / false positives | 82 |
| Unmatched ground truths / false negatives | 49 |
| Small-object recall | 15.38% (39 GT) |
| Medium-object recall | 69.44% (36 GT) |
| Large-object recall | 81.48% (27 GT) |
| Prediction-score median / p90 | 0.8499 / 0.9692 |
| Predictions below 0.60 / 0.70 | 25 / 45 |
| GT with agnostic IoU ≥ 0.50 but class-aware IoU < 0.50 | 6 |
| Image megapixels vs recall, Spearman rho | 0.1114 (`p=0.7171`, n=13) |

The dominant measured failure is scale, not image resolution: small-object recall is dramatically lower, while the exploratory resolution correlation is weak and uninformative at n=13. Books are a conspicuous category failure (0/15 class-aware matches); people are comparatively strong (31/37). `potted plant` has 0/3 class-aware matches while two instances have class-agnostic IoU ≥ 0.50, consistent with a label-taxonomy signal such as `potted plant` versus `pottedplant`. Similar saved outputs contain `tvmonitor` versus COCO `tv`. These are hypotheses, not corrected benchmark scores: Phase 10 did not retain the matched prediction-mask identity needed to separate aliases from true misclassification retrospectively.

High confidence does not imply correct instance matching: 48/135 predictions scored at least 0.90 while the class-aware precision at IoU 0.50 was only 39.26%. Raising the confidence threshold might remove some low-score false positives, but it would also change recall; no post-hoc threshold was selected on this holdout.

Occlusion was not analyzed because the frozen set has no independent occlusion labels. A controlled downscale/crop experiment was not run on the holdout because repeated post-hoc probing would turn a diagnostic set into a development set. Future controlled experiments should use the historical 20-image pilot as development data, freeze label aliases and thresholds, then evaluate once on a new untouched set.

Machine-readable category, image, confidence, scale, resolution, and not-measured fields are in `evaluation/experiments/phase11a/holdout_failure_analysis.json`.

## 5. Claims and readiness

The README now places the research limitation at the top and documents clean installation, explicit dataset acquisition, separate model installation, license boundaries, and non-destructive downloader behavior.

Claims that remain supportable:

- genuine local Mask2Former inference runs on pixels when optional dependencies, checkpoint, and photographs are present;
- deterministic evidence and tri-state logic execute and are regression-tested;
- controlled reference-derived corruption F1 is reproducible under its narrow protocol;
- SHA-256 receipts are tamper-evident consistency records.

Claims that remain unsupported:

- production readiness;
- independently validated annotation-error detection;
- proof of physical causality or real action completion;
- human ReCapture effectiveness;
- tamper-proof or authenticated-origin receipts;
- official COCO AP or broad natural-photo generalization.

## 6. Remaining release gates

Before any new tag or release, CI should build the committed source archive, run the artifact-level validator, install both sdist/wheel in clean environments, exercise `/api/health`, run the core suite without optional assets, and run the enriched model/photo suite where those prerequisites are provisioned. The published v0.2.6 assets are historical and were not modified.

No merge, tag, or release was performed in Phase 11A.
