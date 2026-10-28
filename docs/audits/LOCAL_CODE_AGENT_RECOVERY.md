# Local Code-Agent Recovery Audit

**Recovered by**: Antigravity AI — session `9ed450ec-806e-4f8c-9a46-578e6fcae947`
**Recovery date**: 2026-10-28
**Branch at recovery**: `phase5-genuine-inference-provenance-audit`
**Baseline commit**: `0a4eb4d3e5f8809c9da34849285c3eb2e09a826c` (tag `v0.2.4`, `origin/main`)

---

## Context

A previous ChatGPT Desktop Code Agent was performing Phase 5 work on the `phase5-genuine-inference-provenance-audit` branch. Its session was interrupted before it could commit or push. This document records the forensic inspection and recovery of those changes.

---

## Pre-recovery safety snapshot

```
/tmp/proofcausality-ai-phase5-safety-snapshot-20261028234900.tar.gz
```

No files were modified before this snapshot was created.

---

## Forensic findings — modified files (11)

| File | What changed | Classification |
|---|---|---|
| `README.md` | Corrected overstated benchmark claims; added Phase 5 audit table; separated controlled-corruption results from genuine model inference | **VERIFIED — improvement** |
| `backend/api/routes.py` | Evidence-category labels updated; provenance gate called on inference paths | **VERIFIED — improvement** |
| `backend/evidence/provenance.py` | `ProvenanceGate` now requires manifest proof for `VERIFIED_DATASET_IMAGE`; decoded bytes alone cannot claim `REAL_PHOTOGRAPH` | **VERIFIED — critical fix** |
| `docs/benchmarks/LABELCOURT_REAL_EVALUATION.md` | Updated denominator and interval labels to use image-grouped bootstrap | **VERIFIED — improvement** |
| `evaluation/benchmarks/labelcourt_real_benchmark.py` | Added image-grouped bootstrap CI columns; uses `grouped_statistics` module | **VERIFIED — improvement** |
| `evaluation/experiments/LABELCOURT_REAL_BENCHMARK_RESULTS.json` | Re-run with image-grouped intervals; numerators/denominators updated | **VERIFIED — reproduced artifact** |
| `frontend/static/app.js` | Evidence-category badges and provenance warnings in UI | **VERIFIED — improvement** |
| `frontend/static/index.html` | Added phase-5 disclaimer banner | **VERIFIED — improvement** |
| `frontend/static/style.css` | New evidence-category card styles | **VERIFIED — improvement** |
| `pyproject.toml` | Added optional `[inference]` dependency group for torch/transformers | **VERIFIED — improvement** |
| `tests/test_defect_repairs.py` | Two minor assertion fixes | **VERIFIED — improvement** |

---

## Forensic findings — new untracked files (9)

| Path | Classification |
|---|---|
| `backend/evidence/dataset_manifest.py` | **VERIFIED** — `DatasetSourceRecord` dataclass; hash and linkage validation; tested |
| `backend/providers/local_segmentation.py` | **VERIFIED** — `LocalMask2FormerProvider`; structurally excludes reference annotations from `predict()`; RLE helpers tested |
| `docs/research/PHASE5_BASELINE_AUDIT.md` | **VERIFIED** — honest record of pre-phase-5 baseline; no fabricated claims |
| `docs/research/PHASE5_FINAL_REPORT.md` | **VERIFIED** — honest gate-by-gate pass/fail; release decision is FAIL (correct; no release created) |
| `evaluation/benchmarks/genuine_model_benchmark.py` | **VERIFIED** — IoU on polygon masks; grouped bootstrap CIs; ground truth excluded from inference |
| `evaluation/experiments/genuine_model_predictions/` | **VERIFIED** — 6 artifacts: raw predictions, execution records, per-annotation results (CSV + JSON), summary, dataset manifest, failure gallery |
| `evaluation/grouped_statistics.py` | **VERIFIED** — `grouped_bootstrap_interval`; `assert_disjoint_groups`; pure NumPy; tested |
| `scripts/run_genuine_model_benchmark.py` | **VERIFIED** — thin CLI wrapper |
| `tests/test_phase5_integrity.py` | **VERIFIED** — provenance gate, manifest hash, RLE round-trip, leakage interface, grouped bootstrap |

---

## Test status at recovery

```
117 passed, 1 warning in 0.30s
```

All pre-existing 107 tests continue to pass. The 10 new Phase 5 tests also pass.

---

## Genuine model inference record (already executed)

| Field | Value |
|---|---|
| Model | `facebook/mask2former-swin-tiny-coco-instance` |
| Revision | `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6` |
| Checkpoint SHA-256 | `3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80` |
| Checkpoint size | 189,998,168 bytes (~190 MB) |
| Device | Apple MPS |
| Images evaluated | 20 |
| Annotations evaluated | 215 |
| Inference mode | UNPROMPTED — no annotation passed to `predict()` |
| Class-agnostic mean best-mask IoU | 0.5002 (95% CI 0.4260–0.5849) |
| Class-aware mean best-mask IoU | 0.3929 (95% CI 0.3205–0.4738) |
| Class-agnostic recall@IoU≥0.5 | 55.81% (95% CI 44.95%–68.26%) |
| Class-aware recall@IoU≥0.5 | 43.26% (95% CI 33.33%–54.29%) |
| Inference p50 latency | 134.382 ms |
| Inference p95 latency | 198.759 ms |

Limitations: 20 source images only; CC BY-NC 4.0 model weights; Transformers 5.17 reported two newly initialized Swin layer-norm parameters.

---

## Pre-publication gate status at recovery

| Gate | Status |
|---|---|
| A — Source commit pinned | PASS |
| B — Working-tree forensics | PASS (this document) |
| C — No data overwritten | PASS |
| D — Tests green | PASS (117/117) |
| E — Genuine model predictions present | PASS |
| F — Provenance leakage closed | PASS |
| G — Grouped bootstrap uncertainty | PASS |
| H — Train/test split disjoint | PASS (10 dev / 10 held-out images) |
| I — Benchmark script reproducible | PASS |
| J — Failure gallery present | PASS (12 images) |
| K — Limitations documented | PASS |
| L — README claims honest | PASS |
| M — No false release created | PASS |
| N — Remote CI | NOT VERIFIED (branch not pushed yet) |

---

## Recommended next steps

1. Commit all Phase 5 changes on `phase5-genuine-inference-provenance-audit`.
2. Push the branch and open a PR to `main`.
3. Verify CI passes (Python tests, Rust build).
4. Merge only after CI is green.
5. Do not tag a new release until the Swin layer-norm compatibility warning is resolved or formally accepted as a documented limitation.
