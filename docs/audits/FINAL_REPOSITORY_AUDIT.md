# Final Repository Audit — v0.2.8 candidate + ProofLoop

**Audit date:** 2026-11-03
**Current audited commit:** `ac3d2560037d9912344700caf9b65aad9d4a347b` (built on preserved start commit `6f9ba97`)
**Scope:** Git-tracked repository files, runtime paths, tests, release evidence, and the ProofLoop change set.

## Executive result

- Working tree at audit start: clean on `main`, aligned with `origin/main`.
- Tracked inventory: **333 files**, with byte counts and SHA-256 hashes in [`FILE_INVENTORY.json`](FILE_INVENTORY.json).
- Historical tags, signed commits, release archives, datasets, and ignored local assets were preserved.
- No open P0 defect was found in the audited runtime paths.
- ProofLoop is implemented as a deterministic policy layer over Evidence Calculus; it does not claim a new trained model or physical causality.
- Independent human validation, authentic temporal-video validation, live hosted Meta inference, and production certification remain unverified or blocked.

The inventory is an automated complete listing of tracked paths. Text source and configuration were inspected through targeted imports/callers/tests; opaque binary assets were hash- and path-audited rather than read line by line. Ignored virtual environments, caches, local checkpoints, and excluded photographs are intentionally outside the tracked-file inventory.

## Preservation checks

| Check | Result | Evidence |
|---|---|---|
| Working tree and branch | PASS at start | `git status --short --branch` → clean `main` |
| Main commit | PASS | `6f9ba97` at audit start; release v0.2.7 targets this commit |
| Remote refs | PASS | `origin` retained; no reset, rebase, force push, or deletion |
| v0.2.7 release | PASS | GitHub release exists with tar, ZIP, manifest, and recorded SHA-256 digests |
| Historical v0.2.6 | PASS | Historical tag/archive left unchanged |
| Signed history | PASS | Existing signed commits/tags preserved; no signing configuration changes |
| User-owned ignored data | PASS | `.venv`, caches, local model/data directories, and benchmark logs preserved |

## Component audit

| Area | Primary paths | Use/dependents | Tests/evidence | Finding |
|---|---|---|---|---|
| FastAPI application and routes | `backend/app.py`, `backend/api/routes.py` | Browser, API clients, all engines | `tests/test_api.py`, security tests, Chrome acceptance | P1 production boundary: no application authentication; local prototype only |
| Evidence schemas and receipts | `backend/evidence/schemas.py`, `receipts.py` | LabelCourt, ActionProof, calculus, vision audit | `tests/test_schemas.py`, `test_receipts.py` | SHA-256 is integrity-only, not a signature or truth proof |
| LabelCourt | `backend/labelcourt/*` | Annotation audit endpoint/UI | `tests/test_labelcourt.py`, real-image reports | Controlled corruption results must not be called independent human validation |
| ActionProof | `backend/actionproof/*` | Action endpoint/UI, legacy callers | `tests/test_actionproof.py` | 2D before/after evidence is not temporal causality proof |
| Evidence Calculus | `backend/evidence/calculus/*` | Calculus API/UI, ProofLoop | calculus, graph, planner, canonical demonstration tests | Deterministic feature policy; thresholds are not learned calibration |
| ProofLoop | `backend/evidence/proofloop.py`, `/api/proofloop/*` | Metadata and server-derived media state machine | `tests/test_proofloop.py`, `evaluation/benchmark_proofloop.py`, `evaluation/benchmark_proofloop_media.py` | Metadata and media paths are explicitly separated; media binds image/model hashes but does not prove temporal causality |
| ReCapture | `backend/evidence/recapture.py`, existing follow-up route | ActionProof/calculus/UI | `tests/test_recapture.py`, API tests | Field effectiveness remains unmeasured |
| Local segmentation | `backend/providers/local_segmentation.py` | `/api/vision/audit`, genuine COCO run | model smoke report, real vision reports | Diagnostic recall is modest; no production accuracy claim |
| Meta adapter | `backend/providers/meta_adapter.py` | Optional remote provider surface | `tests/test_meta_adapter.py`, security tests | Live authenticated request not performed; adapter remains blocked |
| Rust engine | `native/proofcausality_core/*` | mask/blur acceleration | 19 Rust tests, Python accelerator tests | Optional native path with Python fallback |
| Frontend | `frontend/static/*` | FastAPI static UI and browser acceptance | Chrome 5/5 historical v0.2.7 evidence; JS syntax check | ProofLoop UI uses real API calls and labels controlled metadata |
| Evaluation | `evaluation/*`, `docs/benchmarks/*` | Reproduction and release evidence | existing benchmark tests plus ProofLoop report | Split and provenance disclosures retained |
| Dataset/assets | `data/*`, manifests, attribution docs | Tests, model demos, release exclusions | licensing/provenance tests and package checks | Restricted photographs excluded from release archives |
| CI/release tooling | `.github/workflows/ci.yml`, `scripts/*`, `pyproject.toml` | CI, clean install, archives | GitHub CI and local release checks | Model job correctly installs inference extras |
| Documentation | `README.md`, `docs/**` | User/research/release communication | consistency tests and manual audit | Historical documents require version/date context; stale claims are listed below |

## Security and trust-boundary findings

### P0 — none open in this audit

The prior server-authoritative provenance repair, strict image validation, receipt-store bound, error sanitization, and release exclusion checks remain present and covered by the existing tests.

### P1 — important limitations

1. The FastAPI prototype has no user authentication, authorization, durable receipt database, or externally validated rate-limit policy. It must not be deployed as an internet-facing production service.
2. Model checkpoints and hosted-provider responses are trust boundaries. The local model revision/hash checks are present, but live Meta inference has not been authorized or executed in this environment.
3. Metadata ProofLoop accepts measured features and labels receipts `USER_SUPPLIED_METADATA` / `METADATA_ONLY`; the media ProofLoop accepts image bytes, derives measurements server-side, and labels receipts `SERVER_DERIVED_MEDIA`. Neither path establishes temporal continuity or physical causality.
4. Independent human annotation/recapture effectiveness and temporal causal validation are not available.

### P2 — correctness/scientific limitations

1. Current local Mask2Former diagnostic performance is 171/332 taxonomy-aware matches at IoU .50 and 110/332 at IoU .75 on the frozen 50-image diagnostic evaluation; this limits any downstream verification claim.
2. Thresholds and planner efficacy are deterministic policy parameters, not learned calibration from independent labels.
3. Existing ActionProof uses 2D geometry and image-pair/metadata evidence; temporal identity tracking and intervention-controlled causality are absent.
4. The unkeyed SHA-256 receipt detects post-hoc payload modification but does not establish authorship, source authenticity, or physical truth.

### P3 — documentation/operational follow-up

1. Some historical handoff and compliance documents retain older version numbers and dates because they record earlier release states. Current submission material must use v0.2.7 and the current owner-verified portal deadline.
2. A full dependency vulnerability scan and external security certification are not represented as completed evidence in this checkout.

## Required release language

Use `MEASURED`, `CONTROLLED FIXTURE`, `DIAGNOSTIC`, `BLOCKED`, `NOT COMPLETED`, and `NOT CERTIFIED` exactly where applicable. Do not convert any P1/P2 limitation above into a production or scientific-validation claim.
