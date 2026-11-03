# ProofCausality AI — Phase 10 Independent Validation

**Audit date:** 2026-10-30  
**Repository:** `sunny-kavixo/proofcausality-ai`  
**Audited branch / baseline:** `main` at `4e2503a360411a70214e71a45fe8014846e8780a`  
**Audit work branch:** `audit/phase10-independent-validation`  
**Release checked:** `v0.2.6` (`5094ef1e0f9d3a88ea4f67b715d27b39733b6011`)  
**Operating principle:** a passing unit test or prior report was not accepted as independent evidence without execution or traceable artifacts.

## A. Executive verdict

| Question | Verdict | Evidence-based answer |
|---|---:|---|
| Is the project working? | **PARTIAL** | The local developer checkout has a functional FastAPI/UI product, deterministic evidence engines, and genuine local Mask2Former inference. The released archive cannot run its neural demo without separately reacquiring excluded photographs and its bundled tests are not clean-extract compatible. |
| Does it execute a neural model? | **PASS** | Yes. Actual pixels were processed by `facebook/mask2former-swin-tiny-coco-instance` on Apple MPS. Checkpoint revision and SHA-256 matched the pinned values. Raw output hashes reproduced. |
| Is its scientific evidence reliable? | **PARTIAL** | The two historical natural-photo recall values and controlled-corruption F1 reproduce. Their scope is narrow and correctly needs qualification: recall is not AP, controlled corruptions are not natural human errors, and constructed geometry is not physical-action validation. |
| Is it practically useful today? | **PARTIAL** | Useful as a local research/demo tool for comparing a supplied annotation with model evidence and emitting explicit uncertainty. It is not yet a validated annotation-defect detector or action-verification system. |
| Is it production-ready? | **FAIL** | Noncommercial model weights, unlocked runtime dependencies, unsigned receipts, absent field studies, weak calibration evidence, a non-self-contained release, and limited concurrency validation preclude production status. |
| Strongest verified contribution | **PASS** | Honest provenance categories plus a working tri-state arbitration layer around real segmentation and deterministic evidence measurements. |
| Biggest unsolved problem | **FAIL** | The system has no independently reviewed natural annotation-error benchmark and no genuine temporal physical-action dataset, so its central “causality” and annotation-quality claims remain unvalidated. |

**Bottom line:** ProofCausality AI is a credible experimental prototype with real neural inference, not a fake-model demo. Its engineering core is materially stronger than its scientific validation. It should be described as an evidence-aware local research prototype, not an independently validated causality system or production annotation-QA product.

## B. Baseline, environment, and preservation

- Canonical checkout verified at `/Users/sunny/proofcausality-ai`; initial working tree was clean.
- `main`, `origin/main`, and `origin/HEAD` all resolved to `4e2503a…` after a metadata-only fetch.
- Latest GitHub CI for that SHA was successful: run `40912089865` on 2026-10-29.
- Disk: 228 GiB volume, 178 GiB used, about 15 GiB available at audit start.
- Hardware: MacBook Air M4-class Apple Silicon, 16 GiB unified memory; MPS available.
- Runtime: Python 3.12.14, torch 2.15.0, transformers 5.21.0, NumPy 2.5.4, SciPy 1.18.1, FastAPI 0.162.0.
- Cached checkpoint: 189,998,168 bytes; no duplicate model download was performed.
- All historical artifacts, tags, releases, photographs, and reports were left intact. Fresh artifacts are under `evaluation/experiments/phase10/`.

## C. Component status matrix

| Component | Status | Executable evidence / limitation |
|---|---|---|
| Image decoding and upload validation | **WORKING** | Tests pass; 20 MiB base64 limit returned HTTP 413; PIL pixel cap is 25M. |
| Local Mask2Former provider | **WORKING** | Pilot and holdout inference executed on MPS with genuine pixels and matching checkpoint SHA. |
| Model reproducibility | **PARTIALLY WORKING** | Revision/checkpoint are pinned, but Python dependencies are lower-bound-only. Transformers 5.21 reports two missing Swin layernorm tensors newly initialized at load. |
| Masks and overlays | **WORKING** | Real browser returned a rendered disagreement view; mask dimensions matched source pixels. |
| Bounding boxes and deterministic metrics | **WORKING** | Python/Rust parity tests pass. |
| Instance matching | **PARTIALLY WORKING** | API has one-to-one matching, but the historical evaluator uses independent best-overlap per ground truth; its recall can reuse a prediction. |
| ACCEPT / REVIEW / UNCERTAIN | **PARTIALLY WORKING** | Logic executes and abstains on defined signals; thresholds are hand-set, not calibrated on an independent verdict dataset. |
| Audit receipts | **PARTIALLY WORKING** | Canonical unkeyed SHA-256 detects accidental mutation, but anyone can alter payload and recompute a valid digest. No signer/authenticated origin. |
| Authoritative dataset provenance | **WORKING** | Forged IDs, altered bytes, annotation linkage, and mappings have regression coverage. Only the 20-image pilot is in the authoritative registry. |
| LabelCourt implementation | **WORKING** | Deterministic audit and corruption benchmark execute. |
| LabelCourt scientific claim | **NOT VERIFIED** | No independently reviewed natural human-error labels. Model disagreement cannot establish annotation error. |
| ActionProof implementation | **WORKING** | Before/after geometric rules and uncertainty signals execute on metadata masks. |
| ActionProof physical validity | **NOT VERIFIED** | No genuine temporal video, identity tracking, camera-motion handling, or independent success labels. |
| Evidence Calculus | **WORKING** | 37 constructed cases reproduce exactly; predicate/debt/contradiction logic passes tests. |
| Evidence Calculus generalization | **NOT VERIFIED** | Constructed fixtures encode the same conditions consumed by the rules. |
| ReCapture planner | **WORKING** | Signal-to-directive mapping executes. |
| Human ReCapture field validation | **NOT IMPLEMENTED** | UI “re-evaluation success” and benchmark recovery are scripted paired states. |
| Meta SAM/Llama adapters | **MOCKED / BLOCKED** | Interface and deterministic fallback exist; no live Meta model call occurred. |
| NVIDIA provider | **NOT IMPLEMENTED** | No executable NVIDIA inference provider found. |
| FastAPI | **WORKING** | Local checkout and clean archive both served `/api/health`. |
| Frontend | **PARTIALLY WORKING** | All tabs worked locally; stale “Active Version v0.1.1” appears on Overview. |
| Rust acceleration | **WORKING** | 19 tests pass; acceleration is for micro-kernels, not Mask2Former. Clean release fell back to NumPy because compiled native library is excluded. |
| Concurrency | **PARTIALLY WORKING** | Provider has load and inference locks and regression coverage; no sustained load/throughput study. Requests serialize at inference. |
| Packaging/release | **BROKEN** | Clean archive: 160 tests passed, 4 failed, 24 errored; neural samples absent; `.gitattributes` test assumes a Git checkout. |
| CI | **WORKING** | Current main CI is green, but it validates the repository checkout rather than the user experience of the published archive. |

## D. Real natural-photo benchmarks

### Dataset A — historical reproduction

The exact 20 COCO val2017 images and annotation JSON were reused. Image hashes, annotation hash, checkpoint revision, and checkpoint SHA are recorded in the fresh manifest.

| Metric | Historical | Fresh | Status |
|---|---:|---:|---:|
| Images | 20 | 20 | **REPRODUCED** |
| Evaluable polygon annotations | 215 | 215 | **REPRODUCED** |
| Class-aware mean best IoU | 0.3929 | 0.3929 | **REPRODUCED** |
| Class-agnostic mean best IoU | 0.5002 | 0.5002 | **REPRODUCED** |
| Class-aware recall at IoU 0.50 | 43.26% | 43.26% | **REPRODUCED** |
| Class-agnostic recall at IoU 0.50 | 55.81% | 55.81% | **REPRODUCED** |
| Warm neural p50 | 135.8 ms | 154.2 ms | **SUPERSEDED / ENVIRONMENT-SENSITIVE** |

The output hashes matched despite runtime drift. These recall metrics select the best prediction independently for each ground-truth object. They are useful descriptive recalls but are not COCO AP and do not support precision.

### Dataset B — predeclared local holdout

Selection rule, declared before inspecting annotations: every already-preserved COCO val2017 image not present in the pilot, sorted by image ID. IDs: `139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353`.

The official COCO annotation archive was downloaded once to `/tmp`; only annotation metadata was extracted. No additional image dataset was downloaded. Quantitative matching is greedy one-to-one by descending mask IoU. These are custom metrics, not official COCO AP.

| Holdout result | Value |
|---|---:|
| Images / polygon GT instances / predictions | 13 / 102 / 135 |
| Mean / median class-aware best IoU | 0.4659 / 0.5482 |
| Mean / median class-agnostic best IoU | 0.5353 / 0.6246 |
| Class-aware precision / recall @ 0.50 | 39.26% / 51.96% |
| Class-aware precision / recall @ 0.75 | 20.74% / 27.45% |
| Class-agnostic precision / recall @ 0.50 | 43.70% / 57.84% |
| Class-agnostic precision / recall @ 0.75 | 22.96% / 30.39% |
| Small / medium / large class-aware recall @ 0.50 | 15.38% / 69.44% / 81.48% |
| Boundary F1, 2 px tolerance, class-aware IoU50 matches | 0.5745 |
| Warm neural latency p50 / p95 | 151.5 / 278.4 ms |

The dominant failure is small-object detection. At class-aware IoU 0.50 there were 82 unmatched predictions and 49 unmatched ground truths. Because these images were already present in the developer-controlled workspace and the set is only 13 images, this is an independent split, not external independent validation.

### Dataset C — real owner photos

**NOT AVAILABLE — USER PHOTOS NOT PROVIDED.** No photographs were invented or substituted.

## E. Model execution validation

- Model ID: `facebook/mask2former-swin-tiny-coco-instance`.
- Revision: `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`.
- Checkpoint SHA-256: `3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80` (**PASS**).
- Processor and checkpoint were loaded locally; device was MPS.
- Prediction API accepts RGB pixels, threshold, and optional raw bytes—no annotations, masks, or boxes—so ground-truth leakage is structurally excluded from the inference call.
- Postprocessing restored masks to original image dimensions and recorded input/output hashes.
- Current Transformers emits a load warning: Swin `layernorm.weight` and `layernorm.bias` are missing and newly initialized. The project notes this but does not pin exact Transformers/PyTorch builds or the initialized values.
- Browser end-to-end on `000000397133.jpg`: 26 predictions, `REVIEW`, IoU 0.242, verified receipt; cold full request 5,916.4 ms, including 5,409.0 ms inference and 358.6 ms postprocessing.
- Raw repeated pilot outputs matched historical hashes. A formal cross-process stochastic consistency campaign was not performed.

## F. Benchmark reconciliation

| Claim | Classification | Audit conclusion |
|---|---|---|
| Controlled-corruption LabelCourt F1 ≈ 0.9825 | **REPRODUCED** | Exact on dev and held-out splits. Reference-derived shifts/erosions/dilations/class swaps; not natural human-error accuracy. |
| Class-aware real-photo recall 43.26% | **REPRODUCED** | Correct under historical best-overlap definition. Not AP. |
| Class-agnostic real-photo recall 55.81% | **REPRODUCED** | Correct under historical best-overlap definition. Not AP. |
| Action/Evidence Calculus: 28/37 decided, 9 abstentions, risk 0.1429 | **REPRODUCED** | Correct on 12 curated geometric metadata + 25 synthetic cases. No video. |
| Evidence Calculus 0/22 false successes | **SUPPORTED BUT NARROW** | Reproduced; wide Wilson upper bound 14.87%, fixture conditions closely mirror rules, and four decided classifications are wrong. |
| ReCapture 4/4 or simulated 95% resolution | **MISLEADING IF CALLED HUMAN SUCCESS** | The 4/4 benchmark is programmed paired metadata; 95% is Monte Carlo with assumed compliance. Human field result is pending. |
| Integrated warm server p50 ≈ 1017.89 ms | **SUPPORTED BUT NOT FRESHLY RE-RUN AS A SERIES** | Historical raw samples are internally coherent. Fresh cold browser request was 5.92 s; fresh warm neural-only p50 was 0.15 s. Workloads are distinct. |
| Rust 3–11× speedup | **SUPPORTED FOR MICRO-KERNELS** | Tests and artifact structure support equivalent mask/blur kernels. It does not accelerate Mask2Former and has negligible whole-request impact. |
| “Tamper-proof” / cryptographic provenance | **INCORRECT WORDING** | Unkeyed hashes are recomputable. Use “tamper-evident consistency checksum.” |

## G. Verdict correctness and annotation-error validity

The rule order is class mismatch → high uncertainty → IoU/boundary thresholds. It behaves deterministically on unit fixtures for correct masks, shifts, class swaps, missing masks, blur, border contact, occlusion, no detections, and overlaps.

However:

- thresholds (`0.85`, `0.50`/`0.60`, `0.75`, blur `100`) are manual;
- no independent reviewer supplied “annotation correct/incorrect/insufficient” labels;
- natural photographs are evaluated for segmentation overlap, not for human annotation mistakes;
- the final holdout was not used for threshold tuning during this audit, but historical development choices may have seen the same pilot ecosystem;
- false ACCEPT, false REVIEW, selective risk, and decision coverage cannot be scientifically estimated for authentic annotation errors without reviewer ground truth.

**REAL HUMAN ANNOTATION-ERROR BENCHMARK: NOT VERIFIED.**

## H. ActionProof, Evidence Calculus, and ReCapture

ActionProof computes centroid displacement, region containment, blur, border contact, area drop, and missing-final-state signals from supplied masks/boxes. It does not establish object identity or temporal continuity. Camera motion, occlusion recovery, track identity, and success/failure labels from video are absent.

The 37-case Evidence Calculus experiment is reproducible and useful as a rules-engine regression benchmark. It is not independent physical evidence because the fixture metadata directly supplies the variables the engine tests.

ReCapture maps known uncertainty reasons to fixed directives. The UI presents a polished three-step story ending in “SUCCESS,” but those follow-up measurements are scripted. No participant actually followed an instruction during this audit or in the preserved evidence.

**GENUINE TEMPORAL ACTION VALIDATION: PENDING.**  
**HUMAN RECAPTURE FIELD VALIDATION: PENDING.**

## I. Security, provenance, and receipts

| Test | Result |
|---|---:|
| Altered receipt without rehash | **PASS — rejected** |
| Altered receipt with attacker-recomputed SHA-256 | **FAIL — accepted** |
| Oversized base64 payload | **PASS — HTTP 413** |
| Unknown/path-shaped receipt | **PASS — 404 / route rejection** |
| Modified image / forged manifest ID / forged annotation linkage | **PASS — regression tests** |
| Checkpoint hash mismatch | **PASS — fail closed** |
| Receipt storage bound | **PASS — 1,000-entry in-memory LRU** |
| Authenticated origin / digital signature | **NOT IMPLEMENTED** |

The CORS policy permits `*` origins while also setting credentials true; browsers do not allow wildcard credential sharing as intended, but the policy remains unsuitable for a production trust boundary. The in-memory receipt ledger is process-local, unauthenticated, non-durable, and lost on restart.

For authenticated integrity, use a private signing key (for example Ed25519), publish key identity/rotation rules, bind dataset/model manifests to signed release attestations, and persist append-only receipts in an access-controlled store.

## J. Product and release audit

Local browser validation:

- landing page and all nine navigation tabs rendered;
- neural inference, LabelCourt, ActionProof, Evidence Calculus, and ReCapture controls executed;
- result overlays and receipt ledger rendered;
- model-unavailable status was honestly shown for Meta;
- no application-origin console error was observed (extension warnings were unrelated);
- responsive override showed no horizontal overflow, but exact 375 CSS-pixel verification was **PARTIAL** because the controlled Chrome surface reported a doubled inner width and hid navigation.

Release archive validation:

- archive hashes match the GitHub release assets;
- archive excludes COCO photographs as intended for licensing safety;
- `/api/health` starts successfully with the core dependencies;
- neural sample endpoint is empty, so the advertised demo cannot execute in a fresh archive;
- bundled tests: **160 passed, 4 failed, 24 errored**;
- failures are caused by omitted photos, omitted `.gitattributes`, and a test invoking `git archive` without `.git`;
- the clean archive used NumPy fallback because the native compiled library is excluded;
- the downloader depends on `/tmp/annotations_trainval2017.zip` or `/tmp/instances_val2017.json`; without either it attempts to open a missing ZIP rather than downloading annotations;
- the downloader contains outdated broad licensing defaults; the later licensing audit correctly contradicts those defaults.

The Overview’s diagnostic card says `Active Version v0.1.1` while API/engine show v0.2.6. This is a confirmed presentation defect.

## K. Comparison with established approaches

| Approach | Comparable purpose | What it establishes | Relative assessment |
|---|---|---|---|
| Standalone Mask2Former | Instance segmentation | Standard masks/classes/scores; paper reports COCO AP under official protocol | ProofCausality uses this perception model but has not executed official COCO AP, so no accuracy superiority claim is possible. |
| COCO `pycocotools` | Standard evaluation | AP/AP50/AP75, area ranges, max detections, per-image/category matching | More standardized than the project’s best-overlap recall. Phase 10 custom holdout adds one-to-one precision/recall but is still not official AP. |
| Cleanlab Object Detection | Label-quality ranking | Scores likely overlooked boxes, bad localization, and class swaps from labels + predictions | A direct annotation-QA baseline. Not executed here; LabelCourt lacks an authentic error-labelled comparison. |
| FiftyOne mistakenness | Dataset curation/review | Surfaces possible missing, spurious, and localization issues and routes them to human review | More mature review workflow; it also treats outputs as possible mistakes requiring review, matching the caution this project needs. |
| Human double review/adjudication | Ground-truth creation | Independent correction labels and inter-reviewer agreement | Still required to validate LabelCourt. |
| Action/video tracking baselines | Temporal action evidence | Identity, continuity, trajectories, and outcomes over time | ProofCausality currently consumes before/after metadata only and cannot be fairly ranked. |

Primary sources: [Mask2Former paper](https://arxiv.org/abs/2112.01527), [official COCO evaluator](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py), [Cleanlab object-detection API](https://docs.cleanlab.ai/stable/cleanlab/object_detection/index.html), and [FiftyOne detection-mistake workflow](https://docs.voxel51.com/getting_started/object_detection/03_finding_mistakes.html).

## L. Resource profile

| Measurement | Result | Scope |
|---|---:|---|
| Model checkpoint | 181.2 MiB | Cached safetensors only |
| Full project checkout | 1.3 GiB | Includes 1.2 GiB virtual environment |
| Python test wall / max RSS | 19.64 s / 0.91 GiB | 188 tests |
| Historical reproduction wall / peak footprint | 13.61 s / 1.96 GiB | 20 images, one model load |
| Holdout wall / peak footprint | 10.72 s / 2.11 GiB | 13 images, one model load |
| Holdout warm neural p50 / p95 | 151.5 / 278.4 ms | Forward pass timing in provider |
| Fresh browser cold full request | 5,916.4 ms | Decode through receipt, local HTTP |
| Historical warm full TestClient p50 | 1,020.81 ms | In-process ASGI, not network browser RTT |
| Concurrent behavior | **PARTIAL** | Singleton + serialization locks; no throughput/load curve |

MPS cache clearing after each prediction may reduce memory retention but can distort throughput. CPU fallback was not freshly benchmarked because duplicating the full 33-image run would add little evidence under the storage/time constraint. Rust speedups apply to micro-kernels only.

## M. Scientific risk register

| Priority | Risk | Consequence |
|---|---|---|
| P0 | No authentic annotation-error ground truth | Central LabelCourt accuracy claim cannot be estimated. |
| P0 | No temporal action/video benchmark | “Causality” and physical completion claims remain conceptual. |
| P0 | Unsigned, recomputable receipts | No trusted author/origin; forged payloads can be made internally consistent. |
| P1 | Non-self-contained release | Fresh users cannot execute the flagship demo or all tests. |
| P1 | Runtime dependency drift + newly initialized tensors | Exact model state is not fully reproducible from repository metadata. |
| P1 | Hand-tuned thresholds | Selective risk/coverage may not generalize. |
| P1 | Very small developer-controlled photo sets | Wide uncertainty and selection bias. |
| P1 | Noncommercial checkpoint license | Blocks ordinary commercial production use. |
| P2 | Historical evaluator is not bipartite | Recall can over-credit reused predictions. |
| P2 | UI scripted recovery presented as success | Users may mistake simulation for field evidence. |
| P2 | Version drift in UI and receipts (`0.1.1`, `0.2.3`, `0.2.6`) | Weakens traceability and user confidence. |

## N. Completion assessment

Scoring is transparent and deliberately separate by domain. Each listed requirement is equally weighted: `PASS=1`, `PARTIAL=0.5`, and `FAIL/BLOCKED/NOT VERIFIED=0`. These are audit-scope completion scores, not probabilities of correctness.

| Domain | PASS | PARTIAL | Zero-credit | Score | Requirements counted |
|---|---:|---:|---:|---:|---:|
| Engineering implementation | 8 | 4 | 0 | **83.3%** | 12 |
| Functional product readiness | 5 | 1 | 4 | **55.0%** | 10 |
| Scientific validation | 3 | 1 | 6 | **35.0%** | 10 |
| Real-world deployment readiness | 2 | 1 | 5 | **31.3%** | 8 |

Engineering denominator: input validation, local inference, matching, decisions, overlays/receipts, LabelCourt, ActionProof, Evidence Calculus, ReCapture, providers, Rust, concurrency.  
Product denominator: app startup, navigation, upload, honest model status, overlays, errors, mobile, clean archive, archive demo, version consistency.  
Science denominator: pilot reproduction, independent holdout, official AP, natural human errors, calibration, temporal action, human recapture, leakage control, grouped uncertainty, fair external baseline.  
Deployment denominator: clean install, licensing, security controls, authenticated receipts, performance, resource fit, commercial model rights, release verification.

No single blended percentage is reported because engineering implementation and scientific validity are not interchangeable.

## O. Priority roadmap with acceptance criteria

1. **Build an independent human annotation-error benchmark.** Freeze 200–500 natural images before model review; obtain two blinded reviewers plus adjudication; record correct/missing/wrong-class/wrong-mask/insufficient-evidence labels; evaluate LabelCourt and at least Cleanlab/FiftyOne-style baselines with image-grouped intervals and a locked threshold.
2. **Make release and inference reproducible.** Add a lock file with exact hashes; resolve the two missing layernorm tensors or pin a known-good Transformers build; package a license-safe first-run sample or a robust downloader; make clean-extract tests skip/acquire optional media explicitly; verify a fresh environment and CPU/MPS paths in CI.
3. **Validate physical evidence and authenticated provenance.** Collect independently labelled continuous before/during/after videos with camera motion, occlusion, identity switches, and failures; measure risk/coverage; run a real ReCapture user study; replace bare hashes with signed receipts and signed model/dataset manifests.

## P. Artifact index

- `evaluation/experiments/phase10/historical_reproduction/summary.json`
- `evaluation/experiments/phase10/holdout/summary.json`
- `evaluation/experiments/phase10/holdout/dataset_manifest.json`
- `evaluation/experiments/phase10/labelcourt_controlled_corruption_reproduction.json`
- `evaluation/experiments/phase10/h01_reproduction.json`
- `evaluation/experiments/phase10/actionproof_12case_reproduction/benchmark_results_real.json`
- `evaluation/experiments/phase10/security_negative_tests.json`

## Final acceptance status

The repository, release, tests, model, historical benchmark, distinct natural-photo sample, component logic, provenance/security behavior, browser workflow, runtime profile, and relevant external baselines were audited. The report explicitly retains `BLOCKED`, `PENDING`, and `NOT VERIFIED` findings. No source fix, release mutation, tag change, merge, or publication was performed.

