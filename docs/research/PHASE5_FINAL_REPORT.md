# Phase 5 research and engineering report

Status vocabulary: **PASS / FAIL / BLOCKED / NOT VERIFIED**.

1. **Current source commit — PASS:** baseline commit `0a4eb4d3e5f8809c9da34849285c3eb2e09a826c`; Phase 5 changes are on `phase5-genuine-inference-provenance-audit` and are intentionally unreleased.
2. **Files changed — PASS:** local inference provider, manifest validator, grouped statistics, two benchmark paths, provenance/API labels, tests, README, generated prediction bundle, and this audit documentation.
3. **Confirmed defects repaired — PASS:** decoded bytes no longer become verified photographic provenance; model provenance requires a successful execution record; reference annotations are excluded from the prediction API; corruption statistics now resample by image; README categories are separated.
4. **Tests — PASS/PARTIAL:** baseline 107 Python tests passed. Phase 5 adds provenance, manifest, RLE, leakage-interface, grouped-bootstrap, and split-overlap regressions. Rust compiled but still had zero native tests at baseline.
5. **Genuine model — PASS:** `facebook/mask2former-swin-tiny-coco-instance`, revision `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`, local PyTorch on Apple MPS, unprompted.
6. **License/checkpoint — PASS with limitation:** checkpoint 189,998,168 bytes, SHA-256 `3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80`. Upstream model zoo states CC BY-NC 4.0 for weights; implementation is mostly MIT with Apache-2.0 components; Hugging Face card says only `other`.
7. **Unique photographs — PASS:** 20 independent COCO source images.
8. **Dataset manifest — PASS/PARTIAL:** hashes, dimensions, canonical references, recorded licenses, annotation hash, and acquisition procedure are preserved. Redistribution status is `UNKNOWN`, not silently approved.
9. **Annotation linkage — PASS:** 215 polygon annotations were evaluable; three of 218 records did not yield an evaluable polygon mask and are not silently included in the denominator.
10. **Independent inference — PASS:** 20/20 image executions succeeded; ground truth was unavailable to `predict()` and entered only after inference.
11. **Controlled corruption — PASS, scoped:** held-out (10 images, 119 annotations, 714 derived scenarios) precision 0.9752, recall 0.9899, F1 0.9825. Image-grouped intervals: precision [0.9627, 0.9880], recall [0.9802, 0.9966], F1 [0.9744, 0.9894].
12. **Genuine prediction benchmark — PASS:** class-agnostic mean best IoU 0.5002; class-aware mean best IoU 0.3929; class-agnostic recall@0.5 0.5581; class-aware recall@0.5 0.4326.
13. **Grouped uncertainty — PASS:** image-level bootstrap, 2,000 resamples, seed `20261028`; genuine class-aware recall@0.5 interval [0.3333, 0.5429]. Only 20 images makes all inference estimates preliminary.
14. **Baseline comparison — PARTIAL:** the geometry and LabelCourt corruption baselines share the controlled target. Genuine Mask2Former mask quality is reported separately; no invalid cross-task superiority claim is made.
15. **Failure gallery — PASS:** 12 actual COCO evaluation failures are preserved in `evaluation/experiments/genuine_model_predictions/failure_gallery/` with IDs, masks, categories, decisions, and overlays.
16. **Performance — PARTIAL:** final genuine inference run (20 images) p50 134.382 ms, p95 198.759 ms after model load, with a first recorded inference of 997.324 ms. Historical Rust/NumPy results are kernel-only and are not presented as application speedup. End-to-end API/UI timing including decode and model load remains NOT VERIFIED.
17. **Receipts — PASS:** finalized payload digest verification and mutations of measurements, provenance, and verdict are covered. No Merkle-tree claim is made.
18. **ReCapture — PARTIAL:** media and metadata-only modes are distinct; decoded follow-up media is now `DECODED_IMAGE`, not verified photographic origin. Physical task completion remains NOT VERIFIED without task-specific temporal measurement.
19. **Live Meta inference — BLOCKED:** no authorized endpoint/credentials; no paid call attempted. Local Mask2Former is not represented as official live Meta inference.
20. **UI/browser — NOT VERIFIED until the final browser pass is recorded.**
21. **README/diagrams — PASS/PARTIAL:** benchmark claims and reproduction commands corrected. Existing diagrams retained because no measured architectural change required a redraw.
22. **GitHub CI — NOT VERIFIED:** local commands can be run, but remote CI is not triggered because the branch is not pushed.
23. **Limitations — PASS:** small independent sample, noncommercial model weights, converted-checkpoint layer-norm load warning, no temporal action benchmark, no human recapture study, no live Meta inference, and no fair external SAM/Cleanlab comparison.
24. **Release decision — FAIL / not justified:** no `v0.2.5` should be created until the checkpoint compatibility warning, native parity tests, full end-to-end timing, and remote CI are resolved.

## Evidence categories

| Category | Result |
|---|---|
| `CONTROLLED_CORRUPTION_ON_REAL_PHOTOS` | MEASURED |
| `GENUINE_MODEL_PREDICTIONS` | MEASURED |
| `REAL_TEMPORAL_ACTION` | NOT MEASURED |
| `SIMULATED_RECAPTURE` | MEASURED only in existing fixtures/simulations |
| `ACTUAL_HUMAN_RECAPTURE` | NOT MEASURED |
| `LIVE_META_INFERENCE` | BLOCKED |

