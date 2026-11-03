# PR #7 Final Pre-Merge Review

**Review date:** 2026-10-31  
**Pull request:** https://github.com/sunny-kavixo/proofcausality-ai/pull/7  
**Branch:** `fix/phase11a-release-reproducibility`  
**Disposition:** ready for owner review after CI; not merged and no release published

## Findings and changes

### Genuine clean-install gate

The initial Phase 11A validator built a wheel correctly but created its verification
environment with `--system-site-packages` and installed the wheel with `--no-deps`.
That proved wheel structure, not runtime dependency completeness.

The validator now:

1. creates a standard isolated virtual environment with no inherited site packages;
2. installs the built wheel normally, resolving every declared runtime dependency;
3. imports `fastapi`, `httpx`, `numpy`, `Pillow`, `pydantic`, `scipy`, and `uvicorn`;
4. imports the installed backend and packaged frontend outside the source directory;
5. verifies the packaged `frontend/static/index.html`;
6. calls `/api/health` through `TestClient` and checks the package version; and
7. retains the Rust `cargo check` gate.

A regression test rejects reintroduction of `--system-site-packages` or dependency-free
wheel installation.

### Failure-gallery licensing

All 12 new failure-gallery PNGs are adaptations of the same COCO val2017 photograph,
image ID `397133`. Flickr's metadata identifies it as:

- title: **Kitchen**;
- creator: **Pot Noodle** (`maggiew`);
- source: https://www.flickr.com/photos/maggiew/6255196340/;
- license: **CC BY 2.0**;
- modifications: COCO-preserved sizing plus reference/prediction mask, label, and
  diagnostic-text overlays.

The underlying photograph may be redistributed and adapted only with appropriate
creator/source/license credit and an indication of changes. COCO annotation geometry
also carries CC BY 4.0 attribution obligations. The model checkpoint is documented as
CC BY-NC 4.0; the project does not assert that generated overlays erase upstream model
obligations. The gallery is therefore retained as noncommercial research/audit evidence,
not MIT-licensed project artwork. `failure_gallery/ATTRIBUTION.md` records these terms,
and a regression test requires the notice while all 12 PNGs are present.

### Optional-test skip audit

The clean archive now reports **167 passed, 26 skipped**. The skips are narrowly grouped:

- 2 require excluded local COCO image files for disk-hash/historical inventory checks;
- 8 require genuine COCO bytes (7 manifest-backed provenance cases and 1 sample-schema case);
- 14 require optional PyTorch/Transformers plus the pinned local checkpoint (some also
  exercise the genuine-photo workflow when the model is available);
- 2 are release-maintainer checks requiring `.git` or `.gitattributes` in a checkout.

No core invalid-input, oversized-payload, provider-unavailable, sanitized-error,
failed-receipt, downloader-safety, health, package-import, or deterministic-engine test is
hidden by these skips. Two previously photo-coupled core API tests now use a synthetic
user-upload fallback and execute in the clean archive.

### Genuine-model evidence and CI status

The `Genuine Mask2Former Model Inference Smoke Test` CI job is intentionally conditional:
it runs only for manual workflow dispatch or when a PR has the `run-model-ci` label. PR #7
does not carry that label, so the job is expected to show **skipped**, not silently pass.

Existing local evidence remains preserved:

- model `facebook/mask2former-swin-tiny-coco-instance` at immutable revision
  `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`;
- checkpoint SHA-256
  `3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80`;
- Apple MPS smoke test on 2026-10-29: 26 detected instances and output SHA-256
  `a5c4d915157d12e4eed08167d79ad35a4451d2b1dca4f6ac3cbddb759d45bb21`;
- Phase 10 unprompted inference over the historical 20 images and predeclared 13-image
  diagnostic holdout, with execution records, input/output hashes, and latency artifacts.

This is genuine local inference evidence, but not a fresh hosted Linux/CPU model run for
this PR and not external scientific validation.

## Verification

- targeted packaging/licensing/downloader suite: **13 passed**;
- enriched local suite: **193 passed**;
- committed clean source archive: **167 passed, 26 justified skips**;
- isolated wheel dependency install, packaged frontend import, `/api/health`, and Cargo
  check: **PASS**.

The final GitHub CI run and exact final head SHA are reported on PR #7 and in the owner
handoff after this review commit is pushed.

## Remaining risks

- Runtime requirements have compatibility ceilings but no fully hashed cross-platform lock.
- The genuine-model CI job remains opt-in because it downloads a large noncommercial
  checkpoint and is not part of ordinary PR CI.
- The source archive deliberately omits COCO photographs, so genuine-photo integration
  remains an explicit acquisition path.
- Failure-gallery redistribution depends on preserving its attribution notice; commercial
  use is not cleared by this review.
- Scientific limitations from Phase 10 remain: no external validation, natural
  annotation-error adjudication, temporal action dataset, or human ReCapture study.
