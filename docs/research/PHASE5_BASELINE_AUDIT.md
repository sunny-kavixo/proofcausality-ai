# Phase 5 baseline audit

Baseline: `main` / `v0.2.4` / `0a4eb4d3e5f8809c9da34849285c3eb2e09a826c`, synchronized with `origin/main` on 2026-10-28.

| Area | Baseline finding | Status |
|---|---|---|
| Python tests | 107 tests passed under Python 3.14.5; one Starlette deprecation warning. | PASS |
| Rust tests | Crate compiled, but contained zero unit or doc tests. | NOT VERIFIED |
| Working tree | Clean before branch creation. | PASS |
| COCO data | 20 manifest entries, 218 annotations; 33 local JPEGs exist, but only the manifest-linked 20 are in this experiment. | PASS |
| Existing COCO benchmark | Candidate masks are derived from reference masks via controlled corruption. | PASS as controlled corruption; FAIL as independent perception |
| Provenance gate | A decodable image could be labeled `REAL_PHOTOGRAPH`; no manifest proof was required. | FAIL |
| Receipts | Canonical SHA-256 covers finalized payload fields; mutation tests exist. This is a digest, not a Merkle tree. | PASS |
| Model adapters | Hosted Meta adapter and geometric fallback exist. No live Meta credentials or endpoint were configured. | BLOCKED |
| Local models | PyTorch, Transformers, ONNX Runtime and Apple MPS were available; no segmentation checkpoint was cached. | PASS |
| Resources | 16 GiB RAM; approximately 17 GiB free disk; Hugging Face cache approximately 1.8 GiB. | PASS with constraint |
| README claims | Corruption results were presented too broadly as a real-world photographic benchmark. | FAIL |
| CI | Builds Rust, runs Python tests and historical benchmarks; no genuine model checkpoint job. | PARTIAL |

The baseline records were preserved. Historical experimental outputs were updated only by the reproducible benchmark command and remain separately categorized from genuine model inference.

