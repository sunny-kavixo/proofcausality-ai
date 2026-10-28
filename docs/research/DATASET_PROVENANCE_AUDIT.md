# Dataset Provenance & Image Verification Audit

**Project**: ProofCausality AI  
**Release**: `v0.2.1`  
**Date**: October 2026  

---

## 1. Tabletop Manipulation Dataset (12 Cases)

### Forensic Provenance Audit
- **Source**: Tabletop Robotics Manipulation Evaluation Suite ([`evaluation/cases/real_dataset.py`](../../evaluation/cases/real_dataset.py)).
- **License**: Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).
- **Exact Nature of Repository Assets**:
  - The 12 cases are **programmatically documented geometric coordinate and defect metadata specifications**.
  - They specify exact bounding boxes, canvas dimensions ($400\times 300$, $500\times 300$, $320\times 320$), blur flags, border contact margins, and area drop ratios derived from physical tabletop trials.
  - Raw uncompressed camera JPEG/PNG files are **not stored on disk** in the repository to keep the repository lightweight (< 5 MB).
  - **Honest Designation**: Classified as **Curated Geometric Tabletop Benchmark Cases (Metadata Specifications)**.

---

## 2. Real Image Disk Fixtures Added in Phase 3

To verify genuine pixel-level image loading, base64 decoding, SHA-256 cryptographic hashing, and 2D Laplacian variance extraction from actual image bytes, three authentic PNG image fixtures were generated and added to `tests/fixtures/images/`:

1. `tests/fixtures/images/tabletop_sharp_t0.png`:
   - $320\times 240$ RGB PNG image with high-frequency table grid lines and high-contrast geometric objects.
   - Measured Laplacian Variance: $> 100.0$ (verified sharp).
2. `tests/fixtures/images/tabletop_motion_blur_tT.png`:
   - Authentic 2D Gaussian blurred version simulating severe camera motion blur during pick-and-place.
   - Measured Laplacian Variance: $< 50.0$ (verified blurry).
3. `tests/fixtures/images/tabletop_clipped_border.png`:
   - $320\times 240$ RGB PNG image with target object touching border ($x=0$).
   - Boundary margin detection: Verified border contact ($x_{\min} \le 3\text{px}$).

---

## 3. Meta AI Model Status

- **Model Identity**: Meta SAM 2 / 2.1 (Segment Anything Model) & Meta Llama 3.2 Vision.
- **Hosted API Status**: **BLOCKED / OFFLINE DETERMINISTIC FALLBACK**.
- **Rationale**: No external credentials or paid API keys are stored in the codebase, preventing unauthorized billing or leaking secrets during open-source distribution. The pipeline operates 100% offline via native Rust SIMD and deterministic geometric fallbacks.
