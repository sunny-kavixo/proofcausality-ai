# Real-Media ProofLoop Path

**Status:** Implemented and verified on a genuinely different, manifest-verified COCO photo pair. Real action-pair and temporal recovery validation remain unverified.

## Trust boundary

The media endpoints are:

- `POST /api/proofloop/media/start`
- `POST /api/proofloop/media/reverify`

The client supplies image bytes and the task claim only. The server:

1. decodes the image and computes its SHA-256 hash;
2. verifies manifest provenance when `MANIFEST_VERIFIED_DATASET_IMAGE` is claimed;
3. runs the pinned local Mask2Former checkpoint on original pixels;
4. selects a server-side category-matched instance and derives its bounding box, confidence, blur measurement, and model output hash;
5. passes those measurements into ProofLoop; and
6. binds initial/follow-up image hashes, model output hashes, checkpoint identity, and the state trace into the receipt.

Client-supplied bounding boxes or metrics are not accepted as measurements. In metadata-only mode, the original ProofLoop endpoints remain explicitly labeled `METADATA_ONLY`.

## Verified smoke result

The real COCO pilot image `000000397133.jpg` (`manifest_image_id=397133`) was sent through both endpoints:

- image SHA-256: `09e1d25c75f7879bdaa69c327fece5cabacd53939c8c2ef9e87f1c97a2e478c4`;
- server-derived Mask2Former output: 26 instances, matching `bottle` score `0.6783`, server bbox `[125,253,139,274]`;
- Laplacian variance: `2083.53`;
- model revision: `22c4a2f15dc88149b8b8d9f4d42c54431fbd66f6`;
- checkpoint SHA-256: `3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80`;
- initial verdict: `INSUFFICIENT_EVIDENCE` with `OBSERVATION_REQUESTED`;
- same-image follow-up: `CONTRADICTED` because displacement was zero;
- receipt integrity: verified.

Machine-readable evidence is [`proofloop_media_smoke_report.json`](../benchmarks/proofloop_media_smoke_report.json). This demonstrates real pixel-to-verdict binding and a safe negative control. It does not demonstrate temporal action completion or real recovery effectiveness.

## Different-photo pair experiment

Reproduce the real pair experiment with:

```bash
.venv/bin/python -m evaluation.benchmark_proofloop_media \
  --output docs/benchmarks/proofloop_real_media_different_photo_report.json
```

The default pair is:

- before: COCO image `174482`, `000000174482.jpg`, `640x388`, Attribution-NoDerivs, restricted redistribution;
- follow-up: COCO image `87038`, `000000087038.jpg`, `640x480`, Attribution-NonCommercial-ShareAlike, restricted redistribution.

Both records are manifest-verified and have documented source URLs/license terms. The server measured different image hashes, ran the pinned model on both images, and normalized the follow-up bbox from `640x480` into the initial `640x388` coordinate frame.

The executed cases produced:

| Case | Follow-up semantic verdict | Server-derived displacement | Interpretation |
|---|---|---:|---|
| Different photo, broad target | `SUPPORTED` | 78.013 px | Observed image-space state difference under the policy |
| Different photo, target claim contradiction | `CONTRADICTED` | 78.013 px | Same measured evidence rejected under a narrower target claim |
| Different photo, rest proof required | `INSUFFICIENT_EVIDENCE` | 78.013 px | Position observed, but rest/release evidence remains unavailable |

All three receipts verified, both input hashes differed, and client measurement fields were rejected. These are real-photo observation experiments, not temporal action validation: two separate photographs do not prove that an object moved between captures or that any intervention caused the observed difference. The complete report is [`proofloop_real_media_different_photo_report.json`](../benchmarks/proofloop_real_media_different_photo_report.json).

## Research benchmark boundary

The repository does not yet contain an independently recorded, rights-cleared before/during/after action dataset with ground-truth outcomes. Therefore no real-media resolution, false-accept, or temporal accuracy rate is reported. The existing 10-case `85.71%` result remains a controlled metadata-fixture benchmark only.
