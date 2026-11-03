# ProofLoop Closed-Loop Evidence Verifier

**Status:** Implemented as a deterministic policy layer in v0.2.7 development work. The benchmark below is a controlled metadata experiment; it is not human validation, perception validation, or proof of physical causality.

## User-visible mechanism

ProofLoop adds a typed state machine around the existing Evidence Calculus engine:

```text
OBSERVED -> ANALYZED -> UNCERTAIN -> OBSERVATION_REQUESTED
                                     -> NEW_EVIDENCE -> REVERIFIED -> FINAL_VERDICT
```

When an observation is decisive, the path is `OBSERVED -> ANALYZED -> FINAL_VERDICT`. When evidence is incomplete, the verifier:

1. preserves the legacy `SUCCESS` / `FAILED` / `UNCERTAIN` ActionProof contract;
2. maps it explicitly to `SUPPORTED` / `CONTRADICTED` / `INSUFFICIENT_EVIDENCE`;
3. exposes the Evidence Debt and predicate rationale;
4. selects a targeted next observation through the existing Evidence Budget Planner; and
5. accepts one follow-up observation, re-evaluates it, and records the complete state trace in a SHA-256 receipt.

The semantic mapping is:

| Legacy result | ProofLoop meaning |
|---|---|
| `SUCCESS` | `SUPPORTED` |
| `FAILED` | `CONTRADICTED` |
| `UNCERTAIN` | `INSUFFICIENT_EVIDENCE` |

An optional supplied class-agreement flag can explicitly contradict a geometrically plausible claim. An optional upstream model-confidence value can cause abstention without converting uncertainty into contradiction. Neither input is fabricated by ProofLoop.

## API

### Start an evaluation

```bash
curl -s http://127.0.0.1:8000/api/proofloop/start \
  -H 'content-type: application/json' \
  -d '{
    "task_id":"DEMO_001",
    "primary_object":"cup",
    "before_bbox":{"x_min":50,"y_min":50,"x_max":100,"y_max":100},
    "after_bbox":{"x_min":310,"y_min":110,"x_max":390,"y_max":190},
    "target_region":{"x_min":300,"y_min":100,"x_max":400,"y_max":200},
    "initial_region":{"x_min":20,"y_min":20,"x_max":120,"y_max":120},
    "laplacian_variance":20.0,
    "has_final_state":true
  }'
```

The response includes `session_id`, `state_trace`, both verdict vocabularies, `observation_request`, predicate measurements, and a verified receipt. The current endpoint accepts measured features rather than media bytes and reports `provenance_status: METADATA_ONLY`.

### Reverify one follow-up observation

```bash
curl -s http://127.0.0.1:8000/api/proofloop/reverify \
  -H 'content-type: application/json' \
  -d '{
    "session_id":"PLS-<ID>",
    "after_bbox":{"x_min":310,"y_min":110,"x_max":390,"y_max":190},
    "laplacian_variance":250.0,
    "area_drop_ratio":0.0,
    "has_final_state":true
  }'
```

Sessions are bounded in memory and receipts use the existing unkeyed canonical SHA-256 integrity checksum. The checksum detects payload tampering; it does not prove source authenticity or physical truth.

## Controlled benchmark

Reproduce the benchmark with:

```bash
.venv/bin/python -m evaluation.benchmark_proofloop \
  --output docs/benchmarks/proofloop_closed_loop_report.json
```

The frozen report contains 10 cases covering missing final state, blur, border clipping, partial occlusion, class mismatch, incorrect destination claims, ambiguous confidence, successful recovery, unresolved recovery, and clean success. Current measured output:

- 7 cases initially abstained;
- 6 of those 7 were resolved by the supplied follow-up feature set;
- 1 remained `INSUFFICIENT_EVIDENCE` after follow-up;
- class mismatch and incorrect destination cases were `CONTRADICTED` rather than accepted;
- resolution rate on these controlled fixtures: **85.71% (6/7)**.

These numbers measure deterministic policy behavior on authored fixtures. They do not estimate real-world recapture success, annotator agreement, model accuracy, or causal validity.

The same flow was exercised in Google Chrome through the real FastAPI UI with zero console errors; the acceptance record and genuine screenshot are [`browser_proofloop_acceptance_report.json`](../benchmarks/browser_proofloop_acceptance_report.json) and [`browser_proofloop_closed_loop.png`](../benchmarks/browser_proofloop_closed_loop.png).

## What remains outside the claim

- The current ProofLoop endpoint does not independently decode or authenticate image/video media.
- No independent human recapture study has been conducted.
- No intervention-controlled temporal causal study has been conducted.
- No learned ProofLoop verifier is claimed. Available independent labels are insufficient for a defensible train/validation/test calibration model; the deterministic baseline is retained.
- Hosted Meta inference remains blocked unless the owner supplies and authorizes configured credentials.
