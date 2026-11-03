# ProofLoop Model and Policy Comparison

**Evaluation status:** Partially measured. The local Mask2Former comparison is measured on the v0.2.7 COCO diagnostic set. The ProofLoop comparison is measured on controlled feature fixtures. Live Meta inference, human recapture effectiveness, and causal video validation are blocked.

## Components compared

| Component | What it contributes | Evidence status |
|---|---|---|
| Mask2Former alone | Unprompted local instance predictions from image pixels | Measured: 332 GT, 366 predictions; taxonomy-aware recall 171/332 (51.51%) at IoU .50 and 110/332 (33.13%) at IoU .75 |
| LabelCourt | Deterministic annotation-to-prediction geometry and boundary audit | Measured on controlled corruptions; not an independent human-error accuracy study |
| Meta SAM adapter | Optional provider boundary and mask parser | Contract and offline fallback tested; live authenticated request blocked |
| Evidence Calculus | Predicate decomposition, Evidence Debt, and abstention | Measured on the existing 37-case constructed benchmark; 0/22 false successes at 75.68% coverage |
| ProofLoop | Explicit closed-loop state trace, semantic mapping, and targeted follow-up request | Measured on 10 controlled feature fixtures; 6/7 initial uncertainties resolved, 1/7 unresolved |
| ProofLoop + recovery | Re-evaluates new evidence rather than treating the directive as success | Same controlled benchmark; resolution is based on supplied metadata, not a human response study |

## Interpretation

ProofLoop does not replace Mask2Former, SAM, or LabelCourt. Perception supplies observations; Evidence Calculus measures observable deficits; ProofLoop decides whether another observation is warranted and records the transition. The differentiating mechanism is evidence closure and abstention semantics, not a new foundation model.

The 51.51% COCO recall is a limitation of the current local perception provider, not a ProofLoop accuracy result. The 85.71% controlled resolution rate is likewise a policy-fixture result and must not be presented as field performance.

## Reproduction

```bash
.venv/bin/python -m pytest tests/test_proofloop.py -q
.venv/bin/python -m evaluation.benchmark_proofloop \
  --output docs/benchmarks/proofloop_closed_loop_report.json
```

## Missing comparisons

The following rows remain `BLOCKED` until authorized data or credentials exist:

- live Meta SAM 3.1 inference;
- a learned calibrated verifier trained on independent labels;
- human recapture response and effort;
- authentic temporal video tracking and intervention-controlled causal attribution.
