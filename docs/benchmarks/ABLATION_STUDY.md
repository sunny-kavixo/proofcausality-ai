# ProofLoop Ablation Study

**Status:** Controlled policy ablation executed; learned-model ablation not executed because there is no independent labeled calibration set.

## Pre-declared factors

The ProofLoop policy has four material decision inputs:

1. Evidence Calculus predicate/debt evaluation;
2. targeted recovery action selection;
3. explicit low-confidence abstention;
4. explicit claim-class mismatch contradiction.

## Executed ablation design

The benchmark report `docs/benchmarks/proofloop_closed_loop_report.json` contains 10 fixed cases. For each case, the initial and follow-up observations are authored before execution and passed through the same verifier. The following factors are isolated in the case set:

| Ablation | Case evidence | Expected distinction |
|---|---|---|
| Remove recovery step | `missing_final_state`, `blurred_object`, `partial_occlusion` initial rows | Initial result stays `INSUFFICIENT_EVIDENCE`; no claim of resolution is allowed |
| Add sharpness evidence | `blurred_object` follow-up | `UNCERTAIN` → `SUPPORTED` when blur is removed |
| Add final-state evidence | `missing_final_state` follow-up | Missing final observation is resolved only after an observed final state |
| Add visibility evidence | `partial_occlusion` follow-up | Area-drop deficit resolves only after measured surface retention |
| Add confidence gate | `ambiguous_model_output` | Low confidence abstains despite otherwise sufficient geometry |
| Add claim consistency gate | `class_mismatch` | Geometrically plausible claim becomes `CONTRADICTED` |
| Keep deficit after recovery | `unresolved_after_recapture` | Follow-up remains `INSUFFICIENT_EVIDENCE`; no forced success |

Current result: 7 initial uncertainties, 6 resolved and 1 unresolved (**85.71%** controlled resolution). This is a deterministic implementation check, not a statistical estimate.

## Not executed

A feature ablation for a trained verifier (confidence, overlap, boundary disagreement, blur, truncation, occlusion and temporal consistency) is **BLOCKED**. The repository does not contain independent human labels sufficient to create development, validation, and untouched test splits. Training on the current model outputs or authored scenarios would not establish independent truth, so no trained model or calibration claim is made.

## Reproduction

```bash
.venv/bin/python -m evaluation.benchmark_proofloop \
  --output docs/benchmarks/proofloop_closed_loop_report.json
```
