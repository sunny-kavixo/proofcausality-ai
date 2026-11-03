# Temporal Evidence Validation Protocol

**Status: PROTOCOL ONLY; no authentic temporal video validation completed.**

## Scope

This protocol distinguishes temporal observation from action-completion inference and from causal attribution. The first feasible experiment is a controlled before/during/after observation task such as moving a marked object from region A to region B.

## Required data

- Rights-cleared or owner-recorded video clips with provenance and consent.
- Independently recorded task outcome and intervention timestamps.
- Frozen train/development/test split by clip, not by frame.
- Object identity and region annotations from blinded adjudicators for the test clips.

## Measurements

For each clip, report:

- track continuity and identity-switch count;
- before/during/after visibility and occlusion intervals;
- measured displacement and target containment;
- motion cessation and release evidence where observable;
- missing-frame and blur indicators;
- `SUPPORTED`, `CONTRADICTED`, or `INSUFFICIENT_EVIDENCE` decision;
- abstention coverage, selective risk, and per-failure-case analysis.

Compare ProofLoop with a frame-only baseline under the same frozen split. A temporal sequence can support an observable completion claim, but sequence order alone does not establish that an intervention caused the outcome. A causal claim requires an intervention-controlled design and remains **NOT VERIFIED** here.

## Reproduction and disclosure

The future runner should save a clip manifest, SHA-256 media hashes, annotation provenance, frame sampling settings, model revision, decision traces, and receipts. Until that data and runner exist, ActionProof/ReCapture temporal performance is **BLOCKED**, not a measured result.
