# Remaining Risks and Release Boundaries

## Open risk register

| ID | Priority | Risk | Current status | Mitigation/evidence |
|---|---:|---|---|---|
| R-01 | P1 | No authentication or durable multi-tenant receipt storage | BLOCKED for production | Keep deployment local; production certification remains not granted |
| R-02 | P1 | Metadata ProofLoop endpoints accept features; media endpoints require image bytes but do not establish temporal continuity | DISCLOSED | Metadata receipts say `METADATA_ONLY`; media receipts say `SERVER_DERIVED_MEDIA` and bind hashes |
| R-03 | P1 | Live Meta provider request not authorized or executed | BLOCKED | Offline adapter and parser tests only; no key or billable call used |
| R-04 | P1 | Human annotation and recapture study absent | NOT COMPLETED | Ready-to-run protocol, CSV sheet, and analysis runner added |
| R-05 | P1 | Authentic temporal video experiment absent | BLOCKED | Temporal protocol added; no causal claim made |
| R-06 | P2 | Local segmentation diagnostic recall is limited | MEASURED LIMITATION | Report 171/332 @ .50 and 110/332 @ .75; do not advertise production accuracy |
| R-07 | P2 | ProofLoop resolution benchmark uses authored feature fixtures | CONTROLLED ONLY | Report 6/7 resolution as policy behavior, not field effectiveness |
| R-08 | P2 | No independent labels for learned calibration | BLOCKED | Deterministic baseline retained; no trained verifier claimed |
| R-09 | P2 | Receipts are unkeyed integrity hashes | DISCLOSED | Never call them signatures, HMACs, Merkle proofs, or proof of truth |
| R-10 | P3 | Hackathon participant dashboard/deadline is owner-account scoped | BLOCKED | Owner must verify dashboard and submit; repository contains prepared text only |

## Owner-controlled actions

1. Verify the exact submission deadline and required fields in the authenticated participant dashboard.
2. If live Meta integration is required, authorize a real provider endpoint and credential, then record the endpoint/model/request/response without exposing the key.
3. Record the real demo video and submit the portal form.
4. If making a human or temporal validation claim, collect the required independent evidence first.
