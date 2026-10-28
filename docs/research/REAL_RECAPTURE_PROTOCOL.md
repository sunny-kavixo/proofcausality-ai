# Real-World ReCapture Evaluation Protocol & Manifest

**Project**: ProofCausality AI  
**Release**: `v0.2.1`  
**Date**: October 2026  

---

## 1. Protocol Objective

The purpose of this protocol is to define a reproducible, evidence-based standard for evaluating missing-evidence recovery directives, distinguishing simulated Monte Carlo results from observed physical recovery trials.

---

## 2. Experimental Data Schema (`ReCaptureRecord`)

Every re-capture evaluation must log a verified tuple:
1. `record_id`: Unique identifier for the trial.
2. `original_case_id`: Identifier of the original defective recording.
3. `initial_verdict`: Initial arbitration outcome (must be `UNCERTAIN`).
4. `detected_defect`: Specific physical or perceptual deficiency (`BLUR`, `BORDER_CLIPPING`, `OCCLUSION`, `MISSING_REST_STATE`).
5. `instruction_type`: Directives category (`GENERIC`, `FIXED_TEMPLATE`, `VOI_PLANNER`).
6. `generated_instruction`: Exact human-readable text directive emitted by the system.
7. `follow_up_observation_available`: Boolean indicating whether a follow-up recording exists.
8. `follow_up_resolved_verdict`: New verdict on follow-up (`SUCCESS`, `FAILED`, `UNCERTAIN`).
9. `effort_seconds`: Measured physical execution time in seconds.
10. `instruction_followed`: Independent annotator confirmation of operator compliance.
11. `is_real_human_trial`: Explicit flag separating live human trials from static paired counterfactuals.

---

## 3. Current Measured Evidence Status

| Evaluation Category | Test Set | Result | Status |
| :--- | :--- | :--- | :--- |
| **Curated Paired Counterfactual States** | 4 tabletop pairs (`REAL-U01` to `REAL-U04`) | **4 / 4 resolved (100.0%)** | **VALIDATED ON PAIRED STATES** |
| **Monte Carlo Algorithmic Simulation** | 20 synthetic defect cases | **95.0% VOI vs 30% generic** | **CONTROLLED SIMULATION** |
| **In-the-Wild Live Multi-User Human Trials** | Unconstrained human participants | N/A | **NOT MEASURED** |
