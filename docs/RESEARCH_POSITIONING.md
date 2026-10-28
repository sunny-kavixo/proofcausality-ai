# ProofCausality AI — Research Positioning & Theoretical Foundations

**Project:** ProofCausality AI  
**Permanent Tagline:** "Prove what happened. Detect what is missing. Capture the evidence that changes the decision."  
**Author:** AI Research & Computer Vision Engineering Team  
**Event:** Meta Global AI Developer Hackathon 2026 (Cerebral Valley) — Indie Developer Track / Push the Frontier  

---

## 1. Primary Research Hypothesis

> **Central Hypothesis:**  
> Explicitly decomposing visual physical-action verification and annotation auditing into deterministic geometric evidence, uncertainty signals (occlusion, boundary clipping, motion blur, and temporal truncation), and targeted re-capture instructions significantly reduces **false-success judgments** and increases **uncertain-case resolution rates** compared to end-to-end vision-language model confidence scores or generic "re-record" prompts.

### Formalized Claims vs. Non-Claims
- **What ProofCausality AI Claims:**
  1. High model confidence does not equate to mathematical proof. When critical visual frames are missing, blurred, or occluded, abstaining with  and identifying the *exact missing signal* prevents costly false-positive decisions in robotic manipulation and dataset curation.
  2. Targeted re-capture instructions (e.g., *"Hold steady until object is sharp"* vs *"Re-record the task"*) dramatically improve the probability that a second observation resolves ambiguity into a decisive  or  verdict.
- **What ProofCausality AI Does NOT Claim:**
  1. We do NOT claim to have solved general visual causal inference or 3D physical dynamics from uncalibrated 2D monocular video.
  2. We do NOT claim that SHA-256 evidence receipt hashes prove physical real-world truth; cryptographic hashes guarantee audit-trail immutability and data integrity, not physical ground reality.
  3. We do NOT claim segmentation alone proves semantic class correctness without independent multi-modal or class-specific evidence.

---

## 2. Prior Art Review & Context

### A. Dataset Annotation Auditing & Label Error Detection
- **Existing Approaches:** Cleanlab (Northcutt et al., 2021) and confident learning estimate label noise in classification datasets by modeling out-of-sample predicted probabilities. For segmentation, consensus metrics across multi-annotator crowds (e.g., STAPLE algorithm, Warfield et al., 2004) compute probabilistic true segmentations.
- **Limitation:** Most methods operate offline in bulk and rely on pre-trained classification heads. They lack real-time visual evidence feedback linking spatial boundary discrepancies directly to human operator review workflows.
- **ProofCausality (LabelCourt) Contribution:** Implements deterministic spatial disagreement heatmaps, boundary agreement metrics, and mask IoU thresholds for interactive single-image and batch COCO auditing with human-in-the-loop review boundaries.

### B. Active Perception & Next-Best-View Planning
- **Existing Approaches:** Active perception (Bajcsy, 1988) and Next-Best-View (NBV) algorithms in robotics plan sensor poses to maximize information gain (e.g., mutual information over occupancy grids).
- **Limitation:** Robotics NBV algorithms require calibrated 3D sensor rigs, depth cameras, or robotic arms with 6-DoF inverse kinematics. They cannot communicate semantic, actionable capture advice to human operators holding a smartphone or wearable device.
- **ProofCausality (ReCapture) Contribution:** Formulates a rule-based deterministic taxonomy that maps detected monocular failure modes (Laplacian blur, boundary contact, area occlusion, missing final state) directly to natural-language physical capture directives for humans.

### C. Physical Action Verification in Egocentric & Manipulation Videos
- **Existing Approaches:** Video action recognition (SlowFast, VideoMAE) and state-change detection (Ego4D PNR, Grauman et al., 2022) train deep neural networks to predict whether an action occurred.
- **Limitation:** End-to-end neural classifiers suffer from severe overconfidence under distribution shift, predicting success even when the critical interaction is obscured by the user's hand or tool.
- **ProofCausality (ActionProof) Contribution:** Employs an evidence-first tri-state decision model (, , ). When intermediate frames fail observable visibility criteria, the system refuses to guess, emitting  along with an Evidence Receipt and ReCapture plan.

---

## 3. The Evidence-Aware Architecture



---

## 4. Empirical Evaluation Protocol

We evaluate ProofCausality AI across two empirical benchmarks:
1. **False-Success Rate (FSR):**
   91624	ext{FSR} = rac{	ext{False Success Count}}{	ext{Total Failed or Insufficiently Evidenced Cases}}91624
   Evaluated across controlled test cases containing ground-truth occlusions, incomplete movements, boundary escapes, and camera blur.
2. **Re-Capture Resolution Rate (RRR):**
   91624	ext{RRR} = rac{	ext{Initially Uncertain Cases Decided Correctly Post-Recapture}}{	ext{Total Uncertain Cases}}91624
   Evaluated by pairing uncertain cases with guided re-capture inputs following the targeted template vs. generic re-recording.
