# FINAL META HACKATHON SUBMISSION HANDOFF — PROOFCAUSALITY AI (v0.2.3)

**Project Name:** ProofCausality AI  
**Repository:** [https://github.com/sunny-kavixo/proofcausality-ai](https://github.com/sunny-kavixo/proofcausality-ai)  
**Target Competition:** Meta Global AI Developer Hackathon 2026 (Hosted by Meta & Cerebral Valley)  
**Track:** Indie Developer  
**Theme:** Push the Frontier  
**Official Portal:** [https://cerebralvalley.ai/e/meta-ai-developer-hackathon](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)  
**Official Submission Deadline:** **November 4, 2026, 9:00 PM EST** (Nov 5, 2026, 02:00 UTC)  
**Latest Production Release:** `v0.2.3` ([GitHub Release](https://github.com/sunny-kavixo/proofcausality-ai/releases/tag/v0.2.3))  

---

## 1. Project Overview & Pitch Summary

> **"Prove what happened. Detect what is missing. Capture the evidence that changes the decision."**

### Problem Statement
Standard computer vision models and vision-language models (VLMs) hallucinate physical success under adverse conditions—such as severe camera motion blur, gripper occlusions, boundary clipping, and unobserved object releases. In high-consequence robotic manipulation, logistics verification, and dataset curation, **a false success is catastrophic**. Traditional systems either guess with uncalibrated confidence or issue vague advice (*"please re-record"*), failing to pinpoint the missing evidence.

### The Solution: ProofCausality AI
ProofCausality AI introduces **Evidence Calculus**, a condition-aware verification engine that converts perceptual inputs into a falsifiable system of 8 observable physical conditions. Rather than outputting a single probabilistic score, it calculates **Evidence Debt ($D(E,T)$)**. When perceptual deficits or unverified physical rest states exist, the engine abstains (`ActionVerdict.UNCERTAIN`), emits a structured finding (`POSITION_VERIFIED` or `PERCEPTUAL_DEFICIT`), and invokes the **ReCapture Loop** to guide the user or robot to collect targeted missing observations.

---

## 2. Meta Technology & Integration Status

| Technology | Architecture & Model ID | Operational State | Provenance / Fallback |
| :--- | :--- | :---: | :--- |
| **Meta Segment Anything (SAM)** | `facebook/sam-2.1-hiera-large` / `SAM 3.1` via `backend/providers/meta_adapter.py` | `META_INFERENCE_BLOCKED` | Fallback: Deterministic 2D bounding-box geometric contour (`GEOMETRIC_FALLBACK`) |
| **Meta Llama 3.2 Vision** | `meta-llama/Llama-3.2-11B-Vision-Instruct` via `backend/providers/meta_adapter.py` | `META_INFERENCE_BLOCKED` | Fallback: Deterministic observable predicate auditor (`OFFLINE_DETERMINISTIC_FALLBACK`) |
| **Meta Inference Partners** | Compatible with Baseten, Modal, and Fireworks endpoints | Ready for Credentials | Offline deterministic verification active; zero unauthorized or billable API calls |

*Truthful Disclosure: Remote Meta model inference requires `META_API_KEY`. When credentials are absent, the application honestly surfaces `META_INFERENCE_BLOCKED` with detailed explanations in `/api/model-info` and the UI header.*

---

## 3. Working Features & Production Verification

1. **LabelCourt (Visual Annotation Evidence Auditor):**
   - Computes mask IoU, Dice coefficients, and contour boundary disagreement with 11.57× Rust SIMD acceleration.
2. **ActionProof (Physical Action Verification Engine):**
   - Verifies before/after spatial transitions (containment, displacement, proximity) under strict nonfinite bounds checking.
3. **Evidence Calculus Engine:**
   - Evaluates 8 typed observable physical predicates (`P_IDENT`, `P_INIT`, `P_SPATIAL`, `P_SHARP`, `P_FRAME`, `P_OCCL`, `P_PHYS`, `P_FINAL`).
   - Guarantees that static spatial placement (`POSITION_VERIFIED`) is never conflated with physical rest/release proof (`ACTION_COMPLETED`).
4. **ReCapture Closed-Loop:**
   - `/api/recapture/evaluate-followup` records before/after state transitions, effort tracking, and resolution status.
5. **Cryptographic Merkle Evidence Receipts:**
   - Tamper-evident SHA-256 Merkle Evidence Receipts (`calc_evidence_receipt_*.json`) downloadable for every verification.

---

## 4. Scientific Baseline Evaluation (H01 Benchmark)

Evaluated on 37 benchmark cases (12 curated tabletop geometric metadata pairs + 25 synthetic cases) at an exact matched coverage of **75.68% (28/37 cases decided)**:

| Method | Coverage | Decided | Abstentions | False-Success Rate | 95% Wilson CI | Selective Risk |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Naive Distance Heuristic** | 100.0% | 37/37 | 0 | 45.45% (10/22) | `[26.92%, 65.34%]` | 0.3514 |
| **2. Continuous Score Margin** | 75.68% | 28/37 | 9 | 13.64% (3/22) | `[4.75%, 33.33%]` | 0.1429 |
| **3. Fixed Rules Baseline** | 75.68% | 28/37 | 9 | 4.55% (1/22) | `[0.81%, 21.80%]` | 0.1429 |
| **4. Evidence Debt Unit Only** | 40.54% | 15/37 | 22 | 0.00% (0/22) | `[0.00%, 14.87%]` | 0.0000 |
| **5. Evidence Calculus (Full)** | **75.68%** | **28/37** | **9** | **0.00% (0/22)** | **`[0.00%, 14.87%]`** | **0.1429** |

---

## 5. Performance Benchmarks (Empirically Measured)

* **Batch Mask IoU ($100 \times 256 \times 256$):** Pure NumPy: 2,348.5 µs $\to$ Rust SIMD: **203.0 µs** (**11.57× speedup**).
* **Laplacian Blur Kernel ($1024 \times 1024$):** Pure NumPy: 2,115.9 µs $\to$ Rust SIMD: **600.4 µs** (**3.52× speedup**).
* **Python-to-Rust CFFI Overhead:** p50 = **1.29 µs**, p95 = 1.50 µs.
* **Image Preprocessing (640×480 RGB $\to$ Gray):** p50 = **410.65 µs**, p95 = 425.21 µs.
* **End-to-End FastAPI HTTP Request Roundtrip:** p50 = **0.730 ms – 0.974 ms**.

---

## 6. Official Architecture Diagrams & Live Screenshots

### Vector Architecture Diagrams
- System Architecture: [`docs/architecture/system-architecture.svg`](../../docs/architecture/system-architecture.svg)
- Evidence Calculus Decision Flow: [`docs/architecture/evidence-calculus-decision-flow.svg`](../../docs/architecture/evidence-calculus-decision-flow.svg)
- ReCapture Closed-Loop: [`docs/architecture/recapture-closed-loop.svg`](../../docs/architecture/recapture-closed-loop.svg)
- Integration Boundaries: [`docs/architecture/integration-boundaries.svg`](../../docs/architecture/integration-boundaries.svg)

### Live Browser UI Screenshots
- Main Studio & Diagnostics: [`docs/screenshots/landing_page.png`](../../docs/screenshots/landing_page.png)
- ActionProof Verification: [`docs/screenshots/actionproof_workflow.png`](../../docs/screenshots/actionproof_workflow.png)
- Evidence Calculus Predicates & Debt: [`docs/screenshots/evidence_calculus_tab.png`](../../docs/screenshots/evidence_calculus_tab.png)
- ReCapture Recovery Guidance: [`docs/screenshots/recapture_tab.png`](../../docs/screenshots/recapture_tab.png)
- Benchmarking & Latency Profiles: [`docs/screenshots/benchmarks_tab.png`](../../docs/screenshots/benchmarks_tab.png)

---

## 7. Judge Quick-Start & Verification Instructions

### 1. Installation from Release Archive (Zero-Config)
```bash
# Clone or download release v0.2.3
git clone https://github.com/sunny-kavixo/proofcausality-ai.git
cd proofcausality-ai

# Python virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run complete test suite (101 tests passed in ~0.5s)
python3 -m pytest -q
```

### 2. Start Application Server
```bash
python3 -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) to inspect the live interface.

---

## 8. 2–3 Minute Demonstration Video Script (For Owner)

| Time | Scene | Voiceover Script | Visual Action on Screen |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:30** | Problem & Architecture | *"Welcome to ProofCausality AI. Vision models routinely hallucinate physical success under motion blur, occlusions, and incomplete actions. In robotics, a false success is catastrophic. We built Evidence Calculus to replace guesswork with observable physical proof."* | Show Landing Page, Rust SIMD badge, and System Architecture Diagram. |
| **0:30 - 1:15** | ActionProof & Static Position | *"Here is an object relocation action. Notice how ProofCausality distinguishes spatial position from physical rest. Even when the object is sharp in the target region, the engine outputs POSITION_VERIFIED rather than blindly declaring success until rest evidence is observed."* | Run ActionProof verification on Canonical Demonstration Case A. Show predicate breakdown. |
| **1:15 - 2:00** | Perceptual Deficit & Abstention | *"Now observe a case with severe motion blur. Standard models hallucinate success. Evidence Calculus penalizes the blur deficit with Evidence Debt D(E,T) = 1.0 and abstains with an UNCERTAIN verdict."* | Upload Canonical Demonstration Case C (blurred). Show Debt Meter rising to 1.0 and verdict becoming UNCERTAIN. |
| **2:00 - 2:35** | ReCapture Closed-Loop | *"Instead of generic advice, ReCapture generates a targeted directive: 'Hold camera steady for 1.5 seconds'. When the steady frame is provided, the follow-up endpoint resolves the deficit, transitioning the decision from UNCERTAIN to SUCCESS."* | Follow ReCapture prompt, submit recaptured frame, show verdict transition to SUCCESS. |
| **2:35 - 3:00** | Tamper-Evident Receipt & Summary | *"Every decision generates a cryptographic SHA-256 Merkle Evidence Receipt. ProofCausality AI: Prove what happened. Detect what is missing. Capture the evidence that changes the decision."* | Click 'Download Evidence Receipt', display JSON receipt with verified Merkle hash. |

---

## 9. Ready-to-Paste Hackathon Submission Text

### Project Title
ProofCausality AI — Evidence-Aware Physical Action Verification & ReCapture Engine

### Elevator Pitch
ProofCausality AI replaces uncalibrated computer vision guesswork with observable physical proof. Using Evidence Calculus and Evidence Debt $D(E,T)$, it detects missing visual evidence in robotics and dataset auditing, abstaining on ambiguous actions and guiding targeted re-captures that change the final decision.

### Theme & Track
* **Track:** Indie Developer
* **Theme:** Push the Frontier (Software, Robotics, Physical Systems)

### Description
In high-consequence physical AI, robotic manipulation, and dataset annotation auditing, a false success is catastrophic. Current vision-language models often hallucinate physical task completion when objects are obscured by grippers, blurred by rapid camera motion, or unobserved after release.

ProofCausality AI introduces **Evidence Calculus**:
1. **Tri-State Predicate Logic:** Decomposes actions into 8 observable physical conditions evaluated as `SUPPORTED`, `CONTRADICTED`, or `UNOBSERVABLE`.
2. **Physics-Constrained Evidence Debt ($D(E,T)$):** Quantifies perceptual and physical deficits, reducing False-Success Rate from 45.45% (naive heuristic) and 13.64% (continuous score thresholding) to **0.00%** at exact matched coverage ($75.68\%$).
3. **Rust SIMD Core:** High-performance native acceleration delivering an **11.57× speedup** on batch mask IoU ($2348.5\,\mu\text{s} \to 203.0\,\mu\text{s}$) with 1.29 µs CFFI overhead.
4. **Closed-Loop ReCapture:** Automatically identifies missing evidence and provides operators or robots with exact corrective directives to resolve deficits and transition decisions.
5. **Tamper-Evident Receipts:** Emits SHA-256 Merkle Evidence Receipts guaranteeing audit reproducibility.

---

## 10. Remaining Owner-Only Actions

1. **Demonstration Video:** Record and upload the 2–3 minute video presentation following the script above to YouTube (public or unlisted).
2. **Portal Submission:** Paste the submission description, repository link ([https://github.com/sunny-kavixo/proofcausality-ai](https://github.com/sunny-kavixo/proofcausality-ai)), release link ([https://github.com/sunny-kavixo/proofcausality-ai/releases/tag/v0.2.3](https://github.com/sunny-kavixo/proofcausality-ai/releases/tag/v0.2.3)), and YouTube URL into the Cerebral Valley portal before **November 4, 2026, 9:00 PM EST**.
