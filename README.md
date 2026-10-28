# ProofCausality AI

> **"Prove what happened. Detect what is missing. Capture the evidence that changes the decision."**

[![CI](https://github.com/sunny-kavixo/proofcausality-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/sunny-kavixo/proofcausality-ai/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Meta Hackathon 2026](https://img.shields.io/badge/Meta_Global_AI_Hackathon-Indie_Developer-purple.svg)](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)

ProofCausality AI is an evidence-aware visual causality and physical-action verification system engineered for high-consequence computer vision, robotic manipulation, and dataset annotation auditing.

Built for the **Meta Global AI Developer Hackathon 2026** (Track: *Indie Developer*, Theme: *Push the Frontier*).

---

## The Problem: The Hallucination of Physical Success

Modern visual-language models (VLMs) and segmentation frameworks are trained for perceptual fluency, not deterministic physical proof. When an object is occluded by a robotic gripper, blurred by fast camera motion, clipped by image boundaries, or left unobserved after a supposed release, standard models routinely return high-confidence predictions of success.

In robotic manipulation, logistics inspection, and dataset quality auditing, **a false success is catastrophic**. Ordinary systems either guess or issue generic advice like *"record again"*, which fails to communicate the exact physical evidence deficit to the operator or robotic agent.

---

## The ProofCausality Solution

ProofCausality AI introduces a bounded, verifiable arbitration pipeline:
1. **Never substitute model confidence for factual proof.**
2. **Support three honest action outcomes:** `SUCCESS`, `FAILED`, and `UNCERTAIN`.
3. **Emit targeted, deterministic ReCapture instructions** when evidence is missing.
4. **Issue tamper-evident, SHA-256 Evidence Receipts** anchoring the observation chain.

![System Architecture](docs/architecture/system-architecture.svg)

---

## Core Engines

| Engine | Name | Responsibility | Deterministic Guarantee |
| :--- | :--- | :--- | :--- |
| **LabelCourt** | Visual Annotation Evidence Auditor | Audits human vs model segmentations, flags boundary discrepancies, and exports verified COCO subsets. | Pixel-accurate IoU, displacement vectors, boundary agreement. |
| **ActionProof** | Physical Action Verification Engine | Verifies before-and-after physical state transitions (e.g., A-to-B relocation, spatial proximity, presence). | Distance thresholds, bounding containment, temporal pair validation. |
| **EvidenceCore** | Measurement & Uncertainty Engine | Detects low-level evidence deficits: 2D Laplacian blur, border contact, area-drop occlusion, missing final state. | Pure mathematical convolution; no black-box hallucination. |
| **ReCapture** | Recovery Planner | Generates deterministic, actionable physical instructions for the operator or robot camera. | Priority-ranked directives resolving specific visual deficits. |

---

## Empirical Benchmark Results

We evaluated ProofCausality AI across a standardized benchmark suite of **25 physical action cases** (`docs/benchmarks/benchmark_results.json`):
- **10 Clean Success Cases**
- **6 Defective/Incomplete Failure Cases**
- **9 Severe Evidence-Deficit Cases** (motion blur, boundary clipping, gripper occlusion)

### 1. False-Success Rate (FSR)
> **Definition:** Fraction of non-success cases (failures + uncertain) incorrectly classified as `SUCCESS`.
> **Denominator:** 15 non-success ground-truth cases.

$$\text{FSR} = \frac{\text{False Success Count}}{\text{Total Non-Success Cases (15)}}$$

- **Naive Vision Baseline:** **80.0%** (12 of 15 non-success cases incorrectly accepted as success)
- **ProofCausality AI:** **0.0%** (0 of 15 non-success cases falsely accepted)
- **Relative Risk Reduction:** **100.0% elimination of false successes**

### 2. Re-Capture Resolution Rate (RRR)
> **Definition:** Fraction of initially `UNCERTAIN` cases that convert to a verified true state after following the recommended capture directive.
> **Denominator:** 9 uncertain cases.

- **Generic Baseline Guidance** (*"Please record again"*): **0.0%** (0/9 cases resolved)
- **Targeted ProofCausality ReCapture**: **100.0%** (9/9 cases decisively resolved)

### 3. Over-Abstention Rate
- **0.0%** false rejections on clear, fully evidenced ground-truth successes (10/10 verified).

---

## Evidence Receipts: Factual Truth vs File Integrity

Every verdict generates a canonical SHA-256 **Evidence Receipt**:
```json
{
  "report_id": "AP-VERIFY-1730123456-ABCDEF",
  "receipt_version": "1.0",
  "timestamp_utc": "2026-10-28T18:35:00Z",
  "software_version": "0.1.0",
  "model_provider": "Meta AI Adapter (Deterministic Core)",
  "verdict": "UNCERTAIN",
  "uncertainty_signals": [
    {
      "reason": "BLUR",
      "severity": "CRITICAL",
      "description": "2D Laplacian variance below sharpness threshold (34.2 < 100.0)"
    }
  ],
  "recapture_directive": "Hold the camera steady and allow the object to become sharp before ending the capture.",
  "receipt_sha256": "4a7d...39e1"
}
```

> **Important Scientific Distinction:**
> ProofCausality explicitly documents that **a cryptographic hash guarantees file and log integrity, not physical truth**. The hash guarantees that the recorded observations, thresholds, and intermediate signals have not been altered after the fact.

---

## Quickstart & Installation

### Requirements
- Python 3.11+ (Tested on Python 3.11, 3.12, 3.13, 3.14 on macOS and Linux)
- Git

### 1. Clone & Setup
```bash
git clone https://github.com/sunny-kavixo/proofcausality-ai.git
cd proofcausality-ai

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Automated Test Suite
```bash
pytest tests/ -v
```
All 39 unit and integration tests execute in <0.5 seconds.

### 3. Run Benchmark Verification
```bash
python3 -m evaluation.benchmark
```
Generates updated `docs/benchmarks/benchmark_results.json` and `docs/benchmarks/benchmark_cases.csv`.

### 4. Launch the Web Application
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8080 --reload
```
Open your browser to `http://localhost:8080` to access the interactive dashboard.

---

## Meta Ecosystem Integration

ProofCausality AI is architected with native adapters for:
- **Meta SAM 2 / 2.1 (Segment Anything Model)**: Ingests segmentation masks for boundary and IoU auditing.
- **Meta Llama 3.2 Vision (11B / 90B)**: Formats structured perceptual observations.
- **EvidenceCore Deterministic Fallback**: In production or offline environments without active Meta API keys, the system executes pure NumPy/Pillow convolution and geometric analysis, maintaining 100% testable local autonomy with zero cloud dependency.

---

## Project Structure

```
proofcausality-ai/
├── .github/workflows/ci.yml       # Automated GitHub Actions CI
├── backend/
│   ├── api/routes.py              # REST API endpoints
│   ├── app.py                     # FastAPI application setup
│   ├── evidence/                  # EvidenceCore: schemas, masks, metrics, blur, receipts
│   ├── labelcourt/                # LabelCourt: annotation auditor & COCO exporter
│   ├── actionproof/               # ActionProof: physical action verifier
│   └── providers/meta_adapter.py  # Meta SAM / Llama Vision adapter
├── evaluation/
│   ├── cases/action_dataset.py    # 25 standardized benchmark cases
│   ├── inject_errors.py           # Controlled perturbation injection
│   └── benchmark.py               # Scientific evaluation runner
├── frontend/static/               # Developer-grade single page app
│   ├── index.html
│   ├── style.css
│   └── app.js
├── docs/
│   ├── architecture/              # System architecture SVG & specs
│   ├── benchmarks/                # JSON and CSV evaluation records
│   ├── evidence/                  # Model feasibility audit
│   └── hackathon/                 # Meta submission handoff & video scripts
├── tests/                         # 39 Pytest test cases
├── pyproject.toml
├── requirements.txt
├── LICENSE                        # MIT License
└── README.md
```

---

## License

This project is open-source under the [MIT License](LICENSE).
