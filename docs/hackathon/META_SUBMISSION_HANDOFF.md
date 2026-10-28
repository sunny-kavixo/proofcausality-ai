# Meta Global AI Developer Hackathon 2026 — Submission Handoff (v0.1.1)

> **Official Event:** [Meta AI Developer Hackathon (Cerebral Valley)](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)  
> **Submission Track:** Indie Developer  
> **Theme:** Push the Frontier  
> **Secondary Track Relevance:** Build for Your World (Robotic physical-task verification & visual annotation quality)  
> **Deadline:** November 4, 2026, 9:00 PM US Eastern  
> **Latest Verified Release:** `v0.1.1`  
> **Public GitHub Repository:** [https://github.com/sunny-kavixo/proofcausality-ai](https://github.com/sunny-kavixo/proofcausality-ai)  

---

## Submission Form Copy-Paste Blocks

### Project Title
`ProofCausality AI`

### Tagline
`Prove what happened. Detect what is missing. Capture the evidence that changes the decision.`

### Short Description (under 200 words)
ProofCausality AI is an evidence-aware visual causality and physical-action verification system designed to eliminate false-success hallucinations in robotic manipulation, dataset annotation auditing, and high-consequence vision tasks.

Standard vision models return high-confidence positive judgments even when objects are occluded, clipped at image borders, blurred by camera motion, or left unverified after execution. ProofCausality AI introduces a bounded, verifiable arbitration architecture based on four core engines: LabelCourt (annotation evidence auditor), ActionProof (before/after physical action verifier), EvidenceCore (Laplacian blur and boundary uncertainty detector accelerated by Rust SIMD kernels), and ReCapture (deterministic missing-evidence recovery planner).

On our curated 12-case real tabletop evaluation suite, ProofCausality reduced the False-Success Rate to 0.0% (95% Wilson CI: [0.0%, 35.4%]) versus 42.9% (95% CI: [15.8%, 75.0%]) for naive coordinate heuristics, and achieved 100.0% ReCapture Resolution (95% CI: [51.0%, 100.0%]) using targeted physical recovery instructions. The system features an optional zero-allocation Rust acceleration core delivering an 11.35x speedup on batch mask operations with pure Python fallback. Every verdict emits a tamper-evident SHA-256 Evidence Receipt separating cryptographic file integrity from physical truth.

### Meta Technologies Used
- **Meta SAM 2 / 2.1 (Segment Anything Model)**: Ingests segmentation masks for high-fidelity contour auditing, mask IoU calculation, and spatial displacement tracking with base64 image encoding.
- **Meta Llama 3.2 Vision (11B/90B)**: Formats multimodal scene observations and structured visual context.
- **EvidenceCore Deterministic Fallback**: Provides a 100% offline, zero-cloud deterministic fallback implementing pure 2D Laplacian convolution and geometric boundary detection so the pipeline operates without black-box dependency.

### Public GitHub Repository URL
`https://github.com/sunny-kavixo/proofcausality-ai`

### Release v0.1.1 URL
`https://github.com/sunny-kavixo/proofcausality-ai/releases/tag/v0.1.1`

### Video Demonstration Script & Shot List (2:45 Target)

| Timestamp | Screen / Scene | Voiceover Narration | Visual Focus |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:25** | System Title & Problem Statement | *"In robotic manipulation and quality control, modern vision models suffer from a dangerous flaw: the hallucination of physical success. When an action is occluded or blurred, models guess."* | Title card, architecture diagram showing Tri-State decision arbitration. |
| **0:25 - 0:50** | LabelCourt & Real Image Processing | *"Here in LabelCourt, an authentic tabletop image is audited. The system computes pixel-accurate IoU, Dice coefficients, boundary agreement, and renders a color-coded disagreement overlay."* | Real image uploader, overlay showing green agreement and red deficit regions, IoU metric. |
| **0:50 - 1:25** | ActionProof & The False-Success Trap | *"In ActionProof, we test a pick-and-place task. Under rapid camera motion blur or gripper occlusion, standard vision models declare success. ProofCausality flags the Laplacian blur variance and emits UNCERTAIN."* | Scenario selector clicking 'Motion Blur', UNCERTAIN badge appearing, showing Laplacian variance 38.4 < 100.0. |
| **1:25 - 1:55** | ReCapture Recovery Loop | *"Instead of generic advice like 'record again', ReCapture issues a targeted physical directive: 'Hold camera steady and allow autofocus lock.' Once re-captured, the system re-evaluates to a verified SUCCESS."* | ReCapture card displaying exact directive, clicking 'Simulate Corrective ReCapture', verdict transitioning to SUCCESS. |
| **1:55 - 2:20** | Rust Acceleration & Real Benchmarks | *"Across our curated real tabletop evaluation suite, ProofCausality reduces the False-Success Rate from 43% to 0%, resolving 100% of uncertain cases. Our Rust native core accelerates batch mask operations by 11.3x."* | Benchmark tab showing Wilson score CIs, Rust performance dashboard showing 11.35x speedup table. |
| **2:20 - 2:45** | Evidence Receipts & Meta Hackathon Close | *"Every decision issues a canonical SHA-256 Evidence Receipt. ProofCausality AI: Prove what happened. Detect what is missing. Capture the evidence that changes the decision."* | Interactive JSON Evidence Receipt viewer, GitHub repo link, release v0.1.1 tag. |

---

## Submission Checklist for Owner (Sunny)
- [ ] Review code and release at `https://github.com/sunny-kavixo/proofcausality-ai/releases/tag/v0.1.1`
- [ ] Record video according to the shot list above (using QuickTime or OBS)
- [ ] Upload video to YouTube (Public or Unlisted)
- [ ] Go to [Cerebral Valley Submission Portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)
- [ ] Paste Title, Tagline, Description, GitHub URL, and YouTube link
- [ ] Click Submit prior to deadline: **November 4, 2026, 9:00 PM EST**
