# Meta Global AI Developer Hackathon 2026 — Submission Handoff

> **Official Event:** [Meta AI Developer Hackathon (Cerebral Valley)](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)  
> **Submission Track:** Indie Developer  
> **Theme:** Push the Frontier  
> **Secondary Relevance:** Build for Your World (Robotic physical-task verification & visual annotation quality)  
> **Deadline:** November 4, 2026, 9:00 PM US Eastern  

---

## Submission Form Copy-Paste Blocks

### Project Title
`ProofCausality AI`

### Tagline
`Prove what happened. Detect what is missing. Capture the evidence that changes the decision.`

### Short Description (under 200 words)
ProofCausality AI is an evidence-aware visual causality and physical-action verification system designed to eliminate false-success hallucinations in robotic manipulation, dataset annotation auditing, and high-consequence vision tasks. 

Standard vision models return high-confidence positive judgments even when objects are occluded, clipped at image borders, blurred by camera motion, or left unverified after execution. ProofCausality AI introduces a bounded, verifiable arbitration architecture based on four core engines: LabelCourt (annotation evidence auditor), ActionProof (before/after physical action verifier), EvidenceCore (Laplacian blur and boundary uncertainty detector), and ReCapture (deterministic missing-evidence recovery planner).

On our 25-case standardized physical benchmark, ProofCausality eliminated false-success judgments entirely (0.0% vs. 80.0% naive vision baseline across 15 non-success cases) and achieved a 100.0% Re-Capture Resolution Rate by providing targeted corrective capture instructions to operators and robotic camera systems. Every verdict emits a tamper-evident SHA-256 Evidence Receipt separating cryptographic file integrity from physical truth.

### Meta Technologies Used
- **Meta SAM 2 / 2.1 (Segment Anything Model)**: Ingests segmentation masks for high-fidelity contour auditing, mask IoU calculation, and spatial displacement tracking.
- **Meta Llama 3.2 Vision (11B/90B)**: Formats multimodal scene observations and structured visual context.
- **EvidenceCore Deterministic Fallback**: Provides a 100% offline, zero-cloud deterministic fallback implementing pure 2D Laplacian convolution and geometric boundary detection so the pipeline operates without black-box dependency.

### Public GitHub Repository URL
`https://github.com/sunny-kavixo/proofcausality-ai`

### Video Demonstration Script & Shot List (2:45 Target)

| Timestamp | Screen / Scene | Voiceover Narration | Visual Focus |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:25** | System Title & Problem Statement | *"In robotic manipulation and quality control, modern vision models suffer from a dangerous flaw: the hallucination of physical success. When an action is occluded or blurred, models guess."* | Title card, architecture diagram showing Tri-State decision arbitration. |
| **0:25 - 0:55** | LabelCourt Annotation Audit | *"Here in LabelCourt, an AI candidate mask is compared against human ground-truth. The system computes pixel-accurate IoU, boundary agreement, and checks for boundary clipping."* | Split-screen showing human mask vs AI candidate, IoU metric (0.94), and ACCEPT verdict. |
| **0:55 - 1:35** | ActionProof & The False-Success Trap | *"In ActionProof, we test a pick-and-place task. Under severe motion blur or gripper occlusion, standard vision models declare success. ProofCausality flags the Laplacian blur variance and emits UNCERTAIN."* | Scenario selector clicking 'Motion Blur', UNCERTAIN badge appearing, showing Laplacian variance 34.2 < 100.0. |
| **1:35 - 2:05** | ReCapture Recovery Loop | *"Instead of generic advice like 'record again', ReCapture issues a targeted physical directive: 'Hold camera steady and allow autofocus lock.' Once re-captured, the system re-evaluates to a verified SUCCESS."* | ReCapture card displaying exact directive, clicking 'Run Guided ReCapture', verdict transitioning to SUCCESS. |
| **2:05 - 2:30** | Benchmark & Evidence Receipts | *"Across 25 standardized physical action cases, ProofCausality reduces the False-Success Rate from 80% to 0%, resolving 100% of uncertain cases. Every verdict generates a canonical SHA-256 Evidence Receipt."* | Benchmark tab showing 0.0% FSR card, 100% RRR card, and interactive JSON Evidence Receipt viewer. |
| **2:30 - 2:45** | Conclusion & Meta Hackathon Close | *"ProofCausality AI: Prove what happened. Detect what is missing. Capture the evidence that changes the decision. Built for the Meta Global AI Developer Hackathon 2026."* | GitHub repo link, release v0.1.0 tag, project summary. |

---

## Submission Checklist for Owner (Sunny)
- [ ] Review code at `https://github.com/sunny-kavixo/proofcausality-ai`
- [ ] Record video according to the shot list above (using QuickTime or OBS)
- [ ] Upload video to YouTube (Public or Unlisted)
- [ ] Go to [Cerebral Valley Submission Portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)
- [ ] Paste Title, Tagline, Description, GitHub URL, and YouTube link
- [ ] Click Submit prior to deadline: **November 4, 2026, 9:00 PM EST**
