# Meta Global AI Developer Hackathon 2026 — Submission Handoff (v0.2.0)

> **Official Event:** [Meta AI Developer Hackathon (Cerebral Valley)](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)  
> **Submission Track:** Indie Developer  
> **Theme:** Push the Frontier  
> **Secondary Track Relevance:** Build for Your World (Physical AI task verification, robotics demonstration QC & visual annotation auditing)  
> **Deadline:** November 4, 2026, 9:00 PM US Eastern  
> **Latest Verified Release:** `v0.2.0` (Phase 2 Evidence Calculus Architecture)  
> **Preserved Historical Releases:** `v0.1.0`, `v0.1.1`  
> **Public GitHub Repository:** [https://github.com/sunny-kavixo/proofcausality-ai](https://github.com/sunny-kavixo/proofcausality-ai)  

---

## Submission Form Copy-Paste Blocks

### Project Title
`ProofCausality AI`

### Tagline
`Prove what happened. Detect what is missing. Capture the evidence that changes the decision.`

### Short Description (under 200 words)
ProofCausality AI is an evidence-aware visual causality and physical-action verification system engineered to eliminate false-success hallucinations in robotics demonstration collection, physical AI task evaluation, and high-consequence vision pipelines.

Standard vision models declare high-confidence success even when crucial interaction frames are occluded, clipped at borders, blurred, or physically contradicted. In Phase 2, ProofCausality AI introduces the **Evidence Calculus Engine**: a rigorous mathematical layer that decomposes tasks into observable predicates with tri-state semantics (`SUPPORTED`, `CONTRADICTED`, `UNOBSERVABLE`), computes **Physics-Constrained Evidence Debt** ($D(E,T)$), and arbitrates decisions over a 6-stage **Directed Evidence Dependency Graph**.

Across 37 held-out benchmark cases (12 curated real-world tabletop pairs + 25 synthetic cases), Evidence Calculus reduced the False-Success Rate from **100.0% (confidence-only baseline)** to **0.0%**, while our **Evidence Budget Planner** achieved a **95.0% resolution rate** under Value-of-Information (VOI) optimization. Native Rust SIMD acceleration delivers an 11.35x speedup on batch mask operations with 100% Python/C-ABI parity. Every decision issues a tamper-evident SHA-256 Merkle Evidence Receipt separating cryptographic integrity from physical ground truth.

### Meta Technologies Used
- **Meta SAM 2 / 2.1 (Segment Anything Model)**: Modular integration via `backend/evidence/meta_adapter.py` for zero-shot prompt-driven mask segmentation and contour agreement.
- **Meta Llama 3.2 Vision (11B/90B)**: Scene context interpretation and structured spatial prompt formatting.
- **EvidenceCore Deterministic Fallback**: 100% offline, zero-cloud fallback implementing pure 2D Laplacian convolution, bounding box geometry, and velocity bounding without black-box dependency.

### Public GitHub Repository URL
`https://github.com/sunny-kavixo/proofcausality-ai`

### Release v0.2.0 URL
`https://github.com/sunny-kavixo/proofcausality-ai/releases/tag/v0.2.0`

### Video Demonstration Script & Shot List (2:55 Target)

| Timestamp | Screen / Scene | Voiceover Narration | Visual Focus |
| :--- | :--- | :--- | :--- |
| **0:00 - 0:25** | System Title & The Problem | *"In physical AI and robotics demonstration collection, standard vision models have a critical flaw: false-success hallucination. When an action is occluded or blurred, models guess."* | Title card, architecture diagram showing Tri-State arbitration. |
| **0:25 - 0:55** | Evidence Calculus & Directed DAG | *"In Phase 2, we introduce the Evidence Calculus Engine. Visual verification is decomposed into explicit observable predicates across a 6-stage Directed Dependency Graph."* | Evidence Calculus tab, interactive DAG SVG showing node transitions from Observations to Verdict. |
| **0:55 - 1:30** | The False-Success Trap & Evidence Debt | *"Here in Scenario 2, an object is moved behind an obstacle. A confidence-only model declares 94% success. ProofCausality calculates Evidence Debt D(E,T) = 2.50, identifies missing critical predicates, and abstains as UNCERTAIN."* | Selecting Scenario 2 (Occluded), Evidence Debt meter showing 2.50, predicate table showing UNOBSERVABLE badges. |
| **1:30 - 2:00** | VOI Budget Planner & Re-Capture | *"When evidence is incomplete, our Evidence Budget Planner uses Value of Information to recommend the exact action that reduces risk at minimal human effort: 'Pan camera 15 cm right to unblock line of sight'."* | Recovery plan card showing VOI score, directive, and human effort seconds. |
| **2:00 - 2:30** | Physical Falsification & Parity | *"If an object instantaneously teleports, our physics constraint triggers immediate falsification—returning FAILED with 0% false-success rate. Our Rust SIMD core delivers an 11.35x speedup with 100% parity."* | Scenario 4 (Contradiction) returning FAILED, Rust benchmark tab showing 11.35x speedup. |
| **2:30 - 2:55** | Evidence Receipts & Hackathon Close | *"Every verdict emits a cryptographic SHA-256 Evidence Receipt. ProofCausality AI: Prove what happened. Detect what is missing. Capture the evidence that changes the decision."* | Interactive JSON Evidence Receipt viewer, GitHub repo link, release v0.2.0 tag. |

---

## Submission Checklist for Owner (Sunny)
- [ ] Review code and release at `https://github.com/sunny-kavixo/proofcausality-ai/releases/tag/v0.2.0`
- [ ] Record video according to the shot list above (using QuickTime or OBS)
- [ ] Upload video to YouTube (Public or Unlisted)
- [ ] Go to [Cerebral Valley Submission Portal](https://cerebralvalley.ai/e/meta-ai-developer-hackathon)
- [ ] Paste Title, Tagline, Description, GitHub URL, and YouTube link
- [ ] Click Submit prior to deadline: **November 4, 2026, 9:00 PM EST**
