# ProofCausality AI — Meta Model Feasibility & Ecosystem Audit

**Audit Date:** October 28, 2026  
**Evaluation Scope:** Meta Model APIs, SAM 2/2.1, Llama 3.2 Vision, Meta Open-Weights  
**Hackathon:** Meta Global AI Developer Hackathon 2026 (Cerebral Valley)  
**Track:** Indie Developer / Push the Frontier  

---

## 1. Feasibility Matrix & Assessment

| Model / API Surface | Official Role | Local / Remote Feasibility | Integration Status | Policy & Safety Note |
| :--- | :--- | :--- | :---: | :--- |
| **Meta SAM 2 / 2.1** (Segment Anything) | Promptable visual mask generation & boundary tracking | Available via open weights or hosted endpoints | **SUPPORTED VIA ADAPTER** | Zero-shot masks; provides raw spatial masks, does not output calibrated uncertainty probabilities. |
| **Llama 3.2 Vision** (11B / 90B) | Multi-modal reasoning & scene understanding | Available via Meta Partner APIs (Together, Fireworks, Replicate, AWS) | **SUPPORTED VIA ADAPTER** | High semantic reasoning; prompts must be constrained with JSON schemas to avoid hallucinated state transitions. |
| **Meta Muse / Generative** | Generative active vision | Specialized research models | **PLANNED EXTENSION** | Not required for deterministic causal verification. |
| **Deterministic EvidenceCore Fallback** | Pure NumPy geometric & spatial calculus | 100% offline, zero-cloud, sub-millisecond execution | **ACTIVE & VERIFIED** | Runs out of the box with zero external API keys or cloud spend. |

---

## 2. Model Adapter Design (`backend/providers/meta_adapter.py`)

To ensure compliance with hackathon rules without incurring unauthorized cloud charges or making brittle assumptions about hosted endpoints, ProofCausality AI implements `MetaAdapter`:
1. **Environment-Driven Configuration:**
   - `META_API_KEY`: API token for hosted Meta / Llama Vision endpoints.
   - `SAM_ENDPOINT_URL`: Optional custom or hosted SAM inference gateway.
   - `META_PROVIDER_ENABLED`: Boolean switch defaulting to `false` for offline evaluation.
2. **Deterministic Geometric Fallback:**
   - When Meta endpoints are unconfigured, `MetaAdapter` leverages EvidenceCore's pure mathematical polygon rasterizer and boundary detector.
   - It **never simulates false API responses** or fabricates model names. Diagnostics truthfully report:
     ```json
     {
       "provider": "EvidenceCore Deterministic Engine",
       "model": "Pure NumPy Geometric Kernel",
       "is_ai_generated": false,
       "configured": false
     }
     ```
3. **Billable Inference Boundary:**
   - Zero billable API calls are made during local testing or evaluation without explicit user authorization.
   - All synthetic and benchmark fixtures operate deterministically and locally.

---

## 3. Findings & Limitations

- **SAM Mask Calibration:** SAM segmentations produce high-fidelity contours but have known failure modes on semi-transparent objects, severe motion blur, and extreme aspect ratio boundaries. EvidenceCore's boundary contact and blur detectors serve as an independent audit layer over SAM masks.
- **Credential Sanitization:** All headers and exception strings passing through `MetaAdapter` are sanitized to prevent secret tokens from appearing in receipts or logs.
