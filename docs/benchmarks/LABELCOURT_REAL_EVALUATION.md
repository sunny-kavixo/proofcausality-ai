# LabelCourt Real-World Photographic Benchmark Report

**Dataset**: MS COCO 2017 Validation Set (20 authentic photographic images, 218 annotations)
**Evaluation Date**: 2026-10-28 17:31:30Z
**Splits**: 10 Development Images (unseen during final test) | 10 Held-Out Test Images
**Injected Corruptions**: Controlled spatial shifts, erosion, dilation, missing objects, and class swaps

## 1. Overall Performance Comparison

| Split | System | Precision (95% CI) | Recall (95% CI) | F1 Score | p50 Latency | p95 Latency |
|---|---|---|---|---|---|---|
| **Dev** | **LabelCourt (Ours)** | **97.3%** (95.5% - 98.4%) | **99.2%** (97.9% - 99.7%) | **0.9825** | 16.95 ms | 20.90 ms |
| Dev | Naive BBox Baseline | 98.4% (96.0% - 99.4%) | 51.7% (47.2% - 56.1%) | 0.6776 | 0.49 ms | 0.57 ms |
| **Held-Out Test** | **LabelCourt (Ours)** | **97.5%** (96.0% - 98.5%) | **99.2%** (98.0% - 99.6%) | **0.9833** | 19.19 ms | 24.91 ms |
| Held-Out Test | Naive BBox Baseline | 98.3% (96.1% - 99.3%) | 49.2% (45.2% - 53.2%) | 0.6562 | 0.56 ms | 0.63 ms |

## 2. Detection by Corruption Type (Held-Out Test Split)

| Corruption Category | Total Injected | LabelCourt Detected | Naive Baseline Detected | Defect Description |
|---|---|---|---|---|
| `CLEAN` | 119 | 15 (12.6%) | 5 (4.2%) | Clean authentic annotation (No defect) |
| `SHIFTED_MASK` | 119 | 119 (100.0%) | 106 (89.1%) | Controlled shifted_mask intervention |
| `REDUCED_MASK` | 119 | 114 (95.8%) | 29 (24.4%) | Controlled reduced_mask intervention |
| `ENLARGED_MASK` | 119 | 119 (100.0%) | 34 (28.6%) | Controlled enlarged_mask intervention |
| `CLASS_SWAP` | 119 | 119 (100.0%) | 5 (4.2%) | Controlled class_swap intervention |
| `MISSING_OBJECT` | 119 | 119 (100.0%) | 119 (100.0%) | Controlled missing_object intervention |

## 3. Visual Verification Artifacts

- Sample Disagreement Overlay: `docs/screenshots/labelcourt_coco_sample_overlay.png`
- Full Raw Experiment Log: `evaluation/experiments/LABELCOURT_REAL_BENCHMARK_RESULTS.json`
