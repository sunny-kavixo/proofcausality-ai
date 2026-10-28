# LabelCourt Controlled Annotation-Corruption Benchmark on Real Photographs

> This benchmark uses authentic image pixels and COCO annotations, but candidate masks are derived from the reference masks. It does **not** measure independent AI perception or general segmentation accuracy.

**Dataset**: MS COCO 2017 Validation Set (20 authentic photographic images, 218 annotations)
**Evaluation Date**: 2026-10-28 19:21:38Z
**Splits**: 10 Development Images (unseen during final test) | 10 Held-Out Test Images
**Injected Corruptions**: Controlled spatial shifts, erosion, dilation, missing objects, and class swaps

**Uncertainty**: Image-grouped bootstrap intervals (2,000 resamples, seed 20261028) are authoritative; scenario-level Wilson intervals are retained only as historical descriptive statistics. With 10 held-out source images, results are preliminary.

## 1. Overall Performance Comparison

| Split | System | Precision (95% CI) | Recall (95% CI) | F1 Score | p50 Latency | p95 Latency |
|---|---|---|---|---|---|---|
| **Dev** | **LabelCourt (Ours)** | **97.3%** (95.5% - 98.4%) | **99.2%** (97.9% - 99.7%) | **0.9825** | 16.75 ms | 20.67 ms |
| Dev | Naive BBox Baseline | 98.4% (96.0% - 99.4%) | 52.3% (47.8% - 56.7%) | 0.6830 | 0.47 ms | 0.56 ms |
| **Held-Out Test** | **LabelCourt (Ours)** | **97.5%** (95.9% - 98.5%) | **99.0%** (97.8% - 99.5%) | **0.9825** | 18.08 ms | 22.63 ms |
| Held-Out Test | Naive BBox Baseline | 98.4% (96.2% - 99.3%) | 50.4% (46.4% - 54.4%) | 0.6667 | 0.52 ms | 0.59 ms |

## 2. Detection by Corruption Type (Held-Out Test Split)

| Corruption Category | Total Injected | LabelCourt Detected | Naive Baseline Detected | Defect Description |
|---|---|---|---|---|
| `CLEAN` | 119 | 15 (12.6%) | 5 (4.2%) | Clean authentic annotation (No defect) |
| `SHIFTED_MASK` | 119 | 119 (100.0%) | 106 (89.1%) | Controlled shifted_mask intervention |
| `REDUCED_MASK` | 119 | 113 (95.0%) | 36 (30.3%) | Controlled reduced_mask intervention |
| `ENLARGED_MASK` | 119 | 119 (100.0%) | 34 (28.6%) | Controlled enlarged_mask intervention |
| `CLASS_SWAP` | 119 | 119 (100.0%) | 5 (4.2%) | Controlled class_swap intervention |
| `MISSING_OBJECT` | 119 | 119 (100.0%) | 119 (100.0%) | Controlled missing_object intervention |

## 3. Visual Verification Artifacts

- Sample Disagreement Overlay: `docs/screenshots/labelcourt_coco_sample_overlay.png`
- Full Raw Experiment Log: `evaluation/experiments/LABELCOURT_REAL_BENCHMARK_RESULTS.json`
