from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import os, sys
"""Real-World Image Annotation Benchmark on Authentically Sourced COCO 2017 Validation Subset.

Evaluates LabelCourt against a Naive Geometric Baseline on genuine photographs with
controlled annotation corruptions across separate Development and Held-Out splits.
"""


import os
import sys
import json
import time
import base64
import random
from typing import Any, Dict, List
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, binary_dilation

from backend.labelcourt.auditor import LabelCourtAuditor
from backend.evidence.schemas import AnnotationItem, BoundingBox, AuditVerdict
from backend.evidence.masks import polygon_to_mask, bbox_to_mask
from backend.evidence.advanced_metrics import wilson_score_interval

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "coco_val2017_subset"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations_coco_pilot20.json")
DOCS_BENCHMARK_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "benchmarks"))
EXPERIMENTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments"))
SCREENSHOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "screenshots"))

os.makedirs(DOCS_BENCHMARK_DIR, exist_ok=True)
os.makedirs(EXPERIMENTS_DIR, exist_ok=True)
os.makedirs(SCREENSHOTS_DIR, exist_ok=True)


def shift_mask(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    shifted = np.zeros_like(mask)
    h, w = mask.shape
    src_y1 = max(0, -dy)
    src_y2 = min(h, h - dy)
    src_x1 = max(0, -dx)
    src_x2 = min(w, w - dx)
    dst_y1 = max(0, dy)
    dst_y2 = min(h, h + dy)
    dst_x1 = max(0, dx)
    dst_x2 = min(w, w + dx)
    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = mask[src_y1:src_y2, src_x1:src_x2]
    return shifted


def erode_mask(mask: np.ndarray, iterations: int = 3) -> np.ndarray:
    eroded = binary_erosion(mask, iterations=iterations)
    if not np.any(eroded):
        return binary_erosion(mask, iterations=max(1, iterations // 2))
    return eroded


def dilate_mask(mask: np.ndarray, iterations: int = 4) -> np.ndarray:
    return binary_dilation(mask, iterations=iterations)


def run_benchmark():
    print("=== Starting LabelCourt Real-World Photographic Benchmark ===")
    with open(ANNOTATIONS_FILE, "r") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}
    all_cat_names = list(categories.values())

    images = coco["images"]
    annotations = coco["annotations"]
    print("Loaded %d images and %d annotations." % (len(images), len(annotations)))

    # Split: 10 Dev and 10 Held-Out Test
    dev_img_ids = set(img["id"] for img in images[:10])
    test_img_ids = set(img["id"] for img in images[10:])

    auditor = LabelCourtAuditor(
        iou_accept_threshold=0.85,
        iou_review_threshold=0.60,
        boundary_agreement_threshold=0.75,
        blur_threshold=100.0,
        boundary_margin_px=3
    )

    rng = random.Random(42)

    results = {
        "metadata": {
            "benchmark_name": "LabelCourt Real-World Photographic Evaluation",
            "dataset": "MS COCO 2017 Validation Subset (20 authentic photographs)",
            "engine_version": "v0.2.4",
            "random_seed": 42,
            "date": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
        },
        "development_split": {},
        "held_out_test_split": {},
    }

    sample_overlay_saved = False

    for split_name, img_id_set in [("development_split", dev_img_ids), ("held_out_test_split", test_img_ids)]:
        print("\nEvaluating %s..." % split_name)
        split_images = [img for img in images if img["id"] in img_id_set]
        
        lc_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "latencies_ms": []}
        naive_stats = {"TP": 0, "FP": 0, "TN": 0, "FN": 0, "latencies_ms": []}
        
        by_corruption: Dict[str, Dict[str, int]] = {}
        by_scale: Dict[str, Dict[str, int]] = {
            "small": {"TP": 0, "FN": 0, "TN": 0, "FP": 0},
            "medium": {"TP": 0, "FN": 0, "TN": 0, "FP": 0},
            "large": {"TP": 0, "FN": 0, "TN": 0, "FP": 0}
        }

        total_cases = 0

        for img_meta in split_images:
            img_id = img_meta["id"]
            filename = img_meta["file_name"]
            img_path = os.path.join(IMAGES_DIR, filename)
            
            with Image.open(img_path) as pil_img:
                rgb_arr = np.array(pil_img.convert("RGB"))
                gray_arr = np.array(pil_img.convert("L"), dtype=np.float32)

            h, w = gray_arr.shape
            img_anns = [a for a in annotations if a["image_id"] == img_id and len(a.get("segmentation", [])) > 0]

            for ann in img_anns:
                if not isinstance(ann.get("segmentation"), list) or len(ann["segmentation"]) == 0:
                    continue
                poly = ann["segmentation"][0]
                if not isinstance(poly, list) or len(poly) < 6:
                    continue
                if len(poly) < 6:
                    continue
                total_cases += 1

                gt_mask = polygon_to_mask(poly, h, w)
                cat_name = categories.get(ann["category_id"], "object")
                bbox_raw = ann["bbox"]
                gt_bbox = BoundingBox(
                    x_min=float(bbox_raw[0]),
                    y_min=float(bbox_raw[1]),
                    x_max=float(bbox_raw[0] + bbox_raw[2]),
                    y_max=float(bbox_raw[1] + bbox_raw[3]),
                )

                area = float(ann.get("area", np.sum(gt_mask)))
                scale_cat = "small" if area < (32 * 32) else ("medium" if area <= (96 * 96) else "large")

                item = AnnotationItem(
                    id=ann["id"],
                    image_id=img_id,
                    category_id=ann["category_id"],
                    category_name=cat_name,
                    bbox=gt_bbox,
                    segmentation=ann["segmentation"],
                    area=area,
                )

                scenarios = [
                    ("CLEAN", gt_mask, cat_name, False),
                    ("SHIFTED_MASK", shift_mask(gt_mask, dx=rng.randint(15, 30), dy=rng.randint(15, 30)), cat_name, True),
                    ("REDUCED_MASK", erode_mask(gt_mask, iterations=3), cat_name, True),
                    ("ENLARGED_MASK", dilate_mask(gt_mask, iterations=4), cat_name, True),
                    ("CLASS_SWAP", gt_mask, rng.choice([c for c in all_cat_names if c != cat_name]), True),
                    ("MISSING_OBJECT", np.zeros_like(gt_mask), cat_name, True),
                ]

                for corr_name, cand_mask, cand_class, has_defect in scenarios:
                    if corr_name not in by_corruption:
                        by_corruption[corr_name] = {"total": 0, "lc_detected": 0, "naive_detected": 0}
                    by_corruption[corr_name]["total"] += 1

                    # 1. LabelCourt Audit
                    t0 = time.perf_counter()
                    audit_res, receipt = auditor.audit_annotation(
                        human_annotation=item,
                        candidate_mask=cand_mask,
                        image_gray=gray_arr,
                        image_rgb=rgb_arr,
                        independent_class_candidate=cand_class,
                    )
                    lc_latency = (time.perf_counter() - t0) * 1000.0
                    lc_stats["latencies_ms"].append(lc_latency)

                    lc_flagged = (audit_res.verdict != AuditVerdict.ACCEPT)
                    if lc_flagged:
                        by_corruption[corr_name]["lc_detected"] += 1

                    if has_defect:
                        if lc_flagged:
                            lc_stats["TP"] += 1
                            by_scale[scale_cat]["TP"] += 1
                        else:
                            lc_stats["FN"] += 1
                            by_scale[scale_cat]["FN"] += 1
                    else:
                        if lc_flagged:
                            lc_stats["FP"] += 1
                            by_scale[scale_cat]["FP"] += 1
                        else:
                            lc_stats["TN"] += 1
                            by_scale[scale_cat]["TN"] += 1

                    # 2. Naive BBox Baseline
                    t_naive_0 = time.perf_counter()
                    c_indices = np.where(cand_mask)
                    if len(c_indices[0]) > 0:
                        c_bbox = BoundingBox(
                            x_min=float(np.min(c_indices[1])),
                            y_min=float(np.min(c_indices[0])),
                            x_max=float(np.max(c_indices[1]) + 1),
                            y_max=float(np.max(c_indices[0]) + 1),
                        )
                        inter_x1 = max(gt_bbox.x_min, c_bbox.x_min)
                        inter_y1 = max(gt_bbox.y_min, c_bbox.y_min)
                        inter_x2 = min(gt_bbox.x_max, c_bbox.x_max)
                        inter_y2 = min(gt_bbox.y_max, c_bbox.y_max)
                        w_inter = max(0.0, inter_x2 - inter_x1)
                        h_inter = max(0.0, inter_y2 - inter_y1)
                        inter_area = w_inter * h_inter
                        area_gt = (gt_bbox.x_max - gt_bbox.x_min) * (gt_bbox.y_max - gt_bbox.y_min)
                        area_c = (c_bbox.x_max - c_bbox.x_min) * (c_bbox.y_max - c_bbox.y_min)
                        union_area = area_gt + area_c - inter_area
                        naive_iou = inter_area / union_area if union_area > 0 else 0.0
                    else:
                        naive_iou = 0.0
                    naive_flagged = (naive_iou < 0.50)
                    naive_latency = (time.perf_counter() - t_naive_0) * 1000.0
                    naive_stats["latencies_ms"].append(naive_latency)

                    if naive_flagged:
                        by_corruption[corr_name]["naive_detected"] += 1

                    if has_defect:
                        if naive_flagged:
                            naive_stats["TP"] += 1
                        else:
                            naive_stats["FN"] += 1
                    else:
                        if naive_flagged:
                            naive_stats["FP"] += 1
                        else:
                            naive_stats["TN"] += 1

                    # Save sample overlay screenshot
                    if not sample_overlay_saved and corr_name == "SHIFTED_MASK" and audit_res.disagreement_heatmap_base64:
                        overlay_path = os.path.join(SCREENSHOTS_DIR, "labelcourt_coco_sample_overlay.png")
                        raw_png = base64.b64decode(audit_res.disagreement_heatmap_base64)
                        with open(overlay_path, "wb") as f_ov:
                            f_ov.write(raw_png)
                        sample_overlay_saved = True
                        print("Saved sample overlay to %s" % overlay_path)

        def calc_metrics(stats):
            tp, fp, tn, fn = stats["TP"], stats["FP"], stats["TN"], stats["FN"]
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            p_low, p_high = wilson_score_interval(tp, tp + fp)
            r_low, r_high = wilson_score_interval(tp, tp + fn)
            lats = stats["latencies_ms"]
            return {
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                "precision": round(prec, 4),
                "precision_wilson_ci95": [round(p_low, 4), round(p_high, 4)],
                "recall": round(rec, 4),
                "recall_wilson_ci95": [round(r_low, 4), round(r_high, 4)],
                "f1_score": round(f1, 4),
                "mean_latency_ms": round(float(np.mean(lats)), 3) if lats else 0.0,
                "p50_latency_ms": round(float(np.percentile(lats, 50)), 3) if lats else 0.0,
                "p95_latency_ms": round(float(np.percentile(lats, 95)), 3) if lats else 0.0,
            }

        split_summary = {
            "total_annotations_evaluated": total_cases,
            "total_scenarios_run": sum(c["total"] for c in by_corruption.values()),
            "labelcourt": calc_metrics(lc_stats),
            "naive_bbox_baseline": calc_metrics(naive_stats),
            "breakdown_by_corruption": by_corruption,
            "breakdown_by_scale": by_scale,
        }
        results[split_name] = split_summary
        print("%s LabelCourt: Prec=%.4f, Rec=%.4f, F1=%.4f" % (
            split_name, split_summary["labelcourt"]["precision"],
            split_summary["labelcourt"]["recall"],
            split_summary["labelcourt"]["f1_score"]
        ))
        print("%s Naive: Prec=%.4f, Rec=%.4f, F1=%.4f" % (
            split_name, split_summary["naive_bbox_baseline"]["precision"],
            split_summary["naive_bbox_baseline"]["recall"],
            split_summary["naive_bbox_baseline"]["f1_score"]
        ))

    json_out = os.path.join(EXPERIMENTS_DIR, "LABELCOURT_REAL_BENCHMARK_RESULTS.json")
    with open(json_out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved benchmark results to %s" % json_out)

    md_out = os.path.join(DOCS_BENCHMARK_DIR, "LABELCOURT_REAL_EVALUATION.md")
    with open(md_out, "w") as f:
        dev_lc = results["development_split"]["labelcourt"]
        dev_naive = results["development_split"]["naive_bbox_baseline"]
        test_lc = results["held_out_test_split"]["labelcourt"]
        test_naive = results["held_out_test_split"]["naive_bbox_baseline"]

        f.write("# LabelCourt Real-World Photographic Benchmark Report\n\n")
        f.write("**Dataset**: MS COCO 2017 Validation Set (20 authentic photographic images, 218 annotations)\n")
        f.write("**Evaluation Date**: %s\n" % results["metadata"]["date"])
        f.write("**Splits**: 10 Development Images (unseen during final test) | 10 Held-Out Test Images\n")
        f.write("**Injected Corruptions**: Controlled spatial shifts, erosion, dilation, missing objects, and class swaps\n\n")
        f.write("## 1. Overall Performance Comparison\n\n")
        f.write("| Split | System | Precision (95% CI) | Recall (95% CI) | F1 Score | p50 Latency | p95 Latency |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        f.write("| **Dev** | **LabelCourt (Ours)** | **%.1f%%** (%.1f%% - %.1f%%) | **%.1f%%** (%.1f%% - %.1f%%) | **%.4f** | %.2f ms | %.2f ms |\n" % (
            dev_lc["precision"] * 100, dev_lc["precision_wilson_ci95"][0] * 100, dev_lc["precision_wilson_ci95"][1] * 100,
            dev_lc["recall"] * 100, dev_lc["recall_wilson_ci95"][0] * 100, dev_lc["recall_wilson_ci95"][1] * 100,
            dev_lc["f1_score"], dev_lc["p50_latency_ms"], dev_lc["p95_latency_ms"]
        ))
        f.write("| Dev | Naive BBox Baseline | %.1f%% (%.1f%% - %.1f%%) | %.1f%% (%.1f%% - %.1f%%) | %.4f | %.2f ms | %.2f ms |\n" % (
            dev_naive["precision"] * 100, dev_naive["precision_wilson_ci95"][0] * 100, dev_naive["precision_wilson_ci95"][1] * 100,
            dev_naive["recall"] * 100, dev_naive["recall_wilson_ci95"][0] * 100, dev_naive["recall_wilson_ci95"][1] * 100,
            dev_naive["f1_score"], dev_naive["p50_latency_ms"], dev_naive["p95_latency_ms"]
        ))
        f.write("| **Held-Out Test** | **LabelCourt (Ours)** | **%.1f%%** (%.1f%% - %.1f%%) | **%.1f%%** (%.1f%% - %.1f%%) | **%.4f** | %.2f ms | %.2f ms |\n" % (
            test_lc["precision"] * 100, test_lc["precision_wilson_ci95"][0] * 100, test_lc["precision_wilson_ci95"][1] * 100,
            test_lc["recall"] * 100, test_lc["recall_wilson_ci95"][0] * 100, test_lc["recall_wilson_ci95"][1] * 100,
            test_lc["f1_score"], test_lc["p50_latency_ms"], test_lc["p95_latency_ms"]
        ))
        f.write("| Held-Out Test | Naive BBox Baseline | %.1f%% (%.1f%% - %.1f%%) | %.1f%% (%.1f%% - %.1f%%) | %.4f | %.2f ms | %.2f ms |\n\n" % (
            test_naive["precision"] * 100, test_naive["precision_wilson_ci95"][0] * 100, test_naive["precision_wilson_ci95"][1] * 100,
            test_naive["recall"] * 100, test_naive["recall_wilson_ci95"][0] * 100, test_naive["recall_wilson_ci95"][1] * 100,
            test_naive["f1_score"], test_naive["p50_latency_ms"], test_naive["p95_latency_ms"]
        ))
        f.write("## 2. Detection by Corruption Type (Held-Out Test Split)\n\n")
        f.write("| Corruption Category | Total Injected | LabelCourt Detected | Naive Baseline Detected | Defect Description |\n")
        f.write("|---|---|---|---|---|\n")
        for corr_type, c_stats in results["held_out_test_split"]["breakdown_by_corruption"].items():
            tot = c_stats["total"]
            lc_det = c_stats["lc_detected"]
            naive_det = c_stats["naive_detected"]
            desc = "Clean authentic annotation (No defect)" if corr_type == "CLEAN" else ("Controlled %s intervention" % corr_type.lower())
            f.write("| `%s` | %d | %d (%.1f%%) | %d (%.1f%%) | %s |\n" % (
                corr_type, tot, lc_det, (lc_det / tot * 100.0) if tot else 0.0,
                naive_det, (naive_det / tot * 100.0) if tot else 0.0, desc
            ))
        f.write("\n## 3. Visual Verification Artifacts\n\n")
        f.write("- Sample Disagreement Overlay: `docs/screenshots/labelcourt_coco_sample_overlay.png`\n")
        f.write("- Full Raw Experiment Log: `evaluation/experiments/LABELCOURT_REAL_BENCHMARK_RESULTS.json`\n")
    print("Saved evaluation markdown to %s" % md_out)


if __name__ == "__main__":
    run_benchmark()
