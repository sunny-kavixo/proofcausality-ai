"""Run the predeclared 13-image Phase 10 COCO holdout benchmark.

The selection is every locally preserved COCO val2017 image whose ID is not in
the historical 20-image pilot, ordered numerically. Ground truth is read only
after unprompted inference. Metrics deliberately are not labelled COCO AP.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion

from backend.evidence.image_processing import parse_coco_polygon_to_mask
from backend.providers.local_segmentation import LocalMask2FormerProvider
from evaluation.grouped_statistics import grouped_bootstrap_interval


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "coco_val2017_subset"
PILOT = DATA / "annotations_coco_pilot20.json"
FULL_ANNOTATIONS = Path("/tmp/phase10_instances_val2017_clean.json")
OUTPUT = ROOT / "evaluation" / "experiments" / "phase10" / "holdout"


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def boundary_f1(a: np.ndarray, b: np.ndarray, tolerance: int = 2) -> float:
    a_boundary = np.logical_xor(a, binary_erosion(a))
    b_boundary = np.logical_xor(b, binary_erosion(b))
    if not a_boundary.any() and not b_boundary.any():
        return 1.0
    if not a_boundary.any() or not b_boundary.any():
        return 0.0
    a_hit = np.logical_and(a_boundary, binary_dilation(b_boundary, iterations=tolerance)).sum()
    b_hit = np.logical_and(b_boundary, binary_dilation(a_boundary, iterations=tolerance)).sum()
    precision = float(a_hit / a_boundary.sum())
    recall = float(b_hit / b_boundary.sum())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def greedy_matches(iou_matrix: np.ndarray, labels_ok: np.ndarray, threshold: float) -> list[tuple[int, int, float]]:
    candidates = [
        (float(iou_matrix[g, p]), g, p)
        for g in range(iou_matrix.shape[0])
        for p in range(iou_matrix.shape[1])
        if labels_ok[g, p] and iou_matrix[g, p] >= threshold
    ]
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched: list[tuple[int, int, float]] = []
    for score, gt_index, pred_index in sorted(candidates, reverse=True):
        if gt_index not in used_gt and pred_index not in used_pred:
            used_gt.add(gt_index)
            used_pred.add(pred_index)
            matched.append((gt_index, pred_index, score))
    return matched


def ci(rows: list[dict], key: str) -> list[float]:
    low, high = grouped_bootstrap_interval(
        rows,
        lambda sample: float(np.mean([float(row[key]) for row in sample])) if sample else 0.0,
        iterations=2000,
        seed=20261030,
    )
    return [round(low, 4), round(high, 4)]


def run() -> dict:
    if not FULL_ANNOTATIONS.is_file():
        raise FileNotFoundError(f"Required official COCO annotation JSON missing: {FULL_ANNOTATIONS}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pilot = json.loads(PILOT.read_text())
    full = json.loads(FULL_ANNOTATIONS.read_text())
    pilot_ids = {int(image["id"]) for image in pilot["images"]}
    local_ids = sorted(int(path.stem) for path in (DATA / "images").glob("*.jpg"))
    holdout_ids = [image_id for image_id in local_ids if image_id not in pilot_ids]
    if holdout_ids != [139, 285, 632, 724, 776, 785, 802, 872, 885, 1000, 1268, 1296, 1353]:
        raise RuntimeError(f"Unexpected holdout selection: {holdout_ids}")

    categories = {int(item["id"]): item["name"] for item in full["categories"]}
    licenses = {int(item["id"]): item for item in full["licenses"]}
    image_meta = {int(item["id"]): item for item in full["images"] if int(item["id"]) in holdout_ids}
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in full["annotations"]:
        if int(annotation["image_id"]) in holdout_ids:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    provider = LocalMask2FormerProvider()
    per_gt: list[dict] = []
    per_image: list[dict] = []
    execution_records: list[dict] = []
    prediction_records: list[dict] = []
    all_boundary_scores: list[float] = []
    started = time.time()

    for image_id in holdout_ids:
        meta = image_meta[image_id]
        path = DATA / "images" / meta["file_name"]
        predictions, execution = provider.predict(path)
        execution_records.append(execution.to_dict())
        with Image.open(path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        gt_items: list[tuple[dict, str, np.ndarray]] = []
        for annotation in annotations_by_image[image_id]:
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation:
                continue
            mask = parse_coco_polygon_to_mask(segmentation, rgb.shape[1], rgb.shape[0])
            if mask.any():
                gt_items.append((annotation, categories[int(annotation["category_id"])], mask))

        matrix = np.zeros((len(gt_items), len(predictions)), dtype=float)
        labels_ok = np.zeros_like(matrix, dtype=bool)
        for gt_index, (_, category, gt_mask) in enumerate(gt_items):
            for pred_index, prediction in enumerate(predictions):
                matrix[gt_index, pred_index] = mask_iou(gt_mask, prediction.mask)
                labels_ok[gt_index, pred_index] = category == prediction.label

        matches: dict[str, list[tuple[int, int, float]]] = {}
        for aware, compatibility in (("aware", labels_ok), ("agnostic", np.ones_like(labels_ok, dtype=bool))):
            for threshold in (0.5, 0.75):
                matches[f"{aware}_{threshold}"] = greedy_matches(matrix, compatibility, threshold)

        aware50_by_gt = {g: (p, score) for g, p, score in matches["aware_0.5"]}
        for gt_index, (annotation, category, gt_mask) in enumerate(gt_items):
            any_ious = matrix[gt_index] if len(predictions) else np.array([], dtype=float)
            class_ious = matrix[gt_index, labels_ok[gt_index]] if len(predictions) else np.array([], dtype=float)
            area = int(gt_mask.sum())
            scale = "small" if area < 32**2 else ("medium" if area <= 96**2 else "large")
            boundary_score = None
            if gt_index in aware50_by_gt:
                pred_index, _ = aware50_by_gt[gt_index]
                boundary_score = boundary_f1(gt_mask, predictions[pred_index].mask)
                all_boundary_scores.append(boundary_score)
            per_gt.append({
                "image_id": image_id,
                "annotation_id": int(annotation["id"]),
                "category": category,
                "scale": scale,
                "area_pixels": area,
                "best_class_agnostic_iou": float(any_ious.max()) if any_ious.size else 0.0,
                "best_class_aware_iou": float(class_ious.max()) if class_ious.size else 0.0,
                "matched_class_aware_iou50": int(gt_index in aware50_by_gt),
                "boundary_f1_tolerance_2px_if_matched": boundary_score,
            })

        image_row = {"image_id": image_id, "ground_truth_instances": len(gt_items), "predicted_instances": len(predictions)}
        for key, value in matches.items():
            image_row[f"matches_{key.replace('.', '_')}"] = len(value)
        per_image.append(image_row)
        prediction_records.append({
            "image_id": image_id,
            "input_sha256": execution.input_sha256,
            "output_sha256": execution.output_sha256,
            "predictions": [{"label": item.label, "score": item.score, "area_pixels": int(item.mask.sum())} for item in predictions],
        })

    total_gt = sum(row["ground_truth_instances"] for row in per_image)
    total_pred = sum(row["predicted_instances"] for row in per_image)
    metrics: dict[str, object] = {
        "best_iou_mean_class_aware": round(float(np.mean([row["best_class_aware_iou"] for row in per_gt])), 4),
        "best_iou_median_class_aware": round(float(np.median([row["best_class_aware_iou"] for row in per_gt])), 4),
        "best_iou_mean_class_agnostic": round(float(np.mean([row["best_class_agnostic_iou"] for row in per_gt])), 4),
        "best_iou_median_class_agnostic": round(float(np.median([row["best_class_agnostic_iou"] for row in per_gt])), 4),
        "best_iou_class_aware_image_grouped_ci95": ci(per_gt, "best_class_aware_iou"),
        "boundary_f1_tolerance_2px_on_class_aware_iou50_matches": round(float(np.mean(all_boundary_scores)), 4) if all_boundary_scores else None,
    }
    for aware in ("aware", "agnostic"):
        for threshold in ("0_5", "0_75"):
            tp = sum(row[f"matches_{aware}_{threshold}"] for row in per_image)
            label = f"{aware}_iou{threshold.replace('_', '')}"
            metrics[f"precision_{label}"] = round(tp / total_pred, 4) if total_pred else 0.0
            metrics[f"recall_{label}"] = round(tp / total_gt, 4) if total_gt else 0.0
            metrics[f"false_positive_instances_{label}"] = total_pred - tp
            metrics[f"false_negative_instances_{label}"] = total_gt - tp

    for scale in ("small", "medium", "large"):
        subset = [row for row in per_gt if row["scale"] == scale]
        metrics[f"class_aware_recall_iou50_{scale}"] = round(float(np.mean([row["matched_class_aware_iou50"] for row in subset])), 4) if subset else None
        metrics[f"ground_truth_instances_{scale}"] = len(subset)

    manifest_records = []
    for image_id in holdout_ids:
        meta = image_meta[image_id]
        path = DATA / "images" / meta["file_name"]
        license_item = licenses.get(int(meta.get("license", 0)), {})
        manifest_records.append({
            "image_id": image_id,
            "file_name": meta["file_name"],
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "canonical_source_url": meta.get("coco_url"),
            "flickr_url": meta.get("flickr_url"),
            "license_id": meta.get("license"),
            "license_name": license_item.get("name"),
            "license_url": license_item.get("url"),
            "redistribution_status": "NOT_ASSESSED_DO_NOT_REDISTRIBUTE",
        })

    summary = {
        "status": "PASS",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "selection_rule": "All 13 locally preserved COCO val2017 images not in the historical pilot, sorted by image ID; declared before ground-truth inspection.",
        "image_ids": holdout_ids,
        "unique_images": len(holdout_ids),
        "ground_truth_polygon_instances": total_gt,
        "predicted_instances": total_pred,
        "metrics": metrics,
        "runtime": {
            "wall_seconds": round(time.time() - started, 3),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "latency_ms_p50": round(float(np.median([item["latency_ms"] for item in execution_records])), 3),
            "latency_ms_p95": round(float(np.percentile([item["latency_ms"] for item in execution_records], 95)), 3),
        },
        "methodology": [
            "Unprompted predictions are generated before annotations are read for each image.",
            "Precision/recall use greedy one-to-one matching by descending mask IoU, class-aware or class-agnostic as named.",
            "These are custom instance metrics, not official COCO AP/AP50/AP75.",
            "Crowd/RLE-only ground truths are excluded because this evaluator supports polygon masks only.",
            "The 13-image holdout is independent of the historical 20 images but was already present in the developer-controlled workspace; it is not external validation.",
        ],
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUTPUT / "per_image.json").write_text(json.dumps(per_image, indent=2))
    (OUTPUT / "per_ground_truth.json").write_text(json.dumps(per_gt, indent=2))
    (OUTPUT / "prediction_inventory.json").write_text(json.dumps(prediction_records, indent=2))
    (OUTPUT / "execution_records.json").write_text(json.dumps(execution_records, indent=2))
    (OUTPUT / "dataset_manifest.json").write_text(json.dumps({
        "annotation_source_url": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
        "annotation_archive_sha256": "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268",
        "instances_val2017_sha256": "e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f",
        "records": manifest_records,
    }, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
