"""Independent, unprompted Mask2Former evaluation on the preserved COCO subset."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image

from backend.evidence.image_processing import create_disagreement_visual_overlay, parse_coco_polygon_to_mask
from backend.providers.local_segmentation import LocalMask2FormerProvider, encode_binary_mask_rle
from evaluation.grouped_statistics import grouped_bootstrap_interval


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "coco_val2017_subset"
ANNOTATIONS = DATA / "annotations_coco_pilot20.json"
OUTPUT = ROOT / "evaluation" / "experiments" / "genuine_model_predictions"


def iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    return float(np.logical_and(left, right).sum() / union) if union else 0.0


def mean_value(rows: list[dict], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows])) if rows else 0.0


def run(output_dir: Path = OUTPUT) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    gallery = output_dir / "failure_gallery"
    gallery.mkdir(exist_ok=True)
    coco = json.loads(ANNOTATIONS.read_text())
    categories = {int(item["id"]): item["name"] for item in coco["categories"]}
    annotations_by_image: dict[int, list[dict]] = {}
    for annotation in coco["annotations"]:
        annotations_by_image.setdefault(int(annotation["image_id"]), []).append(annotation)

    provider = LocalMask2FormerProvider()
    per_annotation: list[dict] = []
    raw_predictions: list[dict] = []
    executions: list[dict] = []
    gallery_count = 0
    for image_meta in coco["images"]:
        image_id = int(image_meta["id"])
        image_path = DATA / "images" / image_meta["file_name"]
        predictions, execution = provider.predict(image_path)
        executions.append(execution.to_dict())
        raw_predictions.append({
            "image_id": image_id,
            "input_sha256": execution.input_sha256,
            "output_sha256": execution.output_sha256,
            "width": execution.input_width,
            "height": execution.input_height,
            "instances": [{
                "instance_id": item.instance_id,
                "label_id": item.label_id,
                "label": item.label,
                "score": item.score,
                "mask_rle_row_major": encode_binary_mask_rle(item.mask),
            } for item in predictions],
        })
        with Image.open(image_path) as opened:
            rgb = np.array(opened.convert("RGB"))
        for annotation in annotations_by_image.get(image_id, []):
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation:
                continue
            gt = parse_coco_polygon_to_mask(segmentation, execution.input_width, execution.input_height)
            if not np.any(gt):
                continue
            category = categories[int(annotation["category_id"])]
            class_agnostic = [(iou(gt, item.mask), item) for item in predictions]
            class_aware = [(score, item) for score, item in class_agnostic if item.label == category]
            best_any_score, best_any = max(class_agnostic, default=(0.0, None), key=lambda pair: pair[0])
            best_class_score, best_class = max(class_aware, default=(0.0, None), key=lambda pair: pair[0])
            area = int(gt.sum())
            scale = "small" if area < 32**2 else ("medium" if area <= 96**2 else "large")
            row = {
                "image_id": image_id,
                "annotation_id": int(annotation["id"]),
                "category": category,
                "area_pixels": area,
                "scale": scale,
                "class_agnostic_best_iou": best_any_score,
                "class_aware_best_iou": best_class_score,
                "class_agnostic_detected_iou50": int(best_any_score >= 0.5),
                "class_aware_detected_iou50": int(best_class_score >= 0.5),
                "matched_prediction_label": None if best_any is None else best_any.label,
                "matched_prediction_score": None if best_any is None else best_any.score,
                "error_category": (
                    "EMPTY_PREDICTION" if not predictions else
                    "CLASS_MISMATCH" if best_any_score >= 0.5 and best_class_score < 0.5 else
                    "LOW_MASK_OVERLAP" if best_class_score < 0.5 else "MATCHED"
                ),
            }
            per_annotation.append(row)
            if row["error_category"] != "MATCHED" and gallery_count < 12:
                candidate = best_any.mask if best_any is not None else np.zeros_like(gt)
                overlay = create_disagreement_visual_overlay(rgb, gt, candidate)
                artifact = gallery / f"image_{image_id}_annotation_{annotation['id']}_{row['error_category'].lower()}.png"
                Image.fromarray(overlay).save(artifact)
                row["artifact_path"] = str(artifact.relative_to(ROOT))
                gallery_count += 1

    def ci(key: str) -> list[float]:
        low, high = grouped_bootstrap_interval(
            per_annotation, lambda sample: mean_value(list(sample), key), iterations=2000, seed=20261028
        )
        return [round(low, 4), round(high, 4)]

    latencies = [float(item["latency_ms"]) for item in executions]
    summary = {
        "evidence_category": "GENUINE_MODEL_PREDICTIONS",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "dataset": "MS COCO 2017 validation subset",
        "unique_images": len(coco["images"]),
        "annotations_evaluated": len(per_annotation),
        "derived_scenarios": 0,
        "model": executions[0] if executions else None,
        "license": {
            "implementation": "Mostly MIT; Deformable-DETR portions Apache-2.0",
            "weights": "CC BY-NC 4.0 per upstream Mask2Former model zoo",
            "hugging_face_card_field": "other",
        },
        "metrics": {
            "class_agnostic_mean_best_iou": round(mean_value(per_annotation, "class_agnostic_best_iou"), 4),
            "class_agnostic_mean_best_iou_image_grouped_ci95": ci("class_agnostic_best_iou"),
            "class_aware_mean_best_iou": round(mean_value(per_annotation, "class_aware_best_iou"), 4),
            "class_aware_mean_best_iou_image_grouped_ci95": ci("class_aware_best_iou"),
            "class_agnostic_recall_iou50": round(mean_value(per_annotation, "class_agnostic_detected_iou50"), 4),
            "class_agnostic_recall_iou50_image_grouped_ci95": ci("class_agnostic_detected_iou50"),
            "class_aware_recall_iou50": round(mean_value(per_annotation, "class_aware_detected_iou50"), 4),
            "class_aware_recall_iou50_image_grouped_ci95": ci("class_aware_detected_iou50"),
        },
        "latency_ms": {
            "samples": len(latencies),
            "p50": round(float(np.percentile(latencies, 50)), 3),
            "p95": round(float(np.percentile(latencies, 95)), 3),
        },
        "limitations": [
            "Only 20 independent source images; intervals are preliminary.",
            "COCO ground truth is used only after inference for evaluation, never as model input.",
            "Transformers 5.17 reported two newly initialized Swin layer-normalization parameters while loading this converted checkpoint.",
            "This measures mask/object recall, not general photographic truth or physical action completion.",
        ],
    }
    (output_dir / "raw_predictions.json").write_text(json.dumps(raw_predictions, indent=2))
    (output_dir / "execution_records.json").write_text(json.dumps(executions, indent=2))
    (output_dir / "per_annotation_results.json").write_text(json.dumps(per_annotation, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "per_annotation_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in per_annotation for key in row}))
        writer.writeheader()
        writer.writerows(per_annotation)
    annotation_sha = hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest()
    historical_manifest = json.loads((ROOT / "docs" / "research" / "REAL_DATASET_MANIFEST.json").read_text())
    provenance_by_id = {int(item["official_image_id"]): item for item in historical_manifest["dataset"]}
    manifest = {
        "schema_version": "2.0",
        "dataset_release": "MS COCO 2017 validation",
        "annotation_file": str(ANNOTATIONS.relative_to(ROOT)),
        "annotation_file_sha256": annotation_sha,
        "acquisition_procedure": "Downloaded from each recorded images.cocodataset.org val2017 canonical URL; SHA-256 verified locally.",
        "preprocessing_history": ["No image preprocessing; inference processor resizes an in-memory copy only."],
        "records": [{
            "official_image_id": int(image["id"]),
            "filename": image["file_name"],
            "annotation_image_id": int(image["id"]),
            "original_sha256": hashlib.sha256((DATA / "images" / image["file_name"]).read_bytes()).hexdigest(),
            "width": int(image["width"]),
            "height": int(image["height"]),
            "canonical_source_url": provenance_by_id[int(image["id"])]["original_source_url"],
            "license_status": "KNOWN" if provenance_by_id[int(image["id"])].get("license_url") else "UNKNOWN",
            "license_name": provenance_by_id[int(image["id"])].get("license_name"),
            "license_url": provenance_by_id[int(image["id"])].get("license_url"),
            "redistribution_status": "UNKNOWN",
            "annotation_linkage_confirmed": True,
        } for image in coco["images"]],
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2))
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

