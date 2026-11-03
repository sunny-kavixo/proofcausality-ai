"""Reproduce the real-media ProofLoop pair experiment.

The runner uses two manifest-verified, licensed COCO photographs and the real
FastAPI endpoints. It deliberately reports observation/state-change evidence
separately from physical causality: two different photos can show a measured
position difference, but they do not establish that an object moved between
the captures.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from backend.app import app


ROOT = Path(__file__).resolve().parent.parent


def image_data_uri(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def manifest_record(image_id: int) -> dict[str, Any]:
    manifest = json.loads(
        (ROOT / "evaluation/experiments/genuine_model_predictions/dataset_manifest.json").read_text()
    )
    for record in manifest.get("records", []):
        if int(record["official_image_id"]) == image_id:
            return {
                "official_image_id": image_id,
                "filename": record.get("filename") or record.get("file_name"),
                "canonical_source_url": record.get("canonical_source_url"),
                "license_status": record.get("license_status"),
                "license_name": record.get("license_name"),
                "license_url": record.get("license_url"),
                "redistribution_status": record.get("redistribution_status"),
                "original_sha256": record.get("original_sha256"),
                "width": record.get("width") or record.get("image_width"),
                "height": record.get("height") or record.get("image_height"),
            }
    raise ValueError(f"Manifest record not found for image {image_id}")


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def box_from_payload(value: dict[str, Any]) -> dict[str, float]:
    return {
        "x_min": float(value["x_min"]),
        "y_min": float(value["y_min"]),
        "x_max": float(value["x_max"]),
        "y_max": float(value["y_max"]),
    }


def run_case(
    client: TestClient,
    *,
    case_id: str,
    before_image: Path,
    before_manifest_id: int,
    followup_image: Path,
    followup_manifest_id: int,
    primary_object: str,
    target_region: dict[str, float],
    require_rest_proof: bool,
) -> dict[str, Any]:
    start = client.post(
        "/api/proofloop/media/start",
        json={
            "task_id": f"MEDIA_BENCH_{case_id.upper()}",
            "primary_object": primary_object,
            "target_region": target_region,
            "image_base64": image_data_uri(before_image),
            "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
            "manifest_image_id": before_manifest_id,
            "require_rest_proof": require_rest_proof,
        },
    )
    if start.status_code != 200:
        raise RuntimeError(f"{case_id}: start failed with {start.status_code}: {start.text}")
    initial = start.json()
    followup = client.post(
        "/api/proofloop/media/reverify",
        json={
            "session_id": initial["session_id"],
            "image_base64": image_data_uri(followup_image),
            "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
            "manifest_image_id": followup_manifest_id,
        },
    )
    if followup.status_code != 200:
        raise RuntimeError(f"{case_id}: follow-up failed with {followup.status_code}: {followup.text}")
    final = followup.json()
    initial_media = final["media_evidence"]["initial"]
    followup_media = final["media_evidence"]["followup"]
    return {
        "case_id": case_id,
        "initial_semantic_verdict": initial["semantic_verdict"],
        "followup_semantic_verdict": final["semantic_verdict"],
        "followup_legacy_verdict": final["legacy_action_verdict"],
        "state_trace": final["state_trace"],
        "receipt_valid": final["receipt_valid"],
        "initial_image_sha256": initial_media["image_sha256"],
        "followup_image_sha256": followup_media["image_sha256"],
        "different_input_bytes": initial_media["image_sha256"] != followup_media["image_sha256"],
        "initial_server_bbox": initial_media["matching_prediction"]["bbox"],
        "followup_server_bbox_raw": followup_media["matching_prediction"]["bbox"],
        "followup_server_bbox_initial_frame": followup_media["coordinate_transform"]["normalized_server_bbox"],
        "coordinate_transform": followup_media["coordinate_transform"],
        "server_derived_displacement_px": followup_media["server_derived_displacement_px"],
        "observed_image_state_change": followup_media["observed_image_state_change"],
        "require_rest_proof": require_rest_proof,
        "structured_finding": final["measurements"]["structured_finding"],
        "provenance_status": final["provenance_status"],
        "client_measurements_accepted": final["media_evidence"]["client_measurements_accepted"],
        "model": {
            "name": initial_media["model_execution"]["model_identifier"],
            "revision": initial_media["model_execution"]["model_revision"],
            "checkpoint_sha256": initial_media["model_execution"]["checkpoint_sha256"],
            "initial_device": initial_media["model_execution"]["device"],
            "followup_device": followup_media["model_execution"]["device"],
        },
        "scientific_disclosure": (
            "Different photographs establish independently hashed observations and a measured image-space "
            "difference. They do not establish temporal continuity, object identity across frames, or physical causality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before-image", type=Path, default=ROOT / "data/coco_val2017_subset/images/000000174482.jpg")
    parser.add_argument("--before-manifest-id", type=int, default=174482)
    parser.add_argument("--followup-image", type=Path, default=ROOT / "data/coco_val2017_subset/images/000000087038.jpg")
    parser.add_argument("--followup-manifest-id", type=int, default=87038)
    parser.add_argument("--primary-object", default="bicycle")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/benchmarks/proofloop_real_media_different_photo_report.json")
    args = parser.parse_args()

    for path in (args.before_image, args.followup_image):
        if not path.exists():
            raise SystemExit(f"Missing local media file: {path}")

    client = TestClient(app)
    broad_target = {"x_min": 0.0, "y_min": 0.0, "x_max": 640.0, "y_max": 640.0}
    supported = run_case(
        client,
        case_id="different_photo_supported",
        before_image=args.before_image,
        before_manifest_id=args.before_manifest_id,
        followup_image=args.followup_image,
        followup_manifest_id=args.followup_manifest_id,
        primary_object=args.primary_object,
        target_region=broad_target,
        require_rest_proof=False,
    )
    initial_bbox = supported["initial_server_bbox"]
    contradiction_target = {
        "x_min": max(0.0, initial_bbox["x_max"] - 60.0),
        "y_min": initial_bbox["y_min"],
        "x_max": initial_bbox["x_max"],
        "y_max": min(640.0, initial_bbox["y_min"] + 60.0),
    }
    contradicted = run_case(
        client,
        case_id="different_photo_contradicted_by_claim",
        before_image=args.before_image,
        before_manifest_id=args.before_manifest_id,
        followup_image=args.followup_image,
        followup_manifest_id=args.followup_manifest_id,
        primary_object=args.primary_object,
        target_region=contradiction_target,
        require_rest_proof=False,
    )
    unresolved = run_case(
        client,
        case_id="different_photo_unresolved_without_rest_evidence",
        before_image=args.before_image,
        before_manifest_id=args.before_manifest_id,
        followup_image=args.followup_image,
        followup_manifest_id=args.followup_manifest_id,
        primary_object=args.primary_object,
        target_region=broad_target,
        require_rest_proof=True,
    )

    report = {
        "benchmark": "ProofLoop real-media different-photo pair",
        "benchmark_version": "proofloop-media-v1",
        "status": "PASS",
        "source_pair": {
            "before": manifest_record(args.before_manifest_id),
            "followup": manifest_record(args.followup_manifest_id),
            "before_file": relative_path(args.before_image),
            "followup_file": relative_path(args.followup_image),
            "primary_object": args.primary_object,
            "license_scope": "Manifest-verified COCO records; source-specific attribution and redistribution restrictions apply.",
        },
        "cases": [supported, contradicted, unresolved],
        "scientific_scope": {
            "real_media": True,
            "different_input_hashes": supported["different_input_bytes"],
            "human_confirmed_labels": False,
            "temporal_observation": False,
            "causal_validation": False,
            "real_world_recovery_accuracy_claim": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({
        "status": report["status"],
        "cases": [
            {"case_id": case["case_id"], "followup_semantic_verdict": case["followup_semantic_verdict"], "displacement_px": case["server_derived_displacement_px"]}
            for case in report["cases"]
        ],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
