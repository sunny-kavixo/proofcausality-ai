#!/usr/bin/env python3
"""Explicit, non-destructive acquisition of the ProofCausality COCO pilot.

COCO image files are intentionally excluded from release archives. This command
requires the caller to provide the official ``instances_val2017.json`` metadata
and performs no writes unless ``--write`` is supplied.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "coco_val2017_subset"
COCO_VAL_BASE_URL = "https://images.cocodataset.org/val2017"
PILOT_IMAGE_IDS = [397133, 37777, 252219, 87038, 174482, 403385, 6818, 480985, 458054, 331352,
                   296649, 386912, 502136, 491497, 184791, 348881, 289393, 522713, 181666, 17627]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def redistribution_status(license_id: int | None) -> str:
    if license_id == 4:
        return "PERMITTED_WITH_ATTRIBUTION"
    if license_id in {1, 3, 6}:
        return "RESTRICTED_REVIEW_SOURCE_LICENSE"
    return "UNKNOWN_DO_NOT_REDISTRIBUTE"


def load_subset(annotations_path: Path) -> tuple[dict, list[dict], list[dict]]:
    raw = json.loads(annotations_path.read_text(encoding="utf-8"))
    target_ids = set(PILOT_IMAGE_IDS)
    images = [item for item in raw.get("images", []) if item.get("id") in target_ids]
    annotations = [item for item in raw.get("annotations", []) if item.get("image_id") in target_ids]
    missing = sorted(target_ids - {item["id"] for item in images})
    if missing:
        raise ValueError(f"Annotation file is not COCO val2017 or lacks pilot IDs: {missing}")
    return raw, images, annotations


def download_atomic(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temp:
        temp_path = Path(temp.name)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "ProofCausality-AI/0.2.6"})
            with urllib.request.urlopen(request, timeout=60) as response:
                shutil.copyfileobj(response, temp)
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
    temp_path.replace(destination)


def acquire(annotations_path: Path, data_dir: Path, replace: bool, base_url: str) -> dict:
    raw, images, annotations = load_subset(annotations_path)
    images_dir = data_dir / "images"
    annotations_output = data_dir / "annotations_coco_pilot20.json"
    manifest_output = data_dir / "acquisition_manifest.json"
    protected = [path for path in (annotations_output, manifest_output) if path.exists()]
    if protected and not replace:
        names = ", ".join(str(path) for path in protected)
        raise FileExistsError(f"Refusing to replace existing generated files: {names}; pass --replace explicitly")

    licenses = {item["id"]: item for item in raw.get("licenses", [])}
    categories = {item["id"]: item.get("name", "unknown") for item in raw.get("categories", [])}
    acquired_at = datetime.now(timezone.utc).isoformat()
    entries = []
    for metadata in images:
        filename = metadata["file_name"]
        destination = images_dir / filename
        url = f"{base_url.rstrip('/')}/{filename}"
        if not destination.exists():
            print(f"download {url}")
            download_atomic(url, destination)
        digest = sha256_file(destination)
        with Image.open(destination) as image:
            width, height = image.size
            image_format = image.format
        if (width, height) != (metadata.get("width"), metadata.get("height")):
            raise ValueError(f"Dimension mismatch for {filename}")
        license_id = metadata.get("license")
        license_record = licenses.get(license_id, {})
        image_annotations = [item for item in annotations if item["image_id"] == metadata["id"]]
        entries.append({"official_image_id": metadata["id"], "filename": filename, "source_url": url,
                        "coco_url": metadata.get("coco_url"), "flickr_url": metadata.get("flickr_url"),
                        "license_id": license_id, "license_name": license_record.get("name", "UNKNOWN"),
                        "license_url": license_record.get("url"),
                        "redistribution_status": redistribution_status(license_id), "sha256": digest,
                        "width": width, "height": height, "format": image_format,
                        "size_bytes": destination.stat().st_size, "annotation_count": len(image_annotations),
                        "categories": sorted({categories.get(item["category_id"], "unknown") for item in image_annotations})})

    subset = {"info": {"description": "ProofCausality COCO 2017 val pilot subset", "created": acquired_at},
              "licenses": raw.get("licenses", []), "categories": raw.get("categories", []),
              "images": images, "annotations": annotations}
    manifest = {"schema_version": 1, "acquired_at": acquired_at,
                "annotations_source": str(annotations_path.resolve()),
                "annotations_sha256": sha256_file(annotations_path),
                "notice": "Licenses are copied from official COCO metadata per image; UNKNOWN entries must not be redistributed.",
                "images": entries}
    data_dir.mkdir(parents=True, exist_ok=True)
    annotations_output.write_text(json.dumps(subset, indent=2) + "\n", encoding="utf-8")
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations-json", type=Path, required=True, help="Official instances_val2017.json path")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--base-url", default=COCO_VAL_BASE_URL, help="HTTPS image origin")
    parser.add_argument("--write", action="store_true", help="Perform network and filesystem writes")
    parser.add_argument("--replace", action="store_true", help="Replace generated annotation/manifest files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _raw, images, annotations = load_subset(args.annotations_json)
    print(f"validated {len(images)} pilot image records and {len(annotations)} annotations")
    print(f"destination: {args.data_dir.resolve()}")
    if not args.write:
        print("dry run only; pass --write to download images and generate files")
        return 0
    manifest = acquire(args.annotations_json, args.data_dir.resolve(), args.replace, args.base_url)
    print(f"acquired {len(manifest['images'])} images; manifest: {args.data_dir / 'acquisition_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
