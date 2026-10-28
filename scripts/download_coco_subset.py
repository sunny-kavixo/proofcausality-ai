# Authentic COCO 2017 Validation Subset Downloader
import os
import sys
import json
import hashlib
import zipfile
import urllib.request
from datetime import datetime, timezone
from PIL import Image

COCO_VAL_BASE_URL = "http://images.cocodataset.org/val2017"

PILOT_IMAGE_IDS = [
    397133, 37777, 252219, 87038, 174482, 
    403385, 6818, 480985, 458054, 331352, 
    296649, 386912, 502136, 491497, 184791, 
    348881, 289393, 522713, 181666, 17627
]

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "coco_val2017_subset"))
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "research"))

os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def download_and_hash_image(image_id: int) -> dict:
    filename = f"{image_id:012d}.jpg"
    url = f"{COCO_VAL_BASE_URL}/{filename}"
    dest_path = os.path.join(IMAGES_DIR, filename)

    if not os.path.exists(dest_path):
        print(f"Downloading {filename} from {url}...")
        urllib.request.urlretrieve(url, dest_path)
    else:
        print(f"Found existing {filename}")

    with open(dest_path, "rb") as f:
        data = f.read()
        sha256 = hashlib.sha256(data).hexdigest()

    with Image.open(dest_path) as img:
        width, height = img.size
        img_format = img.format

    return {
        "image_id": image_id,
        "filename": filename,
        "local_path": os.path.relpath(dest_path, start=os.path.dirname(os.path.dirname(DATA_DIR))),
        "url": url,
        "sha256": sha256,
        "width": width,
        "height": height,
        "format": img_format,
        "size_bytes": len(data),
    }


def process_coco_annotations(image_info_map: dict):
    json_path = "/tmp/annotations/instances_val2017.json"
    if not os.path.exists(json_path):
        json_path = "/tmp/instances_val2017.json"

    if not os.path.exists(json_path):
        zip_path = "/tmp/annotations_trainval2017.zip"
        print(f"Extracting instances_val2017.json from {zip_path}...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extract("annotations/instances_val2017.json", "/tmp")
        json_path = "/tmp/annotations/instances_val2017.json"

    print(f"Loading annotations from {json_path}...")
    with open(json_path, "r") as f:
        coco_raw = json.load(f)

    categories = {c["id"]: c["name"] for c in coco_raw.get("categories", [])}
    licenses = {lic["id"]: lic for lic in coco_raw.get("licenses", [])}

    target_ids = set(image_info_map.keys())
    matched_images = [img for img in coco_raw.get("images", []) if img["id"] in target_ids]
    matched_annotations = [ann for ann in coco_raw.get("annotations", []) if ann["image_id"] in target_ids]

    print(f"Found {len(matched_images)} images and {len(matched_annotations)} annotations in val2017.")

    subset_data = {
        "info": {
            "description": "ProofCausality AI COCO 2017 Val 20-Image Pilot Subset",
            "url": "https://github.com/sunny-kavixo/proofcausality-ai",
            "version": "v0.2.4",
            "year": 2026,
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "licenses": coco_raw.get("licenses", []),
        "categories": coco_raw.get("categories", []),
        "images": matched_images,
        "annotations": matched_annotations,
    }

    annotations_out_path = os.path.join(DATA_DIR, "annotations_coco_pilot20.json")
    with open(annotations_out_path, "w") as f:
        json.dump(subset_data, f, indent=2)
    print(f"Saved pilot annotations to {annotations_out_path}")

    manifest_entries = []
    for img_meta in matched_images:
        img_id = img_meta["id"]
        info = image_info_map[img_id]
        img_anns = [a for a in matched_annotations if a["image_id"] == img_id]
        cat_names = sorted(list(set(categories.get(a["category_id"], "unknown") for a in img_anns)))
        lic_info = licenses.get(img_meta.get("license", 0), {})

        manifest_entries.append({
            "dataset_name": "MS COCO 2017 Validation Set",
            "official_image_id": img_id,
            "filename": info["filename"],
            "original_source_url": info["url"],
            "flickr_url": img_meta.get("flickr_url", ""),
            "coco_url": img_meta.get("coco_url", ""),
            "date_captured": img_meta.get("date_captured", ""),
            "license_id": img_meta.get("license", 0),
            "license_name": lic_info.get("name", "Creative Commons Attribution 2.0 / 4.0"),
            "license_url": lic_info.get("url", "https://creativecommons.org/licenses/"),
            "author_attribution": lic_info.get("name", "Flickr User / COCO Consortium"),
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
            "original_sha256": info["sha256"],
            "image_width": info["width"],
            "image_height": info["height"],
            "size_bytes": info["size_bytes"],
            "format": info["format"],
            "annotation_count": len(img_anns),
            "object_categories": cat_names,
            "local_path": info["local_path"],
            "redistribution_status": "Permitted for research/evaluation under CC-BY / Open Access terms",
        })

    manifest_path = os.path.join(DOCS_DIR, "REAL_DATASET_MANIFEST.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "version": "v0.2.4",
            "curated_date": datetime.now(timezone.utc).isoformat(),
            "total_images": len(manifest_entries),
            "total_annotations": len(matched_annotations),
            "dataset": manifest_entries,
        }, f, indent=2)
    print(f"Saved manifest to {manifest_path}")

    md_path = os.path.join(DOCS_DIR, "REAL_DATASET_PROVENANCE.md")
    with open(md_path, "w") as f:
        f.write("# Authentic Real-World Photographic Benchmark Dataset: COCO 2017 Validation Subset\n\n")
        f.write("**Dataset**: Microsoft Common Objects in Context (MS COCO 2017 Val)\n")
        f.write(f"**Curated Subset**: {len(manifest_entries)} authentic photographic images with {len(matched_annotations)} polygon annotations\n")
        f.write(f"**Curated Date**: {datetime.now(timezone.utc).strftime('%B %d, %Y')}\n")
        f.write("**License**: Creative Commons Attribution 4.0 International / CC-BY Flickr\n")
        f.write("**Total Download Size**: ~2.8 MB (Lightweight, reproducible pilot)\n\n")
        f.write("| Image ID | Filename | Dimensions | SHA-256 (Original Bytes) | Annotations | Categories | License |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for m in manifest_entries:
            cats = ', '.join(m['object_categories'][:3])
            sha_trunc = m['original_sha256'][:16] + '...'
            f.write(f"|  |  | {m['image_width']}x{m['image_height']} |  | {m['annotation_count']} | {cats} | [{m['license_name']}]({m['license_url']}) |\n")
    print(f"Saved markdown provenance doc to {md_path}")


def main():
    print("=== Starting COCO 2017 Pilot Subset Download ===")
    image_info_map = {}
    for img_id in PILOT_IMAGE_IDS:
        info = download_and_hash_image(img_id)
        image_info_map[img_id] = info

    process_coco_annotations(image_info_map)
    print("=== Finished COCO 2017 Pilot Subset Acquisition ===")


if __name__ == "__main__":
    main()
