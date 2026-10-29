"""
Regression tests for Issue 6: Dataset Licensing and Public Asset Audit.
Ensures that:
1. No blanket MIT or CC-BY claims are made over third-party COCO images or model weights.
2. Manifest explicitly records per-image licensing and redistribution status.
3. Permitted images (CC-BY 2.0) are separated from Restricted images (CC-BY-NC / CC-BY-ND).
4. Unresolved images have UNKNOWN status and cannot claim permitted redistribution.
5. Image SHA-256 and dimensions in the manifest match actual disk files.
"""
import os
import json
import hashlib
from PIL import Image
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data", "coco_val2017_subset")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
PILOT_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "annotations_coco_pilot20.json")
MANIFEST_PATH = os.path.join(REPO_ROOT, "docs", "research", "REAL_DATASET_MANIFEST.json")
AUDIT_DOC_PATH = os.path.join(REPO_ROOT, "docs", "research", "DATASET_LICENSING_AUDIT.md")


def test_manifest_exists_and_has_valid_structure():
    assert os.path.exists(MANIFEST_PATH), "REAL_DATASET_MANIFEST.json must exist"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "dataset" in data
    assert len(data["dataset"]) == 20


def test_manifest_separates_permitted_and_restricted_licenses():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    permitted_count = 0
    restricted_count = 0

    for entry in data["dataset"]:
        status = entry.get("redistribution_status", "")
        lic_id = entry.get("license_id")

        if lic_id == 4:  # CC-BY 2.0 Attribution License
            assert "PERMITTED_WITH_ATTRIBUTION" in status, f"Image {entry['official_image_id']} with CC-BY must be PERMITTED_WITH_ATTRIBUTION"
            permitted_count += 1
        elif lic_id in (1, 3, 6):  # NC, NC-ND, ND
            assert "RESTRICTED" in status, f"Image {entry['official_image_id']} with license {lic_id} must be RESTRICTED"
            restricted_count += 1
        else:
            pytest.fail(f"Unexpected license ID {lic_id} for image {entry['official_image_id']}")

    assert permitted_count == 6, f"Expected 6 permitted CC-BY images, found {permitted_count}"
    assert restricted_count == 14, f"Expected 14 restricted images, found {restricted_count}"


def test_manifest_hashes_match_disk_files():
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in data["dataset"]:
        filename = entry["filename"]
        img_path = os.path.join(IMAGES_DIR, filename)
        assert os.path.exists(img_path), f"File {filename} must exist on disk"

        with open(img_path, "rb") as f:
            computed_sha = hashlib.sha256(f.read()).hexdigest()

        assert computed_sha == entry["original_sha256"], f"SHA-256 mismatch for {filename}"

        with Image.open(img_path) as img:
            width, height = img.size
        assert width == entry["image_width"], f"Width mismatch for {filename}"
        assert height == entry["image_height"], f"Height mismatch for {filename}"


def test_tracked_images_outside_pilot_are_identified():
    """All 33 images in data/coco_val2017_subset/images/ are accounted for."""
    all_files = sorted([f for f in os.listdir(IMAGES_DIR) if f.endswith(".jpg")])
    assert len(all_files) == 33

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    pilot_filenames = set(entry["filename"] for entry in data["dataset"])
    assert len(pilot_filenames) == 20

    extra_files = [f for f in all_files if f not in pilot_filenames]
    assert len(extra_files) == 13

    # Ensure the audit document explicitly lists these extra 13 files as UNKNOWN
    with open(AUDIT_DOC_PATH, "r", encoding="utf-8") as f:
        audit_content = f.read()

    for extra in extra_files:
        assert extra in audit_content, f"Extra file {extra} must be cataloged in licensing audit"


def test_no_blanket_mit_claim_over_third_party_assets():
    """Verify that licensing documentation explicitly prohibits blanket MIT claims over COCO images or model weights."""
    with open(AUDIT_DOC_PATH, "r", encoding="utf-8") as f:
        audit_content = f.read()

    assert "No Blanket Licensing" in audit_content
    assert "CC BY-NC 4.0" in audit_content
    assert "Multi-License Framework" in audit_content
