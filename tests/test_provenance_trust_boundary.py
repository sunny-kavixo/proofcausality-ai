"""Regression tests for server-authoritative provenance verification (Issue 1 Trust-Boundary).

Verifies that:
- A request cannot grant itself trusted provenance.
- A source is classified as MANIFEST_VERIFIED_DATASET_IMAGE only after server independently confirms:
    * image bytes match manifest expected SHA-256
    * official image ID exists
    * dimensions match
    * annotation linkage is valid
    * annotation content corresponds to expected source record
    * canonical dataset source and license metadata are recorded
- Violations produce explicit HTTP 422 errors.
- Unverified user uploads are preserved with unverified label, never promoted to verified.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app import app
from backend.evidence.dataset_manifest import (
    CanonicalDatasetManifestRegistry,
    verify_authoritative_provenance,
)
from backend.evidence.provenance import ProvenanceCategory, ProvenanceGate, ProvenanceViolationError
from backend.evidence.schemas import BoundingBox, VisionAuditAnnotation


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def authentic_coco_sample(client: TestClient) -> dict:
    resp = client.get("/api/vision/samples")
    assert resp.status_code == 200
    samples = resp.json()
    assert len(samples) > 0
    return samples[0]


def _create_synthetic_jpeg_b64(width: int = 640, height: int = 427, color=(200, 100, 50)) -> str:
    img = Image.new("RGB", (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


def test_valid_manifest_backed_image_accepted(client: TestClient, authentic_coco_sample: dict):
    """An authentic image matching manifest hash, dimensions, and ID is granted verified status."""
    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200, resp.text
    prov = resp.json()["provenance"]
    assert prov["image_source_type"] == "MANIFEST_VERIFIED_DATASET_IMAGE"
    assert prov["provenance_category"] == "VERIFIED_DATASET_IMAGE"
    assert prov["is_independently_verified"] is True
    assert prov["canonical_source_url"] is not None
    assert prov["license_status"] in ("KNOWN", "UNKNOWN")


def test_correct_image_with_incorrect_manifest_id_rejected(client: TestClient, authentic_coco_sample: dict):
    """Valid image bytes with wrong manifest ID must be rejected with HTTP 422."""
    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
        # 37777 has different hash than 397133
        "manifest_image_id": 37777 if authentic_coco_sample["manifest_image_id"] != 37777 else 397133,
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 422
    assert "Provenance verification failed" in resp.json()["detail"]
    assert "SHA-256 digest" in resp.json()["detail"]


def test_modified_image_bytes_rejected(client: TestClient, authentic_coco_sample: dict):
    """Subtly modifying authentic image bytes must cause hash mismatch and HTTP 422 rejection."""
    raw_b64 = authentic_coco_sample["image_base64"].split(",", 1)[1]
    raw_bytes = bytearray(base64.b64decode(raw_b64))
    # Flip a byte near the end (avoiding JPEG headers)
    raw_bytes[-10] ^= 0xFF
    tampered_b64 = f"data:image/jpeg;base64,{base64.b64encode(raw_bytes).decode('ascii')}"

    payload = {
        "image_base64": tampered_b64,
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 422
    assert "Provenance verification failed" in resp.json()["detail"]


def test_valid_jpeg_falsely_claiming_verified_origin_rejected(client: TestClient, authentic_coco_sample: dict):
    """A valid, newly-encoded JPEG claiming to be a manifest dataset image must be rejected."""
    synthetic_b64 = _create_synthetic_jpeg_b64(width=640, height=427)
    payload = {
        "image_base64": synthetic_b64,
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 422
    assert "Provenance verification failed" in resp.json()["detail"]


def test_forged_annotation_id_rejected(client: TestClient, authentic_coco_sample: dict):
    """Claiming a verified dataset reference annotation with a forged annotation ID must fail."""
    ann = copy.deepcopy(authentic_coco_sample["annotation"])
    ann["id"] = 999999999  # Non-existent annotation ID
    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": ann,
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
        "annotation_source_type": "PUBLIC_DATASET_ANNOTATION",
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 422
    assert "Provenance verification failed" in resp.json()["detail"]
    assert "Annotation ID" in resp.json()["detail"]


def test_modified_reference_mask_rejected(client: TestClient, authentic_coco_sample: dict):
    """Claiming verified reference annotation with modified coordinates or mask must fail."""
    ann = copy.deepcopy(authentic_coco_sample["annotation"])
    # Shift bounding box coordinates
    ann["bbox"]["x_min"] += 50.0
    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": ann,
        "manifest_image_id": authentic_coco_sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
        "annotation_source_type": "PUBLIC_DATASET_ANNOTATION",
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 422
    assert "Provenance verification failed" in resp.json()["detail"]
    assert "bounding box" in resp.json()["detail"]


def test_missing_manifest_entry_rejected(client: TestClient, authentic_coco_sample: dict):
    """Specifying an unknown manifest ID must fail closed with HTTP 422."""
    payload = {
        "image_base64": authentic_coco_sample["image_base64"],
        "annotation": authentic_coco_sample["annotation"],
        "manifest_image_id": 987654321,  # Not in manifest
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 422
    assert "not found in canonical dataset manifest" in resp.json()["detail"]


def test_mismatched_dimensions_rejected():
    """Even if a hash were somehow forged, dimension mismatch fails verification."""
    registry = CanonicalDatasetManifestRegistry.get_instance()
    rec = registry.get_record(397133)
    assert rec is not None

    res = verify_authoritative_provenance(
        image_bytes=b"dummy",
        image_sha256=rec["original_sha256"],
        width=rec["width"] + 10,  # Mismatched dimension
        height=rec["height"],
        claimed_source_type="MANIFEST_VERIFIED_DATASET_IMAGE",
        manifest_image_id=397133,
    )
    assert res.is_verified is False
    assert "dimension" in res.failure_reason.lower()


def test_unknown_license_represented_accurately():
    """Manifest records with unknown or non-standard license must report UNKNOWN, never invented permissions."""
    registry = CanonicalDatasetManifestRegistry.get_instance()
    for rec in registry._records_by_id.values():
        if not rec.get("license_name") or not rec.get("license_url"):
            assert rec["license_status"] == "UNKNOWN"
            assert rec["redistribution_status"] == "UNKNOWN"


def test_user_upload_without_manifest_accepted_as_unverified(client: TestClient):
    """A legitimate user-uploaded photograph without manifest is accepted but labeled unverified."""
    user_img_b64 = _create_synthetic_jpeg_b64(width=300, height=300, color=(120, 180, 240))
    payload = {
        "image_base64": user_img_b64,
        "annotation": {
            "id": 1,
            "category_name": "cup",
            "bbox": {"x_min": 50, "y_min": 50, "x_max": 200, "y_max": 200},
        },
        "image_source_type": "USER_UPLOADED_IMAGE",
        "manifest_image_id": None,
    }
    resp = client.post("/api/vision/audit", json=payload)
    assert resp.status_code == 200, resp.text
    prov = resp.json()["provenance"]
    assert prov["image_source_type"] == "USER_UPLOADED_IMAGE"
    assert prov["provenance_category"] == "DECODED_IMAGE"
    assert prov["is_independently_verified"] is False
    assert prov["manifest_image_id"] is None


def test_synthetic_fixture_cannot_claim_real_dataset_provenance():
    """Synthetic fixtures attempting to pass as REAL_PHOTOGRAPH or VERIFIED_DATASET fail provenance gate."""
    with pytest.raises(ProvenanceViolationError, match="do not establish authentic photographic provenance"):
        ProvenanceGate.audit_claim(
            ProvenanceCategory.REAL_PHOTOGRAPH,
            claimed_effect_or_engine="LabelCourt",
            has_real_image_bytes=True,
            has_authentic_source_record=False,
        )

    with pytest.raises(ProvenanceViolationError, match="validated source-manifest"):
        ProvenanceGate.audit_claim(
            ProvenanceCategory.VERIFIED_DATASET_IMAGE,
            claimed_effect_or_engine="LabelCourt",
            has_real_image_bytes=True,
            has_authentic_source_record=True,
            dataset_manifest_verified=False,
        )


def test_valid_decoded_image_not_automatically_verified_photograph():
    """A valid, successfully decoded RGB image must not be granted REAL_PHOTOGRAPH or VERIFIED status."""
    res = verify_authoritative_provenance(
        image_bytes=b"\x89PNG\r\n\x1a\nfake",
        image_sha256="a" * 64,
        width=500,
        height=500,
        claimed_source_type=None,
        manifest_image_id=None,
    )
    assert res.is_verified is False
    assert res.category == "DECODED_IMAGE"
    assert res.image_source_type == "USER_UPLOADED_IMAGE"
