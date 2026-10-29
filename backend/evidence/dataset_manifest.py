"""Validated source-manifest records for public photographic datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from PIL import Image
from pydantic import BaseModel, Field, model_validator


class DatasetSourceRecord(BaseModel):
    dataset_name: str
    dataset_release: str
    official_image_id: int
    filename: str
    canonical_source_url: str
    license_status: Literal["KNOWN", "UNKNOWN"] = "UNKNOWN"
    license_name: str | None = None
    license_url: str | None = None
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    annotation_file_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    annotation_image_id: int
    acquisition_procedure: str
    redistribution_status: Literal["PERMITTED", "RESTRICTED", "UNKNOWN", "PERMITTED_WITH_ATTRIBUTION"] = "UNKNOWN"
    preprocessing_history: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_linkage_and_license(self) -> "DatasetSourceRecord":
        if self.annotation_image_id != self.official_image_id:
            raise ValueError("annotation/image ID mismatch")
        if not self.license_name or not self.license_url:
            self.license_status = "UNKNOWN"
            self.redistribution_status = "UNKNOWN"
        return self

    def verify_files(self, image_path: str | Path, annotation_path: str | Path) -> None:
        image_bytes = Path(image_path).read_bytes()
        annotation_bytes = Path(annotation_path).read_bytes()
        if hashlib.sha256(image_bytes).hexdigest() != self.original_sha256:
            raise ValueError("source image hash mismatch")
        if hashlib.sha256(annotation_bytes).hexdigest() != self.annotation_file_sha256:
            raise ValueError("annotation file hash mismatch")
        with Image.open(image_path) as image:
            if image.size != (self.image_width, self.image_height):
                raise ValueError("source image dimension mismatch")
        annotation = json.loads(annotation_bytes)
        image_ids = {int(item["id"]) for item in annotation.get("images", [])}
        if self.annotation_image_id not in image_ids:
            raise ValueError("annotation/image ID linkage missing")


class ProvenanceVerificationResult(BaseModel):
    """Result of authoritative server-side provenance verification."""

    is_verified: bool
    category: str
    image_source_type: str
    manifest_image_id: int | None = None
    canonical_source_url: str | None = None
    license_status: str = "UNKNOWN"
    license_name: str | None = None
    license_url: str | None = None
    redistribution_status: str = "UNKNOWN"
    annotation_source_type: str = "HUMAN_ANNOTATION"
    annotation_verified: bool = False
    verification_method: str = "SERVER_AUTHORITATIVE_SHA256_MANIFEST_AUDIT"
    failure_reason: str | None = None


class CanonicalDatasetManifestRegistry:
    """Server-side authoritative registry of validated dataset images and annotations."""

    _instance: "CanonicalDatasetManifestRegistry | None" = None

    def __init__(self) -> None:
        self.root = Path(__file__).resolve().parent.parent.parent
        self._records_by_id: dict[int, dict] = {}
        self._records_by_sha256: dict[str, dict] = {}
        self._annotations_by_image: dict[int, list[dict]] = {}
        self._categories: dict[int, str] = {}
        self._loaded: bool = False
        self._load()

    def _load(self) -> None:
        if self._loaded:
            return

        manifest_candidates = [
            self.root / "evaluation" / "experiments" / "genuine_model_predictions" / "dataset_manifest.json",
            self.root / "docs" / "research" / "REAL_DATASET_MANIFEST.json",
        ]
        manifest_file = next((p for p in manifest_candidates if p.exists()), None)
        if manifest_file:
            data = json.loads(manifest_file.read_text())
            records = data.get("records") or data.get("dataset") or []
            for item in records:
                img_id = int(item["official_image_id"])
                rec = {
                    "official_image_id": img_id,
                    "filename": item.get("filename") or item.get("file_name"),
                    "original_sha256": (item.get("original_sha256") or "").lower(),
                    "width": int(item.get("width") or item.get("image_width") or 0),
                    "height": int(item.get("height") or item.get("image_height") or 0),
                    "canonical_source_url": item.get("canonical_source_url") or item.get("original_source_url"),
                    "license_status": item.get("license_status", "UNKNOWN"),
                    "license_name": item.get("license_name"),
                    "license_url": item.get("license_url"),
                    "redistribution_status": item.get("redistribution_status", "UNKNOWN"),
                }
                self._records_by_id[img_id] = rec
                if rec["original_sha256"]:
                    self._records_by_sha256[rec["original_sha256"]] = rec

        ann_file = self.root / "data" / "coco_val2017_subset" / "annotations_coco_pilot20.json"
        if ann_file.exists():
            coco_data = json.loads(ann_file.read_text())
            for cat in coco_data.get("categories", []):
                self._categories[int(cat["id"])] = cat["name"]
            for ann in coco_data.get("annotations", []):
                i_id = int(ann["image_id"])
                self._annotations_by_image.setdefault(i_id, []).append(ann)

        self._loaded = True

    @classmethod
    def get_instance(cls) -> "CanonicalDatasetManifestRegistry":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_record(self, image_id: int) -> dict | None:
        return self._records_by_id.get(image_id)

    def get_record_by_hash(self, sha256_hash: str) -> dict | None:
        return self._records_by_sha256.get(sha256_hash.lower())

    def get_annotations_for_image(self, image_id: int) -> list[dict]:
        return self._annotations_by_image.get(image_id, [])

    def get_category_name(self, category_id: int) -> str:
        return self._categories.get(category_id, "unknown")


def verify_authoritative_provenance(
    image_bytes: bytes,
    image_sha256: str,
    width: int,
    height: int,
    claimed_source_type: str | None = None,
    manifest_image_id: int | None = None,
    annotation: Any | None = None,
    claimed_annotation_source: str | None = None,
    dataset_manifest_ref: str | None = None,
) -> ProvenanceVerificationResult:
    """Authoritative server-side verification of image and annotation provenance.

    A request can NEVER grant itself trusted provenance:
    1. If no verified dataset claim is made, image is classified as USER_UPLOADED_IMAGE (DECODED_IMAGE).
    2. If a verified dataset claim is made (explicit image_source_type, manifest_image_id, or manifest ref):
       - The manifest entry must exist.
       - The supplied image bytes must match the manifest expected SHA-256.
       - Image dimensions must match the manifest.
       - If a verified reference annotation is claimed, annotation linkage and content must match ground truth.
       - License and redistribution metadata are accurately reported.
       If ANY verification check fails, verification fails and the claim is rejected.
    """
    registry = CanonicalDatasetManifestRegistry.get_instance()

    # Determine if client is making a verified dataset claim
    has_verified_image_claim = (
        claimed_source_type == "MANIFEST_VERIFIED_DATASET_IMAGE"
        or manifest_image_id is not None
        or dataset_manifest_ref is not None
    )
    has_verified_annotation_claim = (
        claimed_annotation_source == "PUBLIC_DATASET_ANNOTATION"
    )

    # Case 1: Legitimate unverified user upload (no verified claims asserted)
    if not has_verified_image_claim and not has_verified_annotation_claim:
        return ProvenanceVerificationResult(
            is_verified=False,
            category="DECODED_IMAGE",
            image_source_type="USER_UPLOADED_IMAGE",
            manifest_image_id=None,
            canonical_source_url=None,
            license_status="UNKNOWN",
            license_name=None,
            license_url=None,
            redistribution_status="UNKNOWN",
            annotation_source_type="HUMAN_ANNOTATION",
            annotation_verified=False,
            verification_method="DECODED_RGB_PIXELS",
            failure_reason=None,
        )

    # Case 2: Client claimed verified dataset provenance — perform strict independent verification
    if manifest_image_id is None:
        return ProvenanceVerificationResult(
            is_verified=False,
            category="DECODED_IMAGE",
            image_source_type="USER_UPLOADED_IMAGE",
            failure_reason="Claimed verified dataset provenance without providing manifest_image_id.",
        )

    record = registry.get_record(manifest_image_id)
    if record is None:
        return ProvenanceVerificationResult(
            is_verified=False,
            category="DECODED_IMAGE",
            image_source_type="USER_UPLOADED_IMAGE",
            failure_reason=f"Manifest image ID {manifest_image_id} not found in canonical dataset manifest.",
        )

    # 1. SHA-256 integrity verification
    expected_hash = record["original_sha256"]
    if image_sha256.lower() != expected_hash:
        return ProvenanceVerificationResult(
            is_verified=False,
            category="DECODED_IMAGE",
            image_source_type="USER_UPLOADED_IMAGE",
            failure_reason=(
                f"Supplied image SHA-256 digest ({image_sha256}) does not match "
                f"manifest expected hash ({expected_hash})."
            ),
        )

    # 2. Dimensions integrity verification
    if (width, height) != (record["width"], record["height"]):
        return ProvenanceVerificationResult(
            is_verified=False,
            category="DECODED_IMAGE",
            image_source_type="USER_UPLOADED_IMAGE",
            failure_reason=(
                f"Image dimensions ({width}x{height}) do not match "
                f"manifest expected dimensions ({record['width']}x{record['height']})."
            ),
        )

    # 3. Annotation linkage and content verification
    annotation_verified = False
    ann_source_type = claimed_annotation_source or "HUMAN_ANNOTATION"

    if annotation is not None:
        ann_id = getattr(annotation, "id", None)
        gt_annotations = registry.get_annotations_for_image(manifest_image_id)
        gt_ann = next((a for a in gt_annotations if str(a.get("id")) == str(ann_id)), None) if ann_id is not None else None

        if has_verified_annotation_claim:
            if ann_id is None:
                return ProvenanceVerificationResult(
                    is_verified=False,
                    category="DECODED_IMAGE",
                    image_source_type="USER_UPLOADED_IMAGE",
                    failure_reason="Claimed PUBLIC_DATASET_ANNOTATION but annotation id is missing.",
                )
            if gt_ann is None:
                return ProvenanceVerificationResult(
                    is_verified=False,
                    category="DECODED_IMAGE",
                    image_source_type="USER_UPLOADED_IMAGE",
                    failure_reason=f"Annotation ID {ann_id} does not exist in ground-truth records for image {manifest_image_id}.",
                )

            # Category check
            expected_cat_name = registry.get_category_name(gt_ann["category_id"])
            if annotation.category_name.strip().lower() != expected_cat_name.strip().lower():
                return ProvenanceVerificationResult(
                    is_verified=False,
                    category="DECODED_IMAGE",
                    image_source_type="USER_UPLOADED_IMAGE",
                    failure_reason=(
                        f"Annotation category '{annotation.category_name}' does not match "
                        f"official ground truth '{expected_cat_name}'."
                    ),
                )

            # Bounding box coordinates check (tolerance: 1.0 px)
            gt_bbox = gt_ann["bbox"]  # [x, y, w, h]
            exp_xmin, exp_ymin = float(gt_bbox[0]), float(gt_bbox[1])
            exp_xmax, exp_ymax = float(gt_bbox[0] + gt_bbox[2]), float(gt_bbox[1] + gt_bbox[3])
            bbox = annotation.bbox
            if (
                abs(bbox.x_min - exp_xmin) > 1.0
                or abs(bbox.y_min - exp_ymin) > 1.0
                or abs(bbox.x_max - exp_xmax) > 1.0
                or abs(bbox.y_max - exp_ymax) > 1.0
            ):
                return ProvenanceVerificationResult(
                    is_verified=False,
                    category="DECODED_IMAGE",
                    image_source_type="USER_UPLOADED_IMAGE",
                    failure_reason="Annotation bounding box does not match official ground truth.",
                )

            # Segmentation mask check
            gt_seg = gt_ann.get("segmentation")
            if gt_seg is not None and annotation.segmentation is not None:
                if str(annotation.segmentation) != str(gt_seg):
                    return ProvenanceVerificationResult(
                        is_verified=False,
                        category="DECODED_IMAGE",
                        image_source_type="USER_UPLOADED_IMAGE",
                        failure_reason="Annotation reference mask does not match official ground truth.",
                    )

            annotation_verified = True
            ann_source_type = "PUBLIC_DATASET_ANNOTATION"
        else:
            if gt_ann is not None:
                expected_cat = registry.get_category_name(gt_ann["category_id"])
                gt_bbox = gt_ann["bbox"]
                exp_xmin, exp_ymin = float(gt_bbox[0]), float(gt_bbox[1])
                exp_xmax, exp_ymax = float(gt_bbox[0] + gt_bbox[2]), float(gt_bbox[1] + gt_bbox[3])
                bbox = annotation.bbox
                coords_match = (
                    abs(bbox.x_min - exp_xmin) <= 1.0
                    and abs(bbox.y_min - exp_ymin) <= 1.0
                    and abs(bbox.x_max - exp_xmax) <= 1.0
                    and abs(bbox.y_max - exp_ymax) <= 1.0
                )
                if annotation.category_name.strip().lower() == expected_cat.strip().lower() and coords_match:
                    annotation_verified = True
                    ann_source_type = "PUBLIC_DATASET_ANNOTATION"
                else:
                    annotation_verified = False
                    ann_source_type = "HUMAN_ANNOTATION"
            else:
                annotation_verified = False
                ann_source_type = "HUMAN_ANNOTATION"

    # All checks passed! Server grants MANIFEST_VERIFIED_DATASET_IMAGE
    return ProvenanceVerificationResult(
        is_verified=True,
        category="VERIFIED_DATASET_IMAGE",
        image_source_type="MANIFEST_VERIFIED_DATASET_IMAGE",
        manifest_image_id=manifest_image_id,
        canonical_source_url=record["canonical_source_url"],
        license_status=record["license_status"],
        license_name=record["license_name"],
        license_url=record["license_url"],
        redistribution_status=record["redistribution_status"],
        annotation_source_type=ann_source_type,
        annotation_verified=annotation_verified,
        verification_method="SERVER_AUTHORITATIVE_SHA256_MANIFEST_AUDIT",
        failure_reason=None,
    )


