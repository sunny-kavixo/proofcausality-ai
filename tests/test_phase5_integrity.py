import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from backend.evidence.dataset_manifest import DatasetSourceRecord
from backend.evidence.provenance import ProvenanceCategory, ProvenanceGate, ProvenanceViolationError
from backend.providers.local_segmentation import (
    LocalMask2FormerProvider,
    ModelOutputError,
    decode_binary_mask_rle,
    encode_binary_mask_rle,
)
from evaluation.grouped_statistics import assert_disjoint_groups, grouped_bootstrap_interval


def test_random_decodable_png_is_not_verified_photographic_provenance():
    with pytest.raises(ProvenanceViolationError, match="do not establish authentic photographic provenance"):
        ProvenanceGate.audit_claim(
            ProvenanceCategory.REAL_PHOTOGRAPH,
            "decoded-png",
            has_real_image_bytes=True,
            has_authentic_source_record=False,
        )


def test_verified_dataset_claim_requires_manifest():
    with pytest.raises(ProvenanceViolationError, match="validated source-manifest"):
        ProvenanceGate.audit_claim(
            ProvenanceCategory.VERIFIED_DATASET_IMAGE,
            "dataset",
            has_real_image_bytes=True,
            has_authentic_source_record=True,
            dataset_manifest_verified=False,
        )


def test_model_prediction_requires_actual_execution_evidence():
    with pytest.raises(ProvenanceViolationError, match="actual inference evidence"):
        ProvenanceGate.audit_claim(ProvenanceCategory.MODEL_PREDICTION, "model", model_execution_evidence={})


def test_manifest_hash_and_linkage_validation(tmp_path):
    image_path = tmp_path / "1.png"
    Image.new("RGB", (3, 2), "red").save(image_path)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(json.dumps({"images": [{"id": 1}]}))
    record = DatasetSourceRecord(
        dataset_name="test",
        dataset_release="1",
        official_image_id=1,
        filename="1.png",
        canonical_source_url="https://example.invalid/1.png",
        license_status="UNKNOWN",
        original_sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(),
        image_width=3,
        image_height=2,
        annotation_file_sha256=hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
        annotation_image_id=1,
        acquisition_procedure="fixture",
    )
    record.verify_files(image_path, annotation_path)
    image_path.write_bytes(image_path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="image hash mismatch"):
        record.verify_files(image_path, annotation_path)


def test_manifest_rejects_annotation_image_id_mismatch():
    with pytest.raises(ValueError, match="annotation/image ID mismatch"):
        DatasetSourceRecord(
            dataset_name="test", dataset_release="1", official_image_id=1, filename="1.png",
            canonical_source_url="https://example.invalid/1.png", original_sha256="0" * 64,
            image_width=3, image_height=2, annotation_file_sha256="1" * 64,
            annotation_image_id=2, acquisition_procedure="fixture",
        )


def test_missing_license_is_unknown_not_permitted():
    record = DatasetSourceRecord(
        dataset_name="test", dataset_release="1", official_image_id=1, filename="1.png",
        canonical_source_url="https://example.invalid/1.png", license_status="KNOWN",
        original_sha256="0" * 64, image_width=3, image_height=2,
        annotation_file_sha256="1" * 64, annotation_image_id=1,
        acquisition_procedure="fixture", redistribution_status="PERMITTED",
    )
    assert record.license_status == "UNKNOWN"
    assert record.redistribution_status == "UNKNOWN"


def test_rle_roundtrip_and_dimension_mismatch():
    mask = np.array([[False, True], [True, True]], dtype=bool)
    assert np.array_equal(decode_binary_mask_rle(encode_binary_mask_rle(mask), 2, 2), mask)
    with pytest.raises(ModelOutputError, match="expected 6"):
        decode_binary_mask_rle(encode_binary_mask_rle(mask), 2, 3)


def test_prediction_interface_structurally_excludes_ground_truth():
    assert list(__import__("inspect").signature(LocalMask2FormerProvider.predict).parameters) == [
        "self", "image_path", "threshold"
    ]


def test_grouped_bootstrap_reproducible_and_preserves_source_groups():
    rows = [
        {"image_id": 1, "value": 1.0}, {"image_id": 1, "value": 1.0},
        {"image_id": 2, "value": 0.0}, {"image_id": 2, "value": 0.0},
    ]
    stat = lambda sample: float(np.mean([row["value"] for row in sample]))
    first = grouped_bootstrap_interval(rows, stat, iterations=200, seed=7)
    second = grouped_bootstrap_interval(rows, stat, iterations=200, seed=7)
    assert first == second
    assert first[0] == 0.0 and first[1] == 1.0


def test_train_test_overlap_rejected():
    with pytest.raises(ValueError, match="overlap"):
        assert_disjoint_groups([1, 2], [2, 3])
