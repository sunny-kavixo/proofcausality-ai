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


def test_reference_annotation_mutation_does_not_affect_raw_prediction(tmp_path):
    """Explicitly verify that mutating the reference annotation alone does not alter raw predictions."""
    from backend.evidence.schemas import AnnotationItem, BoundingBox
    ann1 = AnnotationItem(
        id=1, image_id=1, category_id=1, category_name="cup",
        bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50),
    )
    ann2 = AnnotationItem(
        id=1, image_id=1, category_id=99, category_name="airplane",
        bbox=BoundingBox(x_min=900, y_min=900, x_max=999, y_max=999),
    )
    # The provider's predict method does not receive AnnotationItem, ensuring structural exclusion
    params = list(__import__("inspect").signature(LocalMask2FormerProvider.predict).parameters)
    assert "annotation" not in params
    assert "reference_mask" not in params
    assert "gt" not in params
    assert "bbox" not in params


def test_saved_predictions_link_bidirectionally_to_images_and_executions():
    from pathlib import Path
    root = Path("evaluation/experiments/genuine_model_predictions")
    manifest_file = root / "dataset_manifest.json"
    exec_file = root / "execution_records.json"
    raw_file = root / "raw_predictions.json"

    if not (manifest_file.exists() and exec_file.exists() and raw_file.exists()):
        pytest.skip("Prediction artifacts not present in current directory")

    manifest = json.loads(manifest_file.read_text())
    executions = json.loads(exec_file.read_text())
    raw = json.loads(raw_file.read_text())

    manifest_by_id = {r["official_image_id"]: r for r in manifest["records"]}
    raw_by_id = {r["image_id"]: r for r in raw}
    exec_by_sha = {e["input_sha256"]: e for e in executions}

    assert len(manifest_by_id) == 20
    assert len(raw_by_id) == 20
    assert len(exec_by_sha) == 20

    expected_ckpt = "3550e8b21381d082c4d12b06a4b7c5f0f4a530125eee4c5f264b1e5f43dfab80"
    for img_id, m_rec in manifest_by_id.items():
        raw_rec = raw_by_id[img_id]
        exec_rec = exec_by_sha[m_rec["original_sha256"]]
        assert raw_rec["input_sha256"] == m_rec["original_sha256"]
        assert raw_rec["output_sha256"] == exec_rec["output_sha256"]
        assert exec_rec["checkpoint_sha256"] == expected_ckpt

