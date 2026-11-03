"""Safety tests for the explicit COCO pilot acquisition command."""

from __future__ import annotations

import json

import pytest

from scripts.download_coco_subset import PILOT_IMAGE_IDS, acquire, load_subset, redistribution_status


def _annotations_payload() -> dict:
    return {
        "licenses": [{"id": 4, "name": "Attribution License", "url": "https://creativecommons.org/licenses/by/2.0/"}],
        "categories": [{"id": 1, "name": "person"}],
        "images": [
            {"id": image_id, "file_name": f"{image_id:012d}.jpg", "width": 1, "height": 1, "license": 4}
            for image_id in PILOT_IMAGE_IDS
        ],
        "annotations": [],
    }


def test_annotation_input_must_contain_all_predeclared_ids(tmp_path):
    path = tmp_path / "instances_val2017.json"
    payload = _annotations_payload()
    payload["images"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lacks pilot IDs"):
        load_subset(path)


def test_license_status_never_invents_permission():
    assert redistribution_status(4) == "PERMITTED_WITH_ATTRIBUTION"
    assert redistribution_status(3).startswith("RESTRICTED")
    assert redistribution_status(None) == "UNKNOWN_DO_NOT_REDISTRIBUTE"
    assert redistribution_status(999) == "UNKNOWN_DO_NOT_REDISTRIBUTE"


def test_acquisition_refuses_to_replace_generated_files_before_network(tmp_path):
    annotations = tmp_path / "instances_val2017.json"
    annotations.write_text(json.dumps(_annotations_payload()), encoding="utf-8")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "annotations_coco_pilot20.json").write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to replace"):
        acquire(annotations, data_dir, replace=False, base_url="https://invalid.example")
    assert (data_dir / "annotations_coco_pilot20.json").read_text(encoding="utf-8") == "preserve me"
