"""Unit tests for the explicit, versioned category taxonomy mapping."""

from __future__ import annotations

import pytest
from backend.evidence.taxonomy import (
    TAXONOMY_VERSION,
    COCO_CANONICAL_CATEGORIES,
    MODEL_TO_CANONICAL_ALIASES,
    are_classes_equivalent,
    get_canonical_category,
    normalize_label_string,
)


def test_taxonomy_version():
    assert TAXONOMY_VERSION == "v1.0.0"


def test_coco_canonical_category_count():
    assert len(COCO_CANONICAL_CATEGORIES) == 80
    assert "person" in COCO_CANONICAL_CATEGORIES
    assert "couch" in COCO_CANONICAL_CATEGORIES
    assert "potted plant" in COCO_CANONICAL_CATEGORIES
    assert "tv" in COCO_CANONICAL_CATEGORIES
    assert "motorcycle" in COCO_CANONICAL_CATEGORIES
    assert "airplane" in COCO_CANONICAL_CATEGORIES
    assert "dining table" in COCO_CANONICAL_CATEGORIES


def test_recognized_aliases_map_to_canonical():
    assert get_canonical_category("sofa") == "couch"
    assert get_canonical_category("pottedplant") == "potted plant"
    assert get_canonical_category("tvmonitor") == "tv"
    assert get_canonical_category("motorbike") == "motorcycle"
    assert get_canonical_category("aeroplane") == "airplane"
    assert get_canonical_category("diningtable") == "dining table"


def test_canonical_categories_remain_unchanged():
    assert get_canonical_category("couch") == "couch"
    assert get_canonical_category("person") == "person"
    assert get_canonical_category("bottle") == "bottle"


def test_are_classes_equivalent_direct_match():
    assert are_classes_equivalent("person", "person")
    assert are_classes_equivalent("Person", "person ")
    assert are_classes_equivalent("CAR", "car")


def test_are_classes_equivalent_alias_pairs():
    # Model alias vs COCO annotation
    assert are_classes_equivalent("sofa", "couch")
    assert are_classes_equivalent("couch", "sofa")
    assert are_classes_equivalent("pottedplant", "potted plant")
    assert are_classes_equivalent("potted plant", "pottedplant")
    assert are_classes_equivalent("tvmonitor", "tv")
    assert are_classes_equivalent("tv", "tvmonitor")
    assert are_classes_equivalent("motorbike", "motorcycle")
    assert are_classes_equivalent("motorcycle", "motorbike")
    assert are_classes_equivalent("aeroplane", "airplane")
    assert are_classes_equivalent("airplane", "aeroplane")
    assert are_classes_equivalent("diningtable", "dining table")
    assert are_classes_equivalent("dining table", "diningtable")


def test_are_classes_equivalent_distinct_classes_rejected():
    assert not are_classes_equivalent("cat", "dog")
    assert not are_classes_equivalent("person", "bicycle")
    assert not are_classes_equivalent("car", "truck")
    assert not are_classes_equivalent("cup", "bottle")
    assert not are_classes_equivalent("chair", "couch")
    assert not are_classes_equivalent("unknown_class_a", "unknown_class_b")
