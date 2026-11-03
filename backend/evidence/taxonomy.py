"""Explicit, versioned category taxonomy mapping between upstream vision models and COCO annotations.

Scientific Integrity Framework:
================================
1. PROVENANCE OF VOCABULARY DIVERGENCE:
   The upstream Mask2Former checkpoint (`facebook/mask2former-swin-tiny-coco-instance`)
   was trained on COCO instance segmentation, but its Hugging Face model configuration
   (`id2label`) employs Pascal VOC / Panoptic nomenclature for a small subset of classes:
     - Model: `sofa`        <--> COCO: `couch`
     - Model: `pottedplant` <--> COCO: `potted plant`
     - Model: `tvmonitor`   <--> COCO: `tv`
     - Model: `motorbike`   <--> COCO: `motorcycle`
     - Model: `aeroplane`   <--> COCO: `airplane`
     - Model: `diningtable` <--> COCO: `dining table`

2. STRICT PRESERVATION INVARIANTS:
   - Ground truth annotation labels are NEVER silently rewritten in datasets.
   - Model predictions preserve their original `label` string from `id2label`.
   - Taxonomy equivalence is explicit, versioned, transparent, and audited.
   - Equivalence checking determines whether two labels represent the same
     semantic concept, distinguishing true misclassifications from nomenclature aliases.
"""

from __future__ import annotations

from typing import Final

TAXONOMY_VERSION: Final[str] = "v1.0.0"

# Canonical COCO standard category names (as defined by the COCO Consortium)
COCO_CANONICAL_CATEGORIES: Final[frozenset[str]] = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
})

# Upstream model aliases mapped to canonical COCO names
MODEL_TO_CANONICAL_ALIASES: Final[dict[str, str]] = {
    "sofa": "couch",
    "pottedplant": "potted plant",
    "tvmonitor": "tv",
    "motorbike": "motorcycle",
    "aeroplane": "airplane",
    "diningtable": "dining table",
}

# Reverse mapping from canonical COCO names to recognized model aliases
CANONICAL_TO_MODEL_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "couch": ("sofa",),
    "potted plant": ("pottedplant",),
    "tv": ("tvmonitor",),
    "motorcycle": ("motorbike",),
    "airplane": ("aeroplane",),
    "dining table": ("diningtable",),
}


def normalize_label_string(label: str) -> str:
    """Normalize label string for robust comparison by lowercasing and trimming whitespace."""
    if not isinstance(label, str):
        return ""
    return label.strip().lower()


def get_canonical_category(label: str) -> str:
    """Map any recognized alias to its canonical COCO category name."""
    norm = normalize_label_string(label)
    if norm in MODEL_TO_CANONICAL_ALIASES:
        return MODEL_TO_CANONICAL_ALIASES[norm]
    return norm


def are_classes_equivalent(
    label_a: str,
    label_b: str,
    taxonomy_version: str = TAXONOMY_VERSION,
) -> bool:
    """Determine whether two category labels represent equivalent semantic classes.

    Args:
        label_a: First class label (e.g. from prediction).
        label_b: Second class label (e.g. from human annotation).
        taxonomy_version: Specific taxonomy version to use.

    Returns:
        True if exact string match (case-insensitive) or mapped canonical equivalent;
        False otherwise.
    """
    norm_a = normalize_label_string(label_a)
    norm_b = normalize_label_string(label_b)

    if norm_a == norm_b:
        return True

    canon_a = get_canonical_category(norm_a)
    canon_b = get_canonical_category(norm_b)

    return canon_a == canon_b and canon_a in COCO_CANONICAL_CATEGORIES
