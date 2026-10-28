"""Human-in-the-loop annotation correction export."""

from __future__ import annotations

import json
from typing import Any, List
from backend.evidence.schemas import AnnotationItem, BoundingBox


def export_coco_corrections(
    approved_annotations: List[AnnotationItem],
    image_metadata: List[dict[str, Any]],
    categories: List[dict[str, Any]],
) -> str:
    """Export reviewer-approved annotations into canonical COCO JSON format."""
    coco_dict = {
        "images": image_metadata,
        "annotations": [
            {
                "id": a.id,
                "image_id": a.image_id,
                "category_id": a.category_id,
                "bbox": [a.bbox.x_min, a.bbox.y_min, a.bbox.width, a.bbox.height],
                "area": a.area or a.bbox.area,
                "segmentation": a.segmentation or [],
                "iscrowd": 0,
            }
            for a in approved_annotations
        ],
        "categories": categories,
    }
    return json.dumps(coco_dict, indent=2)
