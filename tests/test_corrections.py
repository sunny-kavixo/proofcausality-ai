import json
from backend.evidence.schemas import AnnotationItem, BoundingBox
from backend.labelcourt.corrections import export_coco_corrections


def test_export_coco_corrections():
    items = [
        AnnotationItem(
            id=10,
            image_id=20,
            category_id=3,
            category_name="mug",
            bbox=BoundingBox(x_min=10, y_min=10, x_max=50, y_max=60),
            area=2000.0,
        )
    ]
    meta = [{"id": 20, "file_name": "frame_001.jpg", "width": 640, "height": 480}]
    cats = [{"id": 3, "name": "mug"}]

    exported_json = export_coco_corrections(items, meta, cats)
    parsed = json.loads(exported_json)

    assert len(parsed["images"]) == 1
    assert len(parsed["annotations"]) == 1
    assert parsed["annotations"][0]["id"] == 10
    assert parsed["annotations"][0]["bbox"] == [10.0, 10.0, 40.0, 50.0]
