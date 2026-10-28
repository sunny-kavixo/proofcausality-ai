from backend.evidence.schemas import AnnotationItem, BoundingBox
from evaluation.inject_errors import (
    inject_boundary_erosion,
    shift_annotation_bbox,
    swap_annotation_class,
)


def test_shift_annotation_bbox():
    item = AnnotationItem(
        id=1,
        image_id=1,
        category_id=1,
        category_name="box",
        bbox=BoundingBox(x_min=10, y_min=10, x_max=40, y_max=40),
        segmentation=[[10, 10, 40, 10, 40, 40, 10, 40]],
    )
    shifted = shift_annotation_bbox(item, dx_px=15, dy_px=-5)
    assert shifted.bbox.x_min == 25
    assert shifted.bbox.y_min == 5
    assert shifted.segmentation[0][0] == 25
    assert shifted.segmentation[0][1] == 5


def test_swap_annotation_class():
    item = AnnotationItem(
        id=1,
        image_id=1,
        category_id=1,
        category_name="cat",
        bbox=BoundingBox(x_min=0, y_min=0, x_max=10, y_max=10),
    )
    swapped = swap_annotation_class(item, "dog", wrong_id=2)
    assert swapped.category_name == "dog"
    assert swapped.category_id == 2


def test_inject_boundary_erosion():
    item = AnnotationItem(
        id=1,
        image_id=1,
        category_id=1,
        category_name="box",
        bbox=BoundingBox(x_min=0, y_min=0, x_max=100, y_max=100),
    )
    eroded = inject_boundary_erosion(item, shrink_ratio=0.5)
    assert eroded.bbox.width == 50.0
    assert eroded.bbox.height == 50.0
