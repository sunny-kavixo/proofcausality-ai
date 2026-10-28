import numpy as np
from backend.evidence.schemas import AnnotationItem, AuditVerdict, BoundingBox
from backend.labelcourt.auditor import LabelCourtAuditor


def test_labelcourt_accept_clean_annotation():
    auditor = LabelCourtAuditor()
    bbox = BoundingBox(x_min=20, y_min=20, x_max=80, y_max=80)
    item = AnnotationItem(
        id=1,
        image_id=101,
        category_id=1,
        category_name="coffee_cup",
        bbox=bbox,
    )
    # Candidate mask matching the bounding box
    candidate_mask = np.zeros((100, 100), dtype=bool)
    candidate_mask[20:80, 20:80] = True

    # Sharp image canvas
    img = np.zeros((100, 100), dtype=np.uint8)
    img[::2, ::2] = 255

    res, receipt = auditor.audit_annotation(item, candidate_mask, image_gray=img)
    assert res.verdict == AuditVerdict.ACCEPT
    assert res.iou == 1.0
    assert res.class_match is True
    assert receipt.verdict == "ACCEPT"


def test_labelcourt_review_on_discrepancy():
    auditor = LabelCourtAuditor()
    bbox = BoundingBox(x_min=20, y_min=20, x_max=80, y_max=80)
    item = AnnotationItem(
        id=2,
        image_id=102,
        category_id=1,
        category_name="coffee_cup",
        bbox=bbox,
    )
    # Shifted candidate mask
    candidate_mask = np.zeros((100, 100), dtype=bool)
    candidate_mask[30:90, 30:90] = True

    img = np.zeros((100, 100), dtype=np.uint8)
    img[::2, ::2] = 255

    res, _ = auditor.audit_annotation(item, candidate_mask, image_gray=img)
    assert res.verdict == AuditVerdict.REVIEW
    assert res.iou < 0.85


def test_labelcourt_uncertain_on_blur():
    auditor = LabelCourtAuditor()
    bbox = BoundingBox(x_min=20, y_min=20, x_max=80, y_max=80)
    item = AnnotationItem(
        id=3,
        image_id=103,
        category_id=1,
        category_name="coffee_cup",
        bbox=bbox,
    )
    candidate_mask = np.zeros((100, 100), dtype=bool)
    candidate_mask[20:80, 20:80] = True

    # Blurry image
    blurry_img = np.full((100, 100), 128, dtype=np.uint8)

    res, _ = auditor.audit_annotation(item, candidate_mask, image_gray=blurry_img)
    assert res.verdict == AuditVerdict.UNCERTAIN
    assert res.recapture_plan is not None
    assert "Hold the camera steady" in res.recapture_plan.directive
