import numpy as np
from backend.actionproof.verifier import ActionProofVerifier
from backend.evidence.schemas import ActionVerdict, ActionVerificationRequest, BoundingBox


def test_actionproof_moved_a_to_b_success():
    verifier = ActionProofVerifier()
    reg_a = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50)
    reg_b = BoundingBox(x_min=150, y_min=150, x_max=200, y_max=200)

    req = ActionVerificationRequest(
        task_type="moved_A_to_B",
        object_name="bottle",
        region_a=reg_a,
        region_b=reg_b,
        distance_threshold_px=80.0,
    )

    before_mask = np.zeros((220, 220), dtype=bool)
    before_mask[15:45, 15:45] = True  # in region A

    after_mask = np.zeros((220, 220), dtype=bool)
    after_mask[155:185, 155:185] = True  # in region B

    sharp = np.zeros((220, 220), dtype=np.uint8)
    sharp[::2, ::2] = 255

    res, receipt = verifier.verify_action(req, before_mask, after_mask, after_image_gray=sharp)
    assert res.verdict == ActionVerdict.SUCCESS
    assert res.displacement_px > 80.0
    assert receipt.verdict == "SUCCESS"


def test_actionproof_failed_when_not_moved():
    verifier = ActionProofVerifier()
    reg_a = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50)
    reg_b = BoundingBox(x_min=150, y_min=150, x_max=200, y_max=200)

    req = ActionVerificationRequest(
        task_type="moved_A_to_B",
        object_name="bottle",
        region_a=reg_a,
        region_b=reg_b,
        distance_threshold_px=80.0,
    )

    mask = np.zeros((220, 220), dtype=bool)
    mask[15:45, 15:45] = True

    sharp = np.zeros((220, 220), dtype=np.uint8)
    sharp[::2, ::2] = 255

    res, _ = verifier.verify_action(req, mask, mask, after_image_gray=sharp)
    assert res.verdict == ActionVerdict.FAILED
    assert res.displacement_px == 0.0


def test_actionproof_uncertain_on_occlusion():
    verifier = ActionProofVerifier()
    reg_a = BoundingBox(x_min=10, y_min=10, x_max=50, y_max=50)
    reg_b = BoundingBox(x_min=150, y_min=150, x_max=200, y_max=200)

    req = ActionVerificationRequest(
        task_type="moved_A_to_B",
        object_name="bottle",
        region_a=reg_a,
        region_b=reg_b,
        distance_threshold_px=80.0,
    )

    before_mask = np.zeros((220, 220), dtype=bool)
    before_mask[15:45, 15:45] = True  # 900 px

    after_mask = np.zeros((220, 220), dtype=bool)
    after_mask[155:165, 155:165] = True  # 100 px (89% drop)

    sharp = np.zeros((220, 220), dtype=np.uint8)
    sharp[::2, ::2] = 255

    res, _ = verifier.verify_action(req, before_mask, after_mask, after_image_gray=sharp)
    assert res.verdict == ActionVerdict.UNCERTAIN
    assert res.recapture_plan is not None
    assert "Keep the object visible" in res.recapture_plan.directive
