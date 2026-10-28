"""Standardized evaluation cases for physical action verification and re-capture resolution."""

from __future__ import annotations

from typing import Any, List, Optional
import numpy as np

from backend.evidence.masks import bbox_to_mask
from backend.evidence.schemas import ActionVerificationRequest, BoundingBox


class ActionTestCase:
    def __init__(
        self,
        case_id: str,
        name: str,
        task_type: str,
        ground_truth: str,  # "SUCCESS", "FAILED", "UNCERTAIN"
        defect_type: Optional[str],  # None, "BLUR", "OUT_OF_FRAME", "OCCLUSION", "MISSING_FINAL_STATE"
        before_bbox: BoundingBox,
        after_bbox: BoundingBox,
        region_a: Optional[BoundingBox] = None,
        region_b: Optional[BoundingBox] = None,
        target_bbox: Optional[BoundingBox] = None,
        is_blurry: bool = False,
        touches_border: bool = False,
        area_drop: bool = False,
        has_final_state: bool = True,
        recaptured_after_bbox: Optional[BoundingBox] = None,
        recaptured_ground_truth: Optional[str] = None,
        generic_recaptured_ground_truth: Optional[str] = None,
    ) -> None:
        self.case_id = case_id
        self.name = name
        self.task_type = task_type
        self.ground_truth = ground_truth
        self.defect_type = defect_type
        self.before_bbox = before_bbox
        self.after_bbox = after_bbox
        self.region_a = region_a
        self.region_b = region_b
        self.target_bbox = target_bbox
        self.is_blurry = is_blurry
        self.touches_border = touches_border
        self.area_drop = area_drop
        self.has_final_state = has_final_state
        self.recaptured_after_bbox = recaptured_after_bbox
        self.recaptured_ground_truth = recaptured_ground_truth
        self.generic_recaptured_ground_truth = generic_recaptured_ground_truth

    def get_masks(self, height: int = 400, width: int = 600) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate binary masks and synthetic grayscale image for testing."""
        before_mask = bbox_to_mask(self.before_bbox, height, width)
        after_mask = bbox_to_mask(self.after_bbox, height, width)

        if self.area_drop:
            # Drop 60% of after mask area to trigger occlusion
            rows = np.where(after_mask)[0]
            if len(rows) > 0:
                mid_row = int(np.mean(rows))
                after_mask[mid_row:, :] = False

        # Generate grayscale canvas
        image_gray = np.full((height, width), 128, dtype=np.uint8)
        if not self.is_blurry:
            # Add sharp edges to produce high Laplacian variance > 150
            image_gray[::4, ::4] = 240
            image_gray[1::4, 1::4] = 10
        else:
            # Smooth blur to keep variance < 50
            pass

        return before_mask, after_mask, image_gray


def get_standard_action_benchmark_cases() -> List[ActionTestCase]:
    """25 canonical physical action cases across clean successes, failures, and uncertain scenarios."""
    cases: List[ActionTestCase] = []

    # Regions for moved_A_to_B
    reg_a = BoundingBox(x_min=20, y_min=20, x_max=180, y_max=180)
    reg_b = BoundingBox(x_min=400, y_min=200, x_max=560, y_max=360)

    # 1-10: Clean SUCCESS cases
    for i in range(1, 11):
        offset = i * 2.0
        cases.append(
            ActionTestCase(
                case_id=f"CASE-S{i:02d}",
                name=f"Clean Movement Success #{i}",
                task_type="moved_A_to_B",
                ground_truth="SUCCESS",
                defect_type=None,
                before_bbox=BoundingBox(x_min=40 + offset, y_min=40 + offset, x_max=100 + offset, y_max=100 + offset),
                after_bbox=BoundingBox(x_min=430 + offset, y_min=230 + offset, x_max=490 + offset, y_max=290 + offset),
                region_a=reg_a,
                region_b=reg_b,
                is_blurry=False,
                touches_border=False,
                area_drop=False,
                has_final_state=True,
            )
        )

    # 11-16: Clean FAILED cases (Object did not leave A, or went to wrong destination)
    for i in range(1, 7):
        offset = i * 3.0
        dest_x = 70.0 if i <= 3 else 250.0  # Stayed in A or stopped midway
        cases.append(
            ActionTestCase(
                case_id=f"CASE-F{i:02d}",
                name=f"Incomplete Action Failure #{i}",
                task_type="moved_A_to_B",
                ground_truth="FAILED",
                defect_type=None,
                before_bbox=BoundingBox(x_min=40 + offset, y_min=40 + offset, x_max=100 + offset, y_max=100 + offset),
                after_bbox=BoundingBox(x_min=dest_x, y_min=50 + offset, x_max=dest_x + 60, y_max=110 + offset),
                region_a=reg_a,
                region_b=reg_b,
                is_blurry=False,
                touches_border=False,
                area_drop=False,
                has_final_state=True,
            )
        )

    # 17-19: UNCERTAIN due to BLUR (Motion blur during recording)
    for i in range(1, 4):
        cases.append(
            ActionTestCase(
                case_id=f"CASE-U_BLUR_{i:02d}",
                name=f"Motion Blur Uncertainty #{i}",
                task_type="moved_A_to_B",
                ground_truth="UNCERTAIN",
                defect_type="BLUR",
                before_bbox=BoundingBox(x_min=40, y_min=40, x_max=100, y_max=100),
                after_bbox=BoundingBox(x_min=430, y_min=230, x_max=490, y_max=290),
                region_a=reg_a,
                region_b=reg_b,
                is_blurry=True,
                touches_border=False,
                area_drop=False,
                has_final_state=True,
                recaptured_after_bbox=BoundingBox(x_min=435, y_min=235, x_max=495, y_max=295),
                recaptured_ground_truth="SUCCESS",  # Stabilized camera confirms success
                generic_recaptured_ground_truth="UNCERTAIN",
            )
        )

    # 20-22: UNCERTAIN due to OUT_OF_FRAME (Boundary contact at border x_max=599)
    for i in range(1, 4):
        cases.append(
            ActionTestCase(
                case_id=f"CASE-U_FRAME_{i:02d}",
                name=f"Boundary Escape Uncertainty #{i}",
                task_type="moved_A_to_B",
                ground_truth="UNCERTAIN",
                defect_type="OUT_OF_FRAME",
                before_bbox=BoundingBox(x_min=40, y_min=40, x_max=100, y_max=100),
                after_bbox=BoundingBox(x_min=540, y_min=230, x_max=600, y_max=290),  # contacts right border
                region_a=reg_a,
                region_b=reg_b,
                is_blurry=False,
                touches_border=True,
                area_drop=False,
                has_final_state=True,
                recaptured_after_bbox=BoundingBox(x_min=440, y_min=230, x_max=500, y_max=290),
                recaptured_ground_truth="SUCCESS",  # Stepping back centers the object
                generic_recaptured_ground_truth="UNCERTAIN",
            )
        )

    # 23-25: UNCERTAIN due to OCCLUSION (Hand or obstacle blocked destination)
    for i in range(1, 4):
        cases.append(
            ActionTestCase(
                case_id=f"CASE-U_OCCLUSION_{i:02d}",
                name=f"Severe Occlusion Uncertainty #{i}",
                task_type="moved_A_to_B",
                ground_truth="UNCERTAIN",
                defect_type="OCCLUSION",
                before_bbox=BoundingBox(x_min=40, y_min=40, x_max=100, y_max=100),
                after_bbox=BoundingBox(x_min=430, y_min=230, x_max=490, y_max=290),
                region_a=reg_a,
                region_b=reg_b,
                is_blurry=False,
                touches_border=False,
                area_drop=True,
                has_final_state=True,
                recaptured_after_bbox=BoundingBox(x_min=430, y_min=230, x_max=490, y_max=290),
                recaptured_ground_truth="SUCCESS",  # Removing hands reveals complete placement
                generic_recaptured_ground_truth="UNCERTAIN",
            )
        )

    return cases
