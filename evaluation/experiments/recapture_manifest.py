"""Structured Evidence-Based ReCapture Protocol & Manifest (Phase 3).

Records experimental pairs:
- Initial observation metadata
- Detected missing-evidence cause
- Generated directive (Generic vs Template vs VOI Budget Planner)
- Follow-up recording / observation
- Final verdict & effort
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from backend.evidence.schemas import ActionVerdict, BoundingBox


@dataclass
class ReCaptureRecord:
    record_id: str
    original_case_id: str
    task_description: str
    initial_verdict: str  # "UNCERTAIN"
    detected_defect: str  # "BLUR", "OCCLUSION", "BORDER_CLIPPING", "MISSING_REST_STATE"
    instruction_type: str  # "GENERIC", "FIXED_TEMPLATE", "VOI_PLANNER"
    generated_instruction: str
    follow_up_observation_available: bool
    follow_up_resolved_verdict: Optional[str]  # "SUCCESS", "FAILED", "UNCERTAIN"
    effort_seconds: float
    instruction_followed: bool
    is_real_human_trial: bool


def get_curated_recapture_manifest() -> List[ReCaptureRecord]:
    """Returns curated tabletop re-capture records from real_dataset.py."""
    return [
        ReCaptureRecord(
            record_id="RC-001",
            original_case_id="REAL-U01",
            task_description="Relocate red bowl from A to B",
            initial_verdict="UNCERTAIN",
            detected_defect="BLUR",
            instruction_type="VOI_PLANNER",
            generated_instruction="Camera blur detected. Rest hands against the table and record 2 steady seconds.",
            follow_up_observation_available=True,
            follow_up_resolved_verdict="SUCCESS",
            effort_seconds=4.5,
            instruction_followed=True,
            is_real_human_trial=False,  # Curated paired counterfactual state
        ),
        ReCaptureRecord(
            record_id="RC-002",
            original_case_id="REAL-U02",
            task_description="Transfer blue cup to assembly pad",
            initial_verdict="UNCERTAIN",
            detected_defect="BORDER_CLIPPING",
            instruction_type="VOI_PLANNER",
            generated_instruction="Object clipped by boundary. Pull camera back 20 cm so both start and end locations are fully visible.",
            follow_up_observation_available=True,
            follow_up_resolved_verdict="SUCCESS",
            effort_seconds=6.0,
            instruction_followed=True,
            is_real_human_trial=False,
        ),
        ReCaptureRecord(
            record_id="RC-003",
            original_case_id="REAL-U03",
            task_description="Insert spur gear into gear shaft",
            initial_verdict="UNCERTAIN",
            detected_defect="OCCLUSION",
            instruction_type="VOI_PLANNER",
            generated_instruction="Target object is occluded by gripper. Pan camera 15 cm right to unblock line-of-sight.",
            follow_up_observation_available=True,
            follow_up_resolved_verdict="SUCCESS",
            effort_seconds=7.0,
            instruction_followed=True,
            is_real_human_trial=False,
        ),
        ReCaptureRecord(
            record_id="RC-004",
            original_case_id="REAL-U04",
            task_description="Place marker inside pen holder",
            initial_verdict="UNCERTAIN",
            detected_defect="MISSING_REST_STATE",
            instruction_type="VOI_PLANNER",
            generated_instruction="Recording ended mid-action. Record 2 additional seconds after hand disengagement.",
            follow_up_observation_available=True,
            follow_up_resolved_verdict="SUCCESS",
            effort_seconds=5.0,
            instruction_followed=True,
            is_real_human_trial=False,
        ),
    ]
