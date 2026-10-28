"""Deterministic rule-based missing-evidence recovery planner."""

from __future__ import annotations

from typing import List, Optional
from backend.evidence.schemas import RecaptureInstruction, UncertaintyReason, UncertaintySignal


RECOVERY_TEMPLATES: dict[UncertaintyReason, dict[str, Any]] = {
    UncertaintyReason.BLUR: {
        "directive": "Hold the camera steady and allow the object to become sharp before ending the capture.",
        "rationale": "High motion blur degraded spatial contours below edge-detection thresholds.",
        "checklist": [
            "Stabilize camera on a flat surface or hold with both hands.",
            "Avoid rapid panning during object interaction.",
            "Ensure adequate ambient lighting to reduce camera exposure time.",
        ],
    },
    UncertaintyReason.OUT_OF_FRAME: {
        "directive": "Move the camera back so the full object remains inside the frame.",
        "rationale": "Object boundaries contact the image border, causing unknown geometry outside field of view.",
        "checklist": [
            "Increase camera distance by 15–30 cm.",
            "Center the target interaction region in the middle 70% of the frame.",
            "Verify entire object perimeter remains visible throughout the motion.",
        ],
    },
    UncertaintyReason.OCCLUSION: {
        "directive": "Keep the object visible after releasing it and record the final state.",
        "rationale": "Visible object surface area dropped sharply, indicating obstruction by hands, tools, or obstacles.",
        "checklist": [
            "Withdraw hands completely after placing or interacting with the object.",
            "Choose a viewpoint angled 30–45 degrees from above to avoid self-occlusion.",
            "Ensure foreground items do not block the destination area.",
        ],
    },
    UncertaintyReason.MISSING_FINAL_STATE: {
        "directive": "Continue recording after the action so the completed result is visible.",
        "rationale": "Observation ended prematurely before post-action equilibrium could be verified.",
        "checklist": [
            "Maintain camera focus for at least 1.5 seconds after action concludes.",
            "Keep recording until the object is completely stationary.",
            "Verify that both the object and its destination are clearly shown.",
        ],
    },
    UncertaintyReason.SEGMENTATION_DISAGREEMENT: {
        "directive": "Capture an unobstructed high-contrast view separating the object from the background.",
        "rationale": "Visual boundaries exhibit low contrast, leading to conflicting candidate masks.",
        "checklist": [
            "Improve lighting on the target object.",
            "Avoid matching background colors.",
            "Capture from an angle showing clear silhouette separation.",
        ],
    },
    UncertaintyReason.UNKNOWN: {
        "directive": "The evidence is insufficient to identify the missing information reliably. Capture an unobstructed before-and-after view.",
        "rationale": "Multiple ambiguous degradation factors prevent reliable deterministic diagnosis.",
        "checklist": [
            "Record a clean 2-second stationary before view.",
            "Record the full action smoothly without blocking.",
            "Record a clean 2-second stationary after view.",
        ],
    },
}


def plan_recapture(signals: List[UncertaintySignal]) -> Optional[RecaptureInstruction]:
    """Select the highest-priority uncertainty reason and produce a targeted re-capture instruction."""
    if not signals:
        return None

    # Priority ordering: MISSING_FINAL_STATE > OCCLUSION > OUT_OF_FRAME > BLUR > SEGMENTATION_DISAGREEMENT > UNKNOWN
    priority_order = [
        UncertaintyReason.MISSING_FINAL_STATE,
        UncertaintyReason.OCCLUSION,
        UncertaintyReason.OUT_OF_FRAME,
        UncertaintyReason.BLUR,
        UncertaintyReason.SEGMENTATION_DISAGREEMENT,
        UncertaintyReason.UNKNOWN,
    ]

    selected_reason = UncertaintyReason.UNKNOWN
    for reason in priority_order:
        if any(s.reason == reason for s in signals):
            selected_reason = reason
            break

    template = RECOVERY_TEMPLATES.get(selected_reason, RECOVERY_TEMPLATES[UncertaintyReason.UNKNOWN])
    return RecaptureInstruction(
        reason=selected_reason,
        directive=template["directive"],
        rationale=template["rationale"],
        recovery_checklist=template["checklist"],
    )
