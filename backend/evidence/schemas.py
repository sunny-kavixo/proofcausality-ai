"""Data models and schemas for ProofCausality AI."""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Tuple
from pydantic import BaseModel, Field


class UncertaintyReason(str, Enum):
    OCCLUSION = "OCCLUSION"
    OUT_OF_FRAME = "OUT_OF_FRAME"
    BLUR = "BLUR"
    MISSING_FINAL_STATE = "MISSING_FINAL_STATE"
    SEGMENTATION_DISAGREEMENT = "SEGMENTATION_DISAGREEMENT"
    UNKNOWN = "UNKNOWN"


class AuditVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    UNCERTAIN = "UNCERTAIN"


class ActionVerdict(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


class BoundingBox(BaseModel):
    x_min: float = Field(..., description="Top-left X coordinate")
    y_min: float = Field(..., description="Top-left Y coordinate")
    x_max: float = Field(..., description="Bottom-right X coordinate")
    y_max: float = Field(..., description="Bottom-right Y coordinate")

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centroid(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0)


class AnnotationItem(BaseModel):
    id: int
    image_id: int
    category_id: int
    category_name: str
    bbox: BoundingBox
    segmentation: Optional[List[List[float]]] = None  # Polygons as [[x1, y1, x2, y2, ...]]
    area: Optional[float] = None
    confidence: Optional[float] = 1.0


class UncertaintySignal(BaseModel):
    reason: UncertaintyReason
    severity: str = Field(..., description="LOW, MEDIUM, or HIGH")
    metric_name: str
    metric_value: float
    threshold: float
    description: str


class RecaptureInstruction(BaseModel):
    reason: UncertaintyReason
    directive: str
    rationale: str
    recovery_checklist: List[str]


class AnnotationAuditResult(BaseModel):
    annotation_id: int
    verdict: AuditVerdict
    iou: float
    boundary_agreement: float
    bbox_displacement: float
    area_ratio: float
    class_match: bool
    uncertainty_signals: List[UncertaintySignal]
    recapture_plan: Optional[RecaptureInstruction] = None
    disagreement_heatmap_base64: Optional[str] = None


class ActionVerificationRequest(BaseModel):
    task_type: str = Field(..., description="moved_A_to_B, placed_near, object_present, object_absent")
    object_name: str
    region_a: Optional[BoundingBox] = None
    region_b: Optional[BoundingBox] = None
    target_object_bbox: Optional[BoundingBox] = None
    distance_threshold_px: float = 60.0


class ActionVerificationResult(BaseModel):
    report_id: str
    task_type: str
    verdict: ActionVerdict
    confidence: float
    displacement_px: float
    distance_to_target_px: Optional[float] = None
    uncertainty_signals: List[UncertaintySignal]
    recapture_plan: Optional[RecaptureInstruction] = None
    evidence_receipt_id: str


class EvidenceReceipt(BaseModel):
    report_id: str
    timestamp: str
    software_version: str
    engine_name: str
    inputs_hash: str
    measurements: dict[str, Any]
    uncertainty_signals: List[dict[str, Any]]
    verdict: str
    recapture_directive: Optional[str] = None
    receipt_sha256: str
