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
    provenance: Optional[dict[str, Any]] = None
    receipt_sha256: str


class VisionAuditAnnotation(BaseModel):
    id: Optional[int | str] = None
    category_name: str
    category_id: Optional[int] = None
    bbox: BoundingBox
    segmentation: Optional[List[List[float]] | List[float]] = None


class VisionAuditSettings(BaseModel):
    confidence_threshold: float = Field(0.50, ge=0.0, le=1.0)
    iou_accept_threshold: float = Field(0.70, ge=0.05, le=1.0)
    iou_review_threshold: float = Field(0.40, ge=0.0, le=0.95)
    boundary_agreement_threshold: float = Field(0.60, ge=0.0, le=1.0)
    matching_strategy: str = Field("class_aware", pattern="^(class_aware|class_agnostic)$")
    blur_threshold: float = Field(100.0, ge=0.0)
    boundary_margin_px: int = Field(3, ge=1, le=50)


class VisionAuditRequest(BaseModel):
    image_base64: str
    annotation: VisionAuditAnnotation
    settings: Optional[VisionAuditSettings] = None
    manifest_image_id: Optional[int] = None
    image_source_type: Optional[str] = "USER_UPLOADED_IMAGE"
    dataset_manifest_ref: Optional[str] = None


class VisionAuditVerdict(str, Enum):
    ACCEPT = "ACCEPT"
    REVIEW = "REVIEW"
    UNCERTAIN = "UNCERTAIN"


class MatchedModelInstance(BaseModel):
    instance_id: int
    label: str
    label_id: int
    score: float
    iou_with_reference: float
    boundary_agreement: float
    class_match: bool
    bbox: BoundingBox
    rle: list[int] | str


class VisionAuditResponse(BaseModel):
    verdict: VisionAuditVerdict
    verdict_reason: str
    explanation: str
    metrics: dict[str, Any]
    matched_prediction: Optional[MatchedModelInstance] = None
    detected_instances_count: int
    all_detected_instances: List[dict[str, Any]]
    model_execution: dict[str, Any]
    evidence_receipt: dict[str, Any]
    receipt_verified: bool
    provenance: dict[str, Any]
    latency_breakdown_ms: dict[str, float]
    visual_overlays: dict[str, str]

