import base64
import json
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from backend.app import app
from backend.evidence.receipts import generate_receipt, verify_receipt_integrity
from backend.evidence.schemas import AuditVerdict, EvidenceReceipt
from backend.evidence.provenance import (
    ProvenanceGate,
    EvidenceProvenance,
    ProvenanceCategory,
    ProvenanceViolationError,
)

client = TestClient(app)

def test_receipt_integrity_and_tamper_detection():
    img_sha = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    measurements = {
        "iou": 0.92,
        "dice": 0.95,
        "source_image_sha256": img_sha,
    }
    receipt = generate_receipt(
        report_id="RCPT_INT_001",
        engine_name="LabelCourt",
        software_version="v0.2.4",
        inputs_hash="feedbeef1234",
        measurements=measurements,
        signals=[],
        verdict=AuditVerdict.ACCEPT.value,
        provenance={
            "category": "REAL_PHOTOGRAPH",
            "source_id": "COCO_VAL_001",
            "verification_method": "SHA256_HASH_VERIFIED"
        }
    )

    # Must verify initially
    assert verify_receipt_integrity(receipt) is True

    # Tampering measurement invalidates
    receipt_copy = EvidenceReceipt(**receipt.model_dump())
    receipt_copy.measurements["iou"] = 0.50
    assert verify_receipt_integrity(receipt_copy) is False

    # Tampering verdict invalidates
    receipt_copy2 = EvidenceReceipt(**receipt.model_dump())
    receipt_copy2.verdict = AuditVerdict.REVIEW.value
    assert verify_receipt_integrity(receipt_copy2) is False

    # Tampering provenance invalidates
    receipt_copy3 = EvidenceReceipt(**receipt.model_dump())
    receipt_copy3.provenance["category"] = "SYNTHETIC_FIXTURE"
    assert verify_receipt_integrity(receipt_copy3) is False

    # Export / Import independently verifies
    serialized = receipt.model_dump_json()
    deserialized_data = json.loads(serialized)
    imported_receipt = EvidenceReceipt(**deserialized_data)
    assert verify_receipt_integrity(imported_receipt) is True


def test_api_receipt_no_post_hash_mutation():
    img = Image.new("RGB", (64, 64), color=(120, 130, 140))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "human_annotation": {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "category_name": "cup",
            "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        },
        "candidate_bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "image_width": 64,
        "image_height": 64,
        "is_blurry": False,
        "image_base64": raw_b64,
    }
    resp = client.post("/api/labelcourt/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    receipt_data = data["evidence_receipt"]
    receipt = EvidenceReceipt(**receipt_data)
    
    # Must verify cleanly because extra_measurements were passed before receipt creation!
    assert verify_receipt_integrity(receipt) is True
    assert "source_image_sha256" in receipt.measurements
    assert receipt.provenance is not None
    assert receipt.provenance["category"] == "REAL_PHOTOGRAPH"


def test_defect_b_malformed_base64_rejection():
    # Defect B: Malformed base64 or undecodable image returns 400
    payload = {
        "human_annotation": {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "category_name": "cup",
            "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        },
        "candidate_bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "image_width": 64,
        "image_height": 64,
        "is_blurry": False,
        "image_base64": "this_is_not_valid_base64!@#%^&*()",
    }
    resp = client.post("/api/labelcourt/audit", json=payload)
    assert resp.status_code == 400
    assert "Malformed base64" in resp.json()["detail"] or "Unsupported image format" in resp.json()["detail"]


def test_defect_b_dimension_mismatch_rejection():
    # Defect B: Mismatch between image dimensions and declared canvas must return 422
    img = Image.new("RGB", (200, 150), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "human_annotation": {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "category_name": "cup",
            "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        },
        "candidate_bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "image_width": 100,  # Mismatch: 100 != 200
        "image_height": 100, # Mismatch: 100 != 150
        "is_blurry": False,
        "image_base64": raw_b64,
    }
    resp = client.post("/api/labelcourt/audit", json=payload)
    assert resp.status_code == 422
    assert "Dimension mismatch" in resp.json()["detail"]


def test_defect_c_recapture_metadata_only_vs_verified():
    # 1. No follow-up image provided -> METADATA_ONLY
    payload_no_media = {
        "task_id": "RC_DEFECT_C_1",
        "primary_object": "cup",
        "initial_verdict": "UNCERTAIN",
        "detected_defect": "BLUR",
        "directive_issued": "Camera blur detected. Stabilize camera.",
        "initial_region": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
        "target_region": {"x_min": 300, "y_min": 100, "x_max": 400, "y_max": 200},
        "original_before_bbox": {"x_min": 50, "y_min": 50, "x_max": 150, "y_max": 150},
        "followup_bbox": {"x_min": 310, "y_min": 105, "x_max": 390, "y_max": 195},
        "canvas_width": 640,
        "canvas_height": 480,
        "followup_laplacian_variance": 250.0,
        "followup_area_drop_ratio": 0.05,
        "followup_has_final_state": True,
        "instruction_followed": True,
        "effort_seconds": 5.0,
    }
    resp = client.post("/api/recapture/evaluate-followup", json=payload_no_media)
    assert resp.status_code == 200
    data = resp.json()
    assert data["provenance_mode"] == "METADATA_ONLY"
    assert data["evidence_receipt"]["provenance"]["category"] == "USER_SUPPLIED_METADATA"
    assert "evidence_receipt" in data
    assert verify_receipt_integrity(EvidenceReceipt(**data["evidence_receipt"])) is True

    # 2. With real follow-up image provided -> VERIFIED_IMAGE_FOLLOWUP
    img = Image.new("RGB", (640, 480), color=(80, 120, 160))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload_with_media = dict(payload_no_media)
    payload_with_media["followup_image_base64"] = raw_b64

    resp2 = client.post("/api/recapture/evaluate-followup", json=payload_with_media)
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2["provenance_mode"] == "VERIFIED_IMAGE_FOLLOWUP"
    assert data2["evidence_receipt"]["provenance"]["category"] == "REAL_PHOTOGRAPH"
    assert data2["evidence_receipt"]["measurements"]["followup_image_sha256"] is not None
    assert verify_receipt_integrity(EvidenceReceipt(**data2["evidence_receipt"])) is True


def test_evidence_provenance_gate_invariants():
    gate = ProvenanceGate()

    # Rule 1: Geometric fallback cannot claim independent segmentation
    prov_geom = EvidenceProvenance(
        category=ProvenanceCategory.GEOMETRIC_FALLBACK,
        source_identifier="BBOX_FALLBACK_01",
        verification_method="CONVERT_BBOX_TO_RECT_POLYGON"
    )
    with pytest.raises(ProvenanceViolationError, match="Geometric fallback cannot claim independent segmentation"):
        gate.audit_claim(prov_geom, "INDEPENDENT_SEGMENTATION")

    # Rule 2: Human annotation bbox cannot automatically prove object identity
    prov_human = EvidenceProvenance(
        category=ProvenanceCategory.HUMAN_ANNOTATION,
        source_identifier="WORKER_42",
        verification_method="MANUAL_BBOX_DRAW"
    )
    with pytest.raises(ProvenanceViolationError, match="Human-supplied bounding box cannot prove physical object identity"):
        gate.audit_claim(prov_human, "PROVED_OBJECT_IDENTITY")

    # Rule 3: Single static image cannot prove physical rest or release
    prov_single_img = EvidenceProvenance(
        category=ProvenanceCategory.REAL_PHOTOGRAPH,
        source_identifier="SHARP_PHOTO_01",
        verification_method="LAPLACIAN_VARIANCE_SHARP"
    )
    with pytest.raises(ProvenanceViolationError, match="Single static photograph cannot prove physical rest"):
        gate.audit_claim(prov_single_img, "PROVED_PHYSICAL_REST")

    # Rule 4: User supplied metadata cannot claim independent compliance proof
    prov_user_meta = EvidenceProvenance(
        category=ProvenanceCategory.USER_SUPPLIED_METADATA,
        source_identifier="CLIENT_FLAG",
        verification_method="instruction_followed_boolean"
    )
    with pytest.raises(ProvenanceViolationError, match="User-supplied metadata cannot independently prove instruction compliance"):
        gate.audit_claim(prov_user_meta, "INDEPENDENT_COMPLIANCE_PROOF")

    # Valid claims must pass without exception
    prov_real = EvidenceProvenance(
        category=ProvenanceCategory.REAL_PHOTOGRAPH,
        source_identifier="COCO_VAL_000000000139",
        verification_method="FAST_LAPLACIAN_PIXELS",
        dimensions=(640, 480)
    )
    assert gate.audit_claim(prov_real, "MEASURED_IMAGE_QUALITY") is True
