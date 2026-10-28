import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["project"] == "ProofCausality AI"
    assert "LabelCourt" in data["engines"]


def test_model_info_endpoint():
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "provider" in data
    assert data["api_key_status"] in ("[NOT_SET]", "[REDACTED]")


def test_labelcourt_audit_api():
    payload = {
        "human_annotation": {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "category_name": "tool",
            "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        },
        "candidate_bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "image_width": 200,
        "image_height": 200,
        "is_blurry": False,
    }
    resp = client.post("/api/labelcourt/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["audit_result"]["verdict"] == "ACCEPT"
    assert data["receipt_valid"] is True

    # Test receipt retrieval
    receipt_id = data["evidence_receipt"]["report_id"]
    r_resp = client.get(f"/api/receipts/{receipt_id}")
    assert r_resp.status_code == 200
    assert r_resp.json()["cryptographic_integrity_verified"] is True


def test_actionproof_verify_api():
    payload = {
        "request": {
            "task_type": "moved_A_to_B",
            "object_name": "wrench",
            "region_a": {"x_min": 10, "y_min": 10, "x_max": 60, "y_max": 60},
            "region_b": {"x_min": 120, "y_min": 120, "x_max": 180, "y_max": 180},
            "distance_threshold_px": 50.0,
        },
        "before_bbox": {"x_min": 15, "y_min": 15, "x_max": 45, "y_max": 45},
        "after_bbox": {"x_min": 130, "y_min": 130, "x_max": 160, "y_max": 160},
        "image_width": 200,
        "image_height": 200,
        "is_blurry": False,
    }
    resp = client.post("/api/actionproof/verify", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["verification_result"]["verdict"] == "SUCCESS"
    assert data["receipt_valid"] is True
