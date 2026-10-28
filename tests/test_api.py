import numpy as np
import base64
import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)


def _sample_png_b64():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[::2, ::2] = 255
    arr[1::2, 1::2] = 100
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_health_endpoint():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["version"] == "0.1.1"
    assert data["project"] == "ProofCausality AI"
    assert "hardware_acceleration" in data
    assert "LabelCourt" in data["engines"]


def test_model_info_endpoint():
    resp = client.get("/api/model-info")
    assert resp.status_code == 200
    data = resp.json()
    assert "provider" in data
    assert "rust_accelerator" in data


def test_upload_image_api():
    b64_str = _sample_png_b64()
    resp = client.post("/api/image/upload", json={"image_base64": b64_str})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "valid"
    assert data["width"] == 64
    assert data["height"] == 64
    assert len(data["sha256"]) == 64
    assert "laplacian_variance" in data

    # Rejection of invalid data
    bad_resp = client.post("/api/image/upload", json={"image_base64": "invalid_base64_text"})
    assert bad_resp.status_code in (400, 422)


def test_labelcourt_audit_api_with_real_image():
    b64_str = _sample_png_b64()
    payload = {
        "human_annotation": {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "category_name": "tool",
            "bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        },
        "candidate_bbox": {"x_min": 10, "y_min": 10, "x_max": 50, "y_max": 50},
        "image_width": 64,
        "image_height": 64,
        "is_blurry": False,
        "image_base64": b64_str,
    }
    resp = client.post("/api/labelcourt/audit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["audit_result"]["verdict"] == "ACCEPT"
    assert "overlay_data_uri" in data
    assert data["overlay_data_uri"].startswith("data:image/png;base64,")


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


def test_benchmark_endpoints():
    # Partitioned benchmark summary
    resp_all = client.get("/api/benchmark/summary?dataset=all")
    assert resp_all.status_code == 200
    data_all = resp_all.json()
    assert "synthetic_baseline" in data_all
    assert "real_world_evaluation" in data_all

    resp_real = client.get("/api/benchmark/summary?dataset=real")
    assert resp_real.status_code == 200
    assert "false_success_rate" in resp_real.json()

    # Latency comparison benchmark
    resp_lat = client.get("/api/benchmark/latency")
    assert resp_lat.status_code == 200
    lat_data = resp_lat.json()
    assert "kernels" in lat_data
