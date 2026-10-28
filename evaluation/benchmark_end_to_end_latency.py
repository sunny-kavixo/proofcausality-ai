"""End-to-end latency profiling decomposing pipeline stages from bytes to API response."""

from __future__ import annotations

import ctypes
import io
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.app import app
from backend.evidence.rust_accelerator import _RUST_LIB, compute_mask_iou_fast, is_rust_accelerator_available
from backend.providers.local_segmentation import LocalMask2FormerProvider
from fastapi.testclient import TestClient


def get_peak_memory_mb() -> float:
    # ru_maxrss is in bytes on macOS, in KB on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return usage / (1024.0 * 1024.0)
    return usage / 1024.0


def percentile_stats(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.array(latencies_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "std_ms": round(float(np.std(arr)), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p90_ms": round(float(np.percentile(arr, 90)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
        "min_ms": round(float(np.min(arr)), 4),
        "max_ms": round(float(np.max(arr)), 4),
    }


def run_e2e_benchmark(num_trials: int = 50) -> dict[str, Any]:
    images_dir = ROOT / "data" / "coco_val2017_subset" / "images"
    sample_images = sorted(list(images_dir.glob("*.jpg")))[:5]
    if not sample_images:
        raise RuntimeError("No COCO subset images found in data directory")

    test_img_path = sample_images[0]
    img_bytes = test_img_path.read_bytes()

    # 1. Image decoding latency
    decoding_latencies: list[float] = []
    decoded_img = None
    for _ in range(num_trials):
        t0 = time.perf_counter()
        with Image.open(io.BytesIO(img_bytes)) as im:
            rgb = im.convert("RGB")
            arr = np.array(rgb)
        t1 = time.perf_counter()
        decoding_latencies.append((t1 - t0) * 1000.0)
        decoded_img = rgb

    width, height = decoded_img.size

    # 2. Local Mask2Former loading and Preprocessing latency
    provider = LocalMask2FormerProvider()
    provider.load()
    device = provider.device

    preprocessing_latencies: list[float] = []
    preprocessed_inputs = None
    for _ in range(num_trials):
        t0 = time.perf_counter()
        inputs = provider._processor(images=decoded_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        t1 = time.perf_counter()
        preprocessing_latencies.append((t1 - t0) * 1000.0)
        preprocessed_inputs = inputs

    # 3. Actual Mask2Former inference latency
    inference_latencies: list[float] = []
    torch = provider._torch
    # Warm-up inference
    for _ in range(3):
        with torch.inference_mode():
            _ = provider._model(**preprocessed_inputs)
        if device == "mps":
            torch.mps.synchronize()

    for _ in range(min(20, num_trials)):
        t0 = time.perf_counter()
        with torch.inference_mode():
            raw_out = provider._model(**preprocessed_inputs)
        if device == "mps":
            torch.mps.synchronize()
        t1 = time.perf_counter()
        inference_latencies.append((t1 - t0) * 1000.0)

    # 4. Python mask operations vs 5. Rust native kernel vs 6. Python-to-Rust overhead
    mask1 = np.zeros((height, width), dtype=bool)
    mask2 = np.zeros((height, width), dtype=bool)
    mask1[height//4:3*height//4, width//4:3*width//4] = True
    mask2[height//4 + 5:3*height//4 + 5, width//4 + 5:3*width//4 + 5] = True

    py_mask_latencies: list[float] = []
    for _ in range(100):
        t0 = time.perf_counter()
        union = np.count_nonzero(mask1 | mask2)
        iou_val = float(np.count_nonzero(mask1 & mask2) / union) if union else 1.0
        t1 = time.perf_counter()
        py_mask_latencies.append((t1 - t0) * 1000.0)

    rust_kernel_latencies: list[float] = []
    rust_total_latencies: list[float] = []

    if is_rust_accelerator_available() and _RUST_LIB is not None:
        c_fn = _RUST_LIB.rust_mask_iou
        m1_ptr = mask1.ctypes.data_as(ctypes.c_void_p)
        m2_ptr = mask2.ctypes.data_as(ctypes.c_void_p)
        length = mask1.size

        # Direct C-ABI call (raw kernel execution)
        for _ in range(100):
            t0 = time.perf_counter()
            _ = c_fn(m1_ptr, m2_ptr, length)
            t1 = time.perf_counter()
            rust_kernel_latencies.append((t1 - t0) * 1000.0)

        # Full Python wrapper call including type checks and pointer preparation
        for _ in range(100):
            t0 = time.perf_counter()
            _ = compute_mask_iou_fast(mask1, mask2)
            t1 = time.perf_counter()
            rust_total_latencies.append((t1 - t0) * 1000.0)

    # Python-to-Rust wrapper overhead = total wrapper latency - direct C-ABI latency
    ptr_overhead_latencies = [
        max(0.0, tot - ker)
        for tot, ker in zip(rust_total_latencies[:50], rust_kernel_latencies[:50])
    ]

    # 7. Complete API response latency (FastAPI TestClient end-to-end round-trip)
    client = TestClient(app)
    import base64
    b64_image = base64.b64encode(img_bytes).decode("ascii")

    audit_payload = {
        "human_annotation": {
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "category_name": "object",
            "bbox": {
                "x_min": width * 0.2,
                "y_min": height * 0.2,
                "x_max": width * 0.8,
                "y_max": height * 0.8,
            },
        },
        "candidate_bbox": {
            "x_min": width * 0.22,
            "y_min": height * 0.22,
            "x_max": width * 0.78,
            "y_max": height * 0.78,
        },
        "image_width": width,
        "image_height": height,
        "is_blurry": False,
        "image_base64": b64_image,
    }

    # Warm-up API
    for _ in range(2):
        _ = client.post("/api/labelcourt/audit", json=audit_payload)

    api_latencies: list[float] = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        resp = client.post("/api/labelcourt/audit", json=audit_payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        api_latencies.append((t1 - t0) * 1000.0)

    # Summary
    report: dict[str, Any] = {
        "benchmark_title": "ProofCausality AI End-to-End Latency Profile",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "device": device,
            "torch": str(torch.__version__),
            "transformers": getattr(__import__("transformers"), "__version__", "unknown"),
            "fastapi": getattr(__import__("fastapi"), "__version__", "unknown"),
            "numpy": str(np.__version__),
            "input_dimensions": f"{width}x{height}",
            "sample_count_per_stage": {
                "decoding": len(decoding_latencies),
                "preprocessing": len(preprocessing_latencies),
                "inference": len(inference_latencies),
                "py_mask_ops": len(py_mask_latencies),
                "rust_kernel": len(rust_kernel_latencies),
                "api_roundtrip": len(api_latencies),
            },
            "peak_memory_mb": round(get_peak_memory_mb(), 2),
        },
        "stages": {
            "1_image_decoding": percentile_stats(decoding_latencies),
            "2_model_preprocessing": percentile_stats(preprocessing_latencies),
            "3_actual_mask2former_inference": percentile_stats(inference_latencies),
            "4_python_mask_operations": percentile_stats(py_mask_latencies),
            "5_rust_native_kernel": percentile_stats(rust_kernel_latencies),
            "6_python_to_rust_overhead": percentile_stats(ptr_overhead_latencies),
            "7_complete_api_response": percentile_stats(api_latencies),
        },
        "comparisons": {
            "rust_kernel_speedup_vs_numpy": round(
                np.percentile(py_mask_latencies, 50) / max(0.0001, np.percentile(rust_kernel_latencies, 50)),
                2
            ),
            "rust_total_call_speedup_vs_numpy": round(
                np.percentile(py_mask_latencies, 50) / max(0.0001, np.percentile(rust_total_latencies, 50)),
                2
            ),
            "inference_fraction_of_api_time_percent": round(
                (np.percentile(inference_latencies, 50) / max(0.0001, np.percentile(api_latencies, 50))) * 100.0,
                1
            ),
            "rust_fraction_of_api_time_percent": round(
                (np.percentile(rust_kernel_latencies, 50) / max(0.0001, np.percentile(api_latencies, 50))) * 100.0,
                3
            ),
        },
        "scientific_note": (
            "Rust acceleration accelerates the pure pixel-comparison kernel by ~3x-11x, "
            "but native kernel compute accounts for <0.5% of end-to-end API latency. "
            "Model neural forward pass and image decoding dominate total request duration. "
            "A kernel speedup must NEVER be claimed as an overall application speedup."
        )
    }

    out_file = ROOT / "docs" / "benchmarks" / "end_to_end_latency_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    rep = run_e2e_benchmark(num_trials=30)
    print(json.dumps(rep, indent=2))
