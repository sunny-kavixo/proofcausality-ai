"""Direct end-to-end latency profiling for the integrated neural vision audit workflow.

Instruments the ACTUAL integrated HTTP request (POST /api/vision/audit):
Authentic photograph -> Image decoding -> Laplacian blur & border check ->
Mask2Former unprompted inference (on Apple MPS) -> Mask postprocessing ->
LabelCourt instance matching -> Grounded verdict -> Canonical Evidence Receipt ->
Visual overlay generation -> Full response.

Measures p50, p95, p99, min, max, and mean directly from complete samples.
Measures both server-side breakdown and complete client wall-clock request duration.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app
from backend.providers.local_segmentation import get_local_mask2former_provider


def get_peak_memory_mb() -> float:
    """Return peak resident set size in megabytes for the current process."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # macOS ru_maxrss is in bytes; Linux is in kilobytes
    if platform.system() == "Darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def run_vision_audit_latency_benchmark(
    num_runs: int = 20,
    warmup_runs: int = 2,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run direct end-to-end benchmark measuring integrated vision audit requests."""
    client = TestClient(app)

    # 1. Fetch authentic COCO sample
    samples_resp = client.get("/api/vision/samples")
    if samples_resp.status_code != 200 or not samples_resp.json():
        raise RuntimeError("Failed to load authentic COCO pilot samples")
    sample = samples_resp.json()[0]

    payload = {
        "image_base64": sample["image_base64"],
        "annotation": sample["annotation"],
        "manifest_image_id": sample["manifest_image_id"],
        "image_source_type": "MANIFEST_VERIFIED_DATASET_IMAGE",
        "settings": {
            "confidence_threshold": 0.50,
            "iou_accept_threshold": 0.70,
            "iou_review_threshold": 0.40,
            "boundary_agreement_threshold": 0.60,
            "matching_strategy": "class_aware",
            "blur_threshold": 100.0,
            "boundary_margin_px": 3,
        },
    }

    # 2. Measure Cold Start (First invocation before warmup)
    t_cold_start = time.perf_counter()
    cold_resp = client.post("/api/vision/audit", json=payload)
    t_cold_end = time.perf_counter()
    assert cold_resp.status_code == 200, f"Cold start failed: {cold_resp.text}"
    cold_data = cold_resp.json()
    cold_client_ms = round((t_cold_end - t_cold_start) * 1000.0, 2)
    cold_server_breakdown = cold_data["latency_breakdown_ms"]

    # 3. Warmup Runs
    for _ in range(warmup_runs):
        w_resp = client.post("/api/vision/audit", json=payload)
        assert w_resp.status_code == 200

    # 4. Measured Warm Invocations
    client_total_durations: list[float] = []
    server_total_durations: list[float] = []
    component_samples: dict[str, list[float]] = {
        "image_decoding_ms": [],
        "preprocessing_ms": [],
        "model_inference_ms": [],
        "mask_postprocessing_ms": [],
        "labelcourt_evaluation_ms": [],
        "overlay_generation_ms": [],
        "receipt_generation_ms": [],
    }

    print(f"Starting {num_runs} measured warm trials...")
    for i in range(num_runs):
        t0 = time.perf_counter()
        resp = client.post("/api/vision/audit", json=payload)
        t1 = time.perf_counter()
        assert resp.status_code == 200
        data = resp.json()

        client_dur_ms = (t1 - t0) * 1000.0
        client_total_durations.append(client_dur_ms)

        breakdown = data["latency_breakdown_ms"]
        server_total_durations.append(breakdown["total_wall_clock_ms"])

        for comp, val in breakdown.items():
            if comp in component_samples:
                component_samples[comp].append(val)

    # 5. Statistical Calculations
    def summarize(values: list[float]) -> dict[str, float]:
        arr = np.array(values)
        return {
            "p50_median_ms": round(float(np.percentile(arr, 50)), 2),
            "p90_ms": round(float(np.percentile(arr, 90)), 2),
            "p95_ms": round(float(np.percentile(arr, 95)), 2),
            "p99_ms": round(float(np.percentile(arr, 99)), 2),
            "mean_ms": round(float(np.mean(arr)), 2),
            "std_dev_ms": round(float(np.std(arr)), 2),
            "min_ms": round(float(np.min(arr)), 2),
            "max_ms": round(float(np.max(arr)), 2),
        }

    provider = get_local_mask2former_provider()

    results: dict[str, Any] = {
        "benchmark_title": "Direct End-to-End Real Vision Audit Latency Benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "workload": {
            "endpoint": "POST /api/vision/audit",
            "protocol": "HTTP via FastAPI TestClient (in-process ASGI roundtrip)",
            "image_filename": sample["file_name"],
            "image_dimensions": f"{sample['image_width']}x{sample['image_height']}",
            "manifest_image_id": sample["manifest_image_id"],
            "category_name": sample["annotation"]["category_name"],
            "detected_instances_count": cold_data["detected_instances_count"],
            "model_identifier": provider.model_id,
            "device": provider.device,
            "checkpoint_sha256": provider._checkpoint_sha256,
        },
        "execution_conditions": {
            "number_of_measured_runs": num_runs,
            "warmup_runs": warmup_runs,
            "concurrency": 1,
            "peak_memory_rss_mb": round(get_peak_memory_mb(), 2),
            "hardware": platform.processor() or platform.machine(),
            "os": f"{platform.system()} {platform.release()}",
            "python_version": platform.python_version(),
        },
        "cold_start": {
            "client_total_wall_clock_ms": cold_client_ms,
            "server_breakdown_ms": cold_server_breakdown,
        },
        "measured_warm_full_request_latency": {
            "client_roundtrip": summarize(client_total_durations),
            "server_total_wall_clock": summarize(server_total_durations),
        },
        "disaggregated_server_components_summary": {
            comp: summarize(vals) for comp, vals in component_samples.items()
        },
        "raw_samples": {
            "client_total_durations_ms": [round(v, 2) for v in client_total_durations],
            "server_total_durations_ms": [round(v, 2) for v in server_total_durations],
        },
    }

    if output_path is None:
        output_path = (
            Path(__file__).resolve().parent.parent
            / "docs"
            / "benchmarks"
            / "real_vision_audit_latency_report.json"
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Benchmark report saved to {output_path}")
    print(f"Warm Client Roundtrip p50: {results['measured_warm_full_request_latency']['client_roundtrip']['p50_median_ms']} ms")
    print(f"Warm Client Roundtrip p95: {results['measured_warm_full_request_latency']['client_roundtrip']['p95_ms']} ms")
    print(f"Warm Server Total p50:     {results['measured_warm_full_request_latency']['server_total_wall_clock']['p50_median_ms']} ms")
    print(f"Warm Server Total p95:     {results['measured_warm_full_request_latency']['server_total_wall_clock']['p95_ms']} ms")
    return results


if __name__ == "__main__":
    run_vision_audit_latency_benchmark(num_runs=20, warmup_runs=2)
