from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

"""High-resolution latency benchmarking comparing pure NumPy vs Rust native acceleration."""


import ctypes
import json
import os
import platform
import sys
import time
import numpy as np

from backend.evidence.rust_accelerator import (
    _RUST_LIB,
    is_rust_accelerator_available,
    compute_mask_iou_fast,
    compute_mask_dice_fast,
    compute_batch_mask_iou_fast,
    compute_laplacian_variance_fast,
)


def run_latency_benchmark(output_path: str | None = None) -> dict[str, Any]:
    """Run rigorous comparative benchmark with 200 measured samples per kernel after warm-up."""
    if not is_rust_accelerator_available():
        raise RuntimeError("Rust acceleration library must be compiled and available to run comparison benchmark.")

    results: dict[str, Any] = {
        "benchmark_name": "ProofCausality AI: NumPy vs Rust Native Acceleration Benchmark",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version.split()[0],
        },
        "sample_size_per_kernel": 200,
        "kernels": {},
    }

    test_configs = [
        ("mask_iou_256x256", 256, 256),
        ("mask_iou_1024x1024", 1024, 1024),
        ("mask_iou_2048x2048", 2048, 2048),
    ]

    rng = np.random.default_rng(42)

    for name, h, w in test_configs:
        m1 = (rng.random((h, w)) > 0.5)
        m2 = (rng.random((h, w)) > 0.48)

        # Warm-up
        for _ in range(20):
            _ = np.count_nonzero(m1 & m2) / float(np.count_nonzero(m1 | m2))
            _ = compute_mask_iou_fast(m1, m2)

        # 1. NumPy Timing
        numpy_times = []
        for _ in range(200):
            t0 = time.perf_counter_ns()
            inter = np.count_nonzero(m1 & m2)
            union = np.count_nonzero(m1 | m2)
            _ = inter / float(union) if union > 0 else 1.0
            t1 = time.perf_counter_ns()
            numpy_times.append((t1 - t0) / 1000.0)  # microseconds

        # 2. Rust Timing (including ctypes call overhead)
        rust_times = []
        c1 = np.ascontiguousarray(m1, dtype=np.uint8)
        c2 = np.ascontiguousarray(m2, dtype=np.uint8)
        ptr1 = c1.ctypes.data_as(ctypes.c_void_p)
        ptr2 = c2.ctypes.data_as(ctypes.c_void_p)
        sz = c1.size

        for _ in range(200):
            t0 = time.perf_counter_ns()
            _ = _RUST_LIB.rust_mask_iou(ptr1, ptr2, sz)
            t1 = time.perf_counter_ns()
            rust_times.append((t1 - t0) / 1000.0)

        np_p50 = float(np.percentile(numpy_times, 50))
        rust_p50 = float(np.percentile(rust_times, 50))
        speedup = round(np_p50 / rust_p50, 2) if rust_p50 > 0 else 1.0

        results["kernels"][name] = {
            "dimensions": f"{w}x{h}",
            "numpy": {
                "p50_us": round(np_p50, 2),
                "mean_us": round(float(np.mean(numpy_times)), 2),
                "p95_us": round(float(np.percentile(numpy_times, 95)), 2),
                "min_us": round(float(np.min(numpy_times)), 2),
                "max_us": round(float(np.max(numpy_times)), 2),
            },
            "rust": {
                "p50_us": round(rust_p50, 2),
                "mean_us": round(float(np.mean(rust_times)), 2),
                "p95_us": round(float(np.percentile(rust_times, 95)), 2),
                "min_us": round(float(np.min(rust_times)), 2),
                "max_us": round(float(np.max(rust_times)), 2),
            },
            "speedup_ratio": speedup,
        }
        print(f"[{name}] NumPy p50: {np_p50:.1f}µs | Rust p50: {rust_p50:.1f}µs | Speedup: {speedup}x")

    # 4. Batch IoU (100 pairs of 256x256)
    b1 = (rng.random((100, 256, 256)) > 0.5)
    b2 = (rng.random((100, 256, 256)) > 0.48)

    # Warmup
    for _ in range(5):
        _ = compute_batch_mask_iou_fast(b1, b2)

    numpy_batch_times = []
    flat1 = b1.reshape(100, -1)
    flat2 = b2.reshape(100, -1)
    for _ in range(100):
        t0 = time.perf_counter_ns()
        inter = np.count_nonzero(flat1 & flat2, axis=1)
        union = np.count_nonzero(flat1 | flat2, axis=1)
        _ = inter / union.astype(float)
        t1 = time.perf_counter_ns()
        numpy_batch_times.append((t1 - t0) / 1000.0)

    rust_batch_times = []
    c_b1 = np.ascontiguousarray(b1, dtype=np.uint8)
    c_b2 = np.ascontiguousarray(b2, dtype=np.uint8)
    p_b1 = c_b1.ctypes.data_as(ctypes.c_void_p)
    p_b2 = c_b2.ctypes.data_as(ctypes.c_void_p)
    out_b = np.empty(100, dtype=np.float64)
    p_out = out_b.ctypes.data_as(ctypes.c_void_p)
    m_len = 256 * 256

    for _ in range(100):
        t0 = time.perf_counter_ns()
        _ = _RUST_LIB.rust_batch_mask_iou(p_b1, p_b2, 100, m_len, p_out)
        t1 = time.perf_counter_ns()
        rust_batch_times.append((t1 - t0) / 1000.0)

    b_np_p50 = float(np.percentile(numpy_batch_times, 50))
    b_rust_p50 = float(np.percentile(rust_batch_times, 50))
    b_speedup = round(b_np_p50 / b_rust_p50, 2)

    results["kernels"]["batch_mask_iou_100x256x256"] = {
        "dimensions": "100x256x256",
        "numpy": {
            "p50_us": round(b_np_p50, 2),
            "mean_us": round(float(np.mean(numpy_batch_times)), 2),
            "p95_us": round(float(np.percentile(numpy_batch_times, 95)), 2),
        },
        "rust": {
            "p50_us": round(b_rust_p50, 2),
            "mean_us": round(float(np.mean(rust_batch_times)), 2),
            "p95_us": round(float(np.percentile(rust_batch_times, 95)), 2),
        },
        "speedup_ratio": b_speedup,
    }
    print(f"[batch_iou_100x256x256] NumPy p50: {b_np_p50:.1f}µs | Rust p50: {b_rust_p50:.1f}µs | Speedup: {b_speedup}x")

    # 5. Laplacian Variance 1024x1024
    gray = rng.random((1024, 1024))
    np_lap_times = []
    for _ in range(100):
        t0 = time.perf_counter_ns()
        lap = gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
        _ = np.var(lap)
        t1 = time.perf_counter_ns()
        np_lap_times.append((t1 - t0) / 1000.0)

    rust_lap_times = []
    c_gray = np.ascontiguousarray(gray, dtype=np.float64)
    p_gray = c_gray.ctypes.data_as(ctypes.c_void_p)
    for _ in range(100):
        t0 = time.perf_counter_ns()
        _ = _RUST_LIB.rust_laplacian_variance(p_gray, 1024, 1024)
        t1 = time.perf_counter_ns()
        rust_lap_times.append((t1 - t0) / 1000.0)

    lap_np_p50 = float(np.percentile(np_lap_times, 50))
    lap_rust_p50 = float(np.percentile(rust_lap_times, 50))
    lap_speedup = round(lap_np_p50 / lap_rust_p50, 2)

    results["kernels"]["laplacian_variance_1024x1024"] = {
        "dimensions": "1024x1024",
        "numpy": {
            "p50_us": round(lap_np_p50, 2),
            "mean_us": round(float(np.mean(np_lap_times)), 2),
            "p95_us": round(float(np.percentile(np_lap_times, 95)), 2),
        },
        "rust": {
            "p50_us": round(lap_rust_p50, 2),
            "mean_us": round(float(np.mean(rust_lap_times)), 2),
            "p95_us": round(float(np.percentile(rust_lap_times, 95)), 2),
        },
        "speedup_ratio": lap_speedup,
    }
    print(f"[laplacian_variance_1024x1024] NumPy p50: {lap_np_p50:.1f}µs | Rust p50: {lap_rust_p50:.1f}µs | Speedup: {lap_speedup}x")

    if output_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_path = os.path.join(base_dir, "docs", "benchmarks", "benchmark_latency_comparison.json")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved benchmark results to {output_path}")
    return results


if __name__ == "__main__":
    run_latency_benchmark()
