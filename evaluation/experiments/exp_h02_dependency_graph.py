"""Experiment H02: Directed Evidence Dependency Graph (H02).

Compares:
A. Full pipeline recomputation
B. Blind cached inference
C. Dependency-aware incremental recomputation

Across 100 perturbed updates on the canonical 6-stage DAG.
"""

from __future__ import annotations

import time
from typing import Any, Dict
import numpy as np

from backend.evidence.calculus.graph import build_canonical_task_graph


def run_experiment_h02() -> Dict[str, Any]:
    n_iterations = 100

    # 1. Full Pipeline Recomputation
    t0 = time.perf_counter()
    full_recomputed_nodes = []
    for i in range(n_iterations):
        g = build_canonical_task_graph()
        g.set_input("obs_final_frame", {"shape": (480, 640), "iteration": i})
        res = g.recompute_full()
        full_recomputed_nodes.append(res["recomputed_count"])
    time_full_ms = (time.perf_counter() - t0) * 1000.0 / n_iterations

    # 2. Dependency-Aware Incremental Recomputation
    t0 = time.perf_counter()
    inc_recomputed_nodes = []
    inc_skipped_nodes = []
    parity_matches = 0

    g_inc = build_canonical_task_graph()
    for i in range(n_iterations):
        # Update only final frame observation
        g_inc.set_input("obs_final_frame", {"shape": (480, 640), "iteration": i})
        res_inc = g_inc.recompute_incremental()
        inc_recomputed_nodes.append(res_inc["recomputed_count"])
        inc_skipped_nodes.append(res_inc["skipped_count"])

        # Check equivalence against full pass
        verdict_inc = g_inc.nodes["verdict_final"].value
        g_full_check = build_canonical_task_graph()
        g_full_check.set_input("obs_final_frame", {"shape": (480, 640), "iteration": i})
        g_full_check.recompute_full()
        verdict_full = g_full_check.nodes["verdict_final"].value

        if verdict_inc == verdict_full:
            parity_matches += 1

    time_inc_ms = (time.perf_counter() - t0) * 1000.0 / n_iterations

    speedup = time_full_ms / time_inc_ms if time_inc_ms > 0 else 1.0

    return {
        "experiment": "H02_DEPENDENCY_GRAPH",
        "iterations": n_iterations,
        "mathematical_parity_rate": round(parity_matches / n_iterations, 4),
        "results": {
            "method_a_full_recomputation": {
                "mean_latency_ms": round(time_full_ms, 3),
                "mean_recomputed_nodes": round(float(np.mean(full_recomputed_nodes)), 1),
                "mean_skipped_nodes": 0.0,
            },
            "method_c_incremental_recomputation": {
                "mean_latency_ms": round(time_inc_ms, 3),
                "mean_recomputed_nodes": round(float(np.mean(inc_recomputed_nodes)), 1),
                "mean_skipped_nodes": round(float(np.mean(inc_skipped_nodes)), 1),
                "speedup_factor": round(speedup, 2),
            },
        },
    }
