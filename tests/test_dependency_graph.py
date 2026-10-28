"""Tests for the Evidence Dependency Graph (H02)."""

import pytest
from backend.evidence.calculus.graph import EvidenceDependencyGraph, build_canonical_task_graph


def test_dependency_graph_construction_and_topological_sort():
    g = EvidenceDependencyGraph()
    g.add_node("root", "OBSERVATION", "Root Frame", initial_value=1)
    g.add_node("mid_1", "DETECTION", "Detector A", parents=["root"], compute_fn=lambda p: p["root"] * 2)
    g.add_node("mid_2", "DETECTION", "Detector B", parents=["root"], compute_fn=lambda p: p["root"] + 5)
    g.add_node("sink", "VERDICT", "Final Verdict", parents=["mid_1", "mid_2"], compute_fn=lambda p: p["mid_1"] + p["mid_2"])

    order = g.get_topological_order()
    assert order == ["root", "mid_1", "mid_2", "sink"] or order == ["root", "mid_2", "mid_1", "sink"]

    full_res = g.recompute_full()
    assert full_res["recomputed_count"] == 3
    assert g.nodes["sink"].value == (1 * 2) + (1 + 5)


def test_cycle_detection():
    g = EvidenceDependencyGraph()
    g.add_node("a", "OBSERVATION", "Node A")
    g.add_node("b", "DETECTION", "Node B", parents=["a"])
    # Artificially inject cycle for cycle detection test
    g.nodes["a"].parents.append("b")

    with pytest.raises(RuntimeError, match="Cycle detected"):
        g.get_topological_order()


def test_dirty_propagation_and_incremental_recompute_parity():
    g = build_canonical_task_graph()
    initial_verdict = g.nodes["verdict_final"].value
    assert initial_verdict == "SUCCESS"

    # Update only the final frame observation to induce blur
    g.compute_fns["measure_blur_final"] = lambda p: 45.0  # Under 100.0 threshold
    g.set_input("obs_final_frame", {"shape": (480, 640), "blur": True})

    # Verify downstream nodes are marked stale
    assert g.nodes["obs_final_frame"].is_stale is False
    assert g.nodes["det_final_cup"].is_stale is True
    assert g.nodes["measure_blur_final"].is_stale is True
    assert g.nodes["pred_observation_sharp"].is_stale is True
    assert g.nodes["verdict_final"].is_stale is True

    # Nodes dependent only on initial frame should NOT be stale
    assert g.nodes["det_initial_cup"].is_stale is False
    assert g.nodes["det_initial_bottle"].is_stale is False

    # Execute incremental recompute
    inc_res = g.recompute_incremental()
    assert inc_res["recomputed_count"] < len(g.nodes)
    assert inc_res["skipped_count"] > 0
    inc_verdict = g.nodes["verdict_final"].value

    # Execute full recompute from scratch
    full_res = g.recompute_full()
    full_verdict = g.nodes["verdict_final"].value

    # Mathematical equivalence check
    assert inc_verdict == full_verdict
    assert inc_verdict == "UNCERTAIN"


def test_export_graph_structure():
    g = build_canonical_task_graph()
    export = g.export_graph()

    assert export["node_count"] == len(g.nodes)
    assert export["edge_count"] > 0
    assert any(n["id"] == "verdict_final" for n in export["nodes"])
    assert any(e["source"] == "obs_final_frame" for e in export["edges"])
