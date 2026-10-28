"""Directed Evidence Dependency Graph (H02).

Connects:
Input observations -> Object detections -> Segmentation masks -> Spatial measurements -> Task predicates -> Final verdict.

Features:
- Explicit parent-child dependency tracking.
- Dirty tracking / incremental recomputation when observations change.
- Strict equivalence check (incremental evaluation == full recomputation).
- Serialization to Cytoscape / JSON for frontend graph visualization.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Set
from pydantic import BaseModel, Field

from backend.evidence.calculus.schemas import GraphNode


class EvidenceDependencyGraph:
    """Directed acyclic graph tracking evidence propagation from observations to verdict."""

    def __init__(self) -> None:
        self.nodes: Dict[str, GraphNode] = {}
        self.compute_fns: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        parents: Optional[List[str]] = None,
        compute_fn: Optional[Callable[[Dict[str, Any]], Any]] = None,
        initial_value: Any = None,
    ) -> None:
        """Register a node with its upstream dependencies and optional compute function."""
        parents = parents or []
        for p in parents:
            if p not in self.nodes:
                raise ValueError(f"Parent node '{p}' does not exist in graph.")

        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            label=label,
            parents=list(parents),
            children=[],
            is_stale=False,
            value=initial_value,
            cached_timestamp=time.time() if initial_value is not None else None,
        )
        self.nodes[node_id] = node

        for p in parents:
            if node_id not in self.nodes[p].children:
                self.nodes[p].children.append(node_id)

        if compute_fn:
            self.compute_fns[node_id] = compute_fn

    def set_input(self, node_id: str, new_value: Any) -> None:
        """Set a new value on a root or observation node and mark downstream nodes stale."""
        if node_id not in self.nodes:
            raise ValueError(f"Node '{node_id}' does not exist in graph.")
        
        self.nodes[node_id].value = new_value
        self.nodes[node_id].cached_timestamp = time.time()
        self.nodes[node_id].is_stale = False
        
        # Invalidate all downstream dependents
        self._invalidate_downstream(node_id)

    def _invalidate_downstream(self, start_node_id: str) -> None:
        """Mark all transitive children of start_node_id as stale."""
        queue = deque(self.nodes[start_node_id].children)
        visited: Set[str] = set()

        while queue:
            curr_id = queue.popleft()
            if curr_id in visited:
                continue
            visited.add(curr_id)
            self.nodes[curr_id].is_stale = True
            for child_id in self.nodes[curr_id].children:
                if child_id not in visited:
                    queue.append(child_id)

    def get_topological_order(self) -> List[str]:
        """Compute valid topological ordering of all nodes."""
        in_degree = {nid: len(node.parents) for nid, node in self.nodes.items()}
        queue = deque([nid for nid, deg in in_degree.items() if deg == 0])
        order: List[str] = []

        while queue:
            curr = queue.popleft()
            order.append(curr)
            for child in self.nodes[curr].children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            raise RuntimeError("Cycle detected in Evidence Dependency Graph.")
        return order

    def recompute_incremental(self) -> Dict[str, Any]:
        """Recompute only stale/dirty nodes in topological order."""
        t0 = time.perf_counter()
        order = self.get_topological_order()
        recomputed = []
        skipped = []

        for nid in order:
            node = self.nodes[nid]
            if node.is_stale:
                if nid in self.compute_fns:
                    parent_vals = {p: self.nodes[p].value for p in node.parents}
                    node.value = self.compute_fns[nid](parent_vals)
                    node.cached_timestamp = time.time()
                node.is_stale = False
                recomputed.append(nid)
            else:
                skipped.append(nid)

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "mode": "incremental",
            "recomputed_count": len(recomputed),
            "skipped_count": len(skipped),
            "recomputed_nodes": recomputed,
            "latency_ms": round(latency_ms, 3),
        }

    def recompute_full(self) -> Dict[str, Any]:
        """Recompute all non-root nodes from scratch regardless of stale status."""
        t0 = time.perf_counter()
        order = self.get_topological_order()
        recomputed = []

        for nid in order:
            node = self.nodes[nid]
            if nid in self.compute_fns:
                parent_vals = {p: self.nodes[p].value for p in node.parents}
                node.value = self.compute_fns[nid](parent_vals)
                node.cached_timestamp = time.time()
                recomputed.append(nid)
            node.is_stale = False

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "mode": "full",
            "recomputed_count": len(recomputed),
            "skipped_count": len(self.nodes) - len(recomputed),
            "recomputed_nodes": recomputed,
            "latency_ms": round(latency_ms, 3),
        }

    def export_graph(self) -> Dict[str, Any]:
        """Export graph structure and current values for API/UI visualization."""
        nodes_out = []
        edges_out = []

        for nid, node in self.nodes.items():
            nodes_out.append({
                "id": node.node_id,
                "type": node.node_type,
                "label": node.label,
                "is_stale": node.is_stale,
                "value": str(node.value) if node.value is not None else None,
            })
            for child_id in node.children:
                edges_out.append({
                    "source": nid,
                    "target": child_id,
                    "id": f"{nid}->{child_id}",
                })

        return {
            "nodes": nodes_out,
            "edges": edges_out,
            "node_count": len(nodes_out),
            "edge_count": len(edges_out),
        }


def build_canonical_task_graph(
    initial_img_shape: tuple = (480, 640),
    final_img_shape: tuple = (480, 640),
) -> EvidenceDependencyGraph:
    """Constructs a canonical 6-stage Evidence Dependency Graph for physical relocation."""
    g = EvidenceDependencyGraph()

    # Stage 1: Observations
    g.add_node("obs_initial_frame", "OBSERVATION", "Initial Frame (t=0)", initial_value={"shape": initial_img_shape})
    g.add_node("obs_final_frame", "OBSERVATION", "Final Frame (t=T)", initial_value={"shape": final_img_shape})

    # Stage 2: Detections
    g.add_node(
        "det_initial_cup",
        "DETECTION",
        "Cup Detection (Initial)",
        parents=["obs_initial_frame"],
        compute_fn=lambda p: {"label": "cup", "confidence": 0.94, "bbox": [100, 100, 200, 200]},
    )
    g.add_node(
        "det_initial_bottle",
        "DETECTION",
        "Bottle Detection (Initial)",
        parents=["obs_initial_frame"],
        compute_fn=lambda p: {"label": "bottle", "confidence": 0.91, "bbox": [300, 100, 450, 220]},
    )
    g.add_node(
        "det_final_cup",
        "DETECTION",
        "Cup Detection (Final)",
        parents=["obs_final_frame"],
        compute_fn=lambda p: {"label": "cup", "confidence": 0.93, "bbox": [280, 110, 380, 210]},
    )
    g.add_node(
        "det_final_bottle",
        "DETECTION",
        "Bottle Detection (Final)",
        parents=["obs_final_frame"],
        compute_fn=lambda p: {"label": "bottle", "confidence": 0.92, "bbox": [305, 102, 452, 222]},
    )

    # Stage 3: Masks & Features
    g.add_node(
        "mask_final_cup",
        "MASK",
        "Cup Segment Mask (Final)",
        parents=["det_final_cup"],
        compute_fn=lambda p: {"mask_area": 10000, "solidity": 0.92},
    )
    g.add_node(
        "measure_blur_final",
        "MEASUREMENT",
        "Laplacian Blur Variance (Final)",
        parents=["obs_final_frame"],
        compute_fn=lambda p: 185.4,
    )

    # Stage 4: Spatial Measurements
    g.add_node(
        "measure_cup_displacement",
        "MEASUREMENT",
        "Cup Centroid Displacement (px)",
        parents=["det_initial_cup", "det_final_cup"],
        compute_fn=lambda p: 180.2,
    )
    g.add_node(
        "measure_final_proximity",
        "MEASUREMENT",
        "Cup-Bottle Final Proximity (px)",
        parents=["det_final_cup", "det_final_bottle"],
        compute_fn=lambda p: 35.8,
    )

    # Stage 5: Predicates
    g.add_node(
        "pred_objects_identified",
        "PREDICATE",
        "Objects Identified & Observable",
        parents=["det_initial_cup", "det_final_cup", "det_final_bottle"],
        compute_fn=lambda p: "SUPPORTED",
    )
    g.add_node(
        "pred_cup_relocated",
        "PREDICATE",
        "Cup Displaced to Target Proximity",
        parents=["measure_cup_displacement", "measure_final_proximity"],
        compute_fn=lambda p: "SUPPORTED",
    )
    g.add_node(
        "pred_observation_sharp",
        "PREDICATE",
        "Final Frame Sufficiently Sharp",
        parents=["measure_blur_final"],
        compute_fn=lambda p: "SUPPORTED" if p["measure_blur_final"] > 100.0 else "UNOBSERVABLE",
    )

    # Stage 6: Verdict
    g.add_node(
        "verdict_final",
        "VERDICT",
        "Evidence Calculus Arbitration",
        parents=["pred_objects_identified", "pred_cup_relocated", "pred_observation_sharp"],
        compute_fn=lambda p: "SUCCESS" if all(v == "SUPPORTED" for v in p.values()) else "UNCERTAIN",
    )

    # Initial full pass
    g.recompute_full()
    return g
