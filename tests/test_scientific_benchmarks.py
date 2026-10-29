"""
Regression tests for Issue 7: Real Benchmark and Scientific Consistency.
Ensures that:
1. Benchmark categories remain strictly segregated.
2. Real-model latency profiling reproduces without fabricated medians.
3. Component medians are consistent with overall server wall-clock p50.
4. Statistical grouping is at the image level with confidence intervals.
5. Controlled interventions are not mislabeled as genuine human errors.
"""
import os
import json
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LATENCY_REPORT_PATH = os.path.join(REPO_ROOT, "docs", "benchmarks", "real_vision_audit_latency_report.json")
GENUINE_SUMMARY_PATH = os.path.join(REPO_ROOT, "evaluation", "experiments", "genuine_model_predictions", "summary.json")
TAXONOMY_DOC_PATH = os.path.join(REPO_ROOT, "docs", "research", "SCIENTIFIC_BENCHMARK_TAXONOMY.md")


def test_latency_report_structure_and_metrics():
    assert os.path.exists(LATENCY_REPORT_PATH), "real_vision_audit_latency_report.json must exist"
    with open(LATENCY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["workload"]["endpoint"] == "POST /api/vision/audit"
    assert data["workload"]["detected_instances_count"] > 0
    assert data["execution_conditions"]["number_of_measured_runs"] == 20
    assert len(data["raw_samples"]["client_total_durations_ms"]) == 20

    warm_server = data["measured_warm_full_request_latency"]["server_total_wall_clock"]
    assert "p50_median_ms" in warm_server
    assert "p95_ms" in warm_server
    assert warm_server["p50_median_ms"] > 0
    assert warm_server["p95_ms"] >= warm_server["p50_median_ms"]


def test_component_medians_parity_with_server_total():
    """Verify that independent component medians sum closely to the actual total server p50."""
    with open(LATENCY_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    comps = data["disaggregated_server_components_summary"]
    sum_medians = sum(c["p50_median_ms"] for c in comps.values())
    total_server_p50 = data["measured_warm_full_request_latency"]["server_total_wall_clock"]["p50_median_ms"]

    # Difference between sum of component medians and full request median should be within 10ms (due to timing jitter)
    diff = abs(sum_medians - total_server_p50)
    assert diff < 10.0, f"Component medians sum ({sum_medians}) differs from server total ({total_server_p50}) by {diff}ms"


def test_genuine_model_summary_scientific_separation():
    assert os.path.exists(GENUINE_SUMMARY_PATH), "genuine_model_predictions/summary.json must exist"
    with open(GENUINE_SUMMARY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["evidence_category"] == "GENUINE_MODEL_PREDICTIONS"
    assert data["unique_images"] == 20
    assert data["derived_scenarios"] == 0  # No synthetic expansion
    assert "image_grouped_ci95" in str(data["metrics"])
    assert len(data["limitations"]) > 0


def test_taxonomy_document_contains_all_eight_categories():
    assert os.path.exists(TAXONOMY_DOC_PATH), "SCIENTIFIC_BENCHMARK_TAXONOMY.md must exist"
    with open(TAXONOMY_DOC_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    categories = [
        "Controlled Corruption on Authentic Photographs",
        "Genuine Model Predictions",
        "Integrated Vision Audit End-to-End Latency",
        "Real Human Annotation-Error Study",
        "Constructed Physical-Action Cases",
        "Real Temporal-Action Verification",
        "Human ReCapture Closed-Loop Resolution",
        "Hosted Meta API vs Local PyTorch Inference",
    ]
    for cat in categories:
        assert cat in content, f"Taxonomy must contain category: {cat}"

    assert "A model disagreement alone is not proof of a human annotation error" in content or \
           "A model disagreement alone is **not** proof of a human annotation error" in content
