import os
from evaluation.benchmark import run_benchmarks


def test_benchmark_execution(tmp_path):
    output_dir = str(tmp_path / "benchmarks")
    summary = run_benchmarks(output_dir=output_dir)

    assert summary["sample_size"] == 25
    assert summary["false_success_rate"]["denominator"] == 15
    assert summary["false_success_rate"]["proofcausality_fsr"] == 0.0
    assert summary["recapture_resolution_rate"]["proofcausality_rrr"] == 1.0

    assert os.path.exists(os.path.join(output_dir, "benchmark_results.json"))
    assert os.path.exists(os.path.join(output_dir, "benchmark_cases.csv"))
