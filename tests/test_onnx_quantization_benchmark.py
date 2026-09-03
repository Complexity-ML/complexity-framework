from scripts.benchmark_onnx_artifacts import summarize_latency_ms


def test_benchmark_report_uses_distribution_not_single_shot() -> None:
    summary = summarize_latency_ms([10.0, 12.0, 14.0])

    assert summary["median_ms"] == 12.0
    assert summary["mean_ms"] == 12.0
    assert summary["stddev_ms"] > 0.0
    assert summary["p95_ms"] == 14.0


def test_benchmark_report_handles_empty_measurements() -> None:
    summary = summarize_latency_ms([])

    assert summary["median_ms"] == 0.0
    assert summary["mean_ms"] == 0.0
    assert summary["stddev_ms"] == 0.0
    assert summary["p95_ms"] == 0.0
