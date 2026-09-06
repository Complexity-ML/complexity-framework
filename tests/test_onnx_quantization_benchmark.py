import numpy as np

from scripts.benchmark_onnx_artifacts import _benchmark_session, summarize_latency_ms
from scripts.check_onnx_quantized_artifacts import check_quantized_benchmark_report


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


def test_benchmark_report_requires_every_branch_and_precision() -> None:
    thresholds = {
        "release_policy": {"required_precisions": ["fp32", "fp16", "int8"]},
        "benchmark": {
            "report": [
                "median_ms",
                "throughput_images_per_second",
                "peak_memory_mb",
            ]
        },
    }
    report = {
        "branches": {
            "o2m": {
                "fp32": {
                    "latency": {"median_ms": 1.0},
                    "throughput_images_per_second": 10.0,
                    "peak_memory_mb": 100.0,
                }
            }
        }
    }

    failures = check_quantized_benchmark_report(
        report,
        thresholds,
        required_branches=["o2m", "nms-free"],
    )

    assert "benchmark report missing o2m fp16" in failures
    assert "benchmark report missing branch nms-free" in failures


def test_benchmark_session_reports_observed_peak_memory(
    monkeypatch,
) -> None:
    memory_samples = iter([100.0, 110.0, 105.0, 125.0])

    class Session:
        def run(self, values):
            assert values.shape == (1, 3, 4, 4)
            return np.zeros((1, 1), dtype=np.float32)

    class Pipeline:
        class Metadata:
            image_size = 4

        metadata = Metadata()
        session = Session()

    monkeypatch.setattr(
        "scripts.benchmark_onnx_artifacts._current_memory_mb",
        lambda: next(memory_samples),
    )

    latencies, peak_memory_mb = _benchmark_session(
        Pipeline(),
        batch_size=1,
        warmup_iterations=1,
        measured_iterations=2,
    )

    assert len(latencies) == 2
    assert peak_memory_mb == 125.0
