"""Benchmark Vision v8 ONNX artifacts with stable release-report methodology."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provider", action="append", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iterations", type=int, default=25)
    parser.add_argument("--measured-iterations", type=int, default=100)
    parser.add_argument("--ort-intra-op-threads", type=int, default=1)
    parser.add_argument("--ort-inter-op-threads", type=int, default=1)
    return parser.parse_args()


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * fraction), len(ordered) - 1)
    return float(ordered[index])


def summarize_latency_ms(values: Sequence[float]) -> dict[str, float]:
    """Summarize latency as a distribution to avoid single-shot noise."""

    if not values:
        return {
            "median_ms": 0.0,
            "mean_ms": 0.0,
            "stddev_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
        }
    return {
        "median_ms": float(statistics.median(values)),
        "mean_ms": float(statistics.fmean(values)),
        "stddev_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
    }


def _current_memory_mb() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)


def _benchmark_session(
    pipeline: Any,
    *,
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
) -> tuple[list[float], float | None]:
    input_shape = (
        batch_size,
        3,
        pipeline.metadata.image_size,
        pipeline.metadata.image_size,
    )
    dummy = np.zeros(input_shape, dtype=np.float32)
    peak_memory_mb = _current_memory_mb()
    for _ in range(warmup_iterations):
        pipeline.session.run(dummy)
        peak_memory_mb = _max_optional(peak_memory_mb, _current_memory_mb())

    latencies: list[float] = []
    for _ in range(measured_iterations):
        started = time.perf_counter()
        pipeline.session.run(dummy)
        latencies.append((time.perf_counter() - started) * 1000.0)
        peak_memory_mb = _max_optional(peak_memory_mb, _current_memory_mb())
    return latencies, peak_memory_mb


def _max_optional(first: float | None, second: float | None) -> float | None:
    if first is None:
        return second
    if second is None:
        return first
    return max(first, second)


def benchmark_onnx_artifact(
    *,
    model_path: Path,
    metadata_path: Path,
    providers: Sequence[str],
    batch_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    ort_intra_op_threads: int,
    ort_inter_op_threads: int,
) -> dict[str, Any]:
    from complexity.deploy.onnx_detector import OnnxDetectorPipeline
    from scripts.quantize_onnx import package_version, sha256_file

    if batch_size <= 0 or warmup_iterations < 0 or measured_iterations <= 0:
        raise ValueError("batch size and measured iterations must be positive")
    pipeline = OnnxDetectorPipeline.from_files(
        model_path,
        metadata_path,
        providers=providers,
        intra_op_num_threads=ort_intra_op_threads,
        inter_op_num_threads=ort_inter_op_threads,
    )
    latencies, peak_memory_mb = _benchmark_session(
        pipeline,
        batch_size=batch_size,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    summary = summarize_latency_ms(latencies)
    mean_ms = summary["mean_ms"]
    throughput = (batch_size * 1000.0 / mean_ms) if mean_ms else 0.0
    return {
        "schema_version": 1,
        "model": str(model_path),
        "metadata": str(metadata_path),
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "requested_provider": list(providers),
        "actual_provider": pipeline.session.provider_used,
        "batch_size": batch_size,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "latency": summary,
        "throughput_images_per_second": throughput,
        "peak_memory_mb": peak_memory_mb,
        "benchmark_methodology": ("fixed warmup, fixed measured iterations, latency distribution"),
        "environment": {
            "python": sys.version.split()[0],
            "os": platform.platform(),
            "onnxruntime": package_version("onnxruntime"),
        },
        "ort_intra_op_threads": ort_intra_op_threads,
        "ort_inter_op_threads": ort_inter_op_threads,
    }


def main() -> None:
    from scripts.onnx_detect import provider_names

    args = parse_args()
    if not math.isfinite(float(args.batch_size)):
        raise ValueError("batch size must be finite")
    report = benchmark_onnx_artifact(
        model_path=args.model,
        metadata_path=args.metadata,
        providers=provider_names(args.provider),
        batch_size=args.batch_size,
        warmup_iterations=args.warmup_iterations,
        measured_iterations=args.measured_iterations,
        ort_intra_op_threads=args.ort_intra_op_threads,
        ort_inter_op_threads=args.ort_inter_op_threads,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
