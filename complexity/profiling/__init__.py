"""
Profiling module for framework-complexity.

Provides comprehensive profiling capabilities:
- Memory profiling and tracking
- FLOPS computation
- Latency measurement
- Throughput analysis

Usage:
    from complexity.profiling import MemoryProfiler, FLOPSCounter, LatencyProfiler

    # Memory profiling
    profiler = MemoryProfiler()
    with profiler.track():
        output = model(input)
    print(profiler.report())

    # FLOPS counting
    flops = FLOPSCounter.count(model, input_shape)
    print(f"Model FLOPS: {flops:,}")

    # Latency profiling
    latency = LatencyProfiler.measure(model, input, num_runs=100)
"""

from .memory_profiler import (
    MemoryProfiler,
    MemoryTracker,
    memory_snapshot,
    get_memory_stats,
    clear_memory,
    estimate_model_memory,
)

from .flops_counter import (
    FLOPSCounter,
    count_flops,
    estimate_macs,
    get_model_complexity,
    ModelComplexity,
)

from .latency_profiler import (
    LatencyProfiler,
    ThroughputBenchmark,
    benchmark_inference,
    benchmark_training,
    ProfileResult,
)

__all__ = [
    # Memory
    "MemoryProfiler",
    "MemoryTracker",
    "memory_snapshot",
    "get_memory_stats",
    "clear_memory",
    "estimate_model_memory",
    # FLOPS
    "FLOPSCounter",
    "count_flops",
    "estimate_macs",
    "get_model_complexity",
    "ModelComplexity",
    # Latency
    "LatencyProfiler",
    "ThroughputBenchmark",
    "benchmark_inference",
    "benchmark_training",
    "ProfileResult",
]
