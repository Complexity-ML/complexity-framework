"""
Latency and throughput profiling.

Provides:
- Inference latency measurement
- Training throughput benchmarking
- Token generation speed
- Batch size optimization
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable, Tuple
from dataclasses import dataclass, field
import time
import statistics


@dataclass
class ProfileResult:
    """Result from profiling."""
    mean_latency_ms: float = 0.0
    std_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p90_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput: float = 0.0  # Samples or tokens per second
    num_runs: int = 0
    warmup_runs: int = 0

    def __str__(self) -> str:
        return (
            f"Latency: {self.mean_latency_ms:.2f} ± {self.std_latency_ms:.2f} ms\n"
            f"  Min: {self.min_latency_ms:.2f} ms, Max: {self.max_latency_ms:.2f} ms\n"
            f"  P50: {self.p50_latency_ms:.2f} ms, P90: {self.p90_latency_ms:.2f} ms, P99: {self.p99_latency_ms:.2f} ms\n"
            f"Throughput: {self.throughput:.2f} samples/sec"
        )


class LatencyProfiler:
    """
    Profile inference latency.

    Example:
        profiler = LatencyProfiler()
        result = profiler.measure(model, input_tensor, num_runs=100)
        print(result)
    """

    def __init__(
        self,
        warmup_runs: int = 10,
        sync_cuda: bool = True,
    ):
        """
        Args:
            warmup_runs: Number of warmup iterations
            sync_cuda: Synchronize CUDA before timing
        """
        self.warmup_runs = warmup_runs
        self.sync_cuda = sync_cuda

    def measure(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_runs: int = 100,
        return_outputs: bool = False,
    ) -> ProfileResult:
        """
        Measure inference latency.

        Args:
            model: Model to profile
            input_tensor: Input tensor
            num_runs: Number of measurement runs
            return_outputs: Also return model outputs

        Returns:
            ProfileResult with latency statistics
        """
        model.eval()
        latencies = []

        # Warmup
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                _ = model(input_tensor)

        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure
        with torch.no_grad():
            for _ in range(num_runs):
                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                output = model(input_tensor)

                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # ms

        # Compute statistics
        latencies_sorted = sorted(latencies)

        result = ProfileResult(
            mean_latency_ms=statistics.mean(latencies),
            std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            min_latency_ms=min(latencies),
            max_latency_ms=max(latencies),
            p50_latency_ms=latencies_sorted[len(latencies) // 2],
            p90_latency_ms=latencies_sorted[int(len(latencies) * 0.9)],
            p99_latency_ms=latencies_sorted[int(len(latencies) * 0.99)],
            throughput=1000 / statistics.mean(latencies),  # samples/sec
            num_runs=num_runs,
            warmup_runs=self.warmup_runs,
        )

        return result

    def measure_generation(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        num_tokens: int = 100,
        num_runs: int = 10,
    ) -> Dict[str, float]:
        """
        Measure token generation throughput.

        Args:
            model: Language model
            input_tensor: Input prompt tokens
            num_tokens: Tokens to generate per run
            num_runs: Number of runs

        Returns:
            Dictionary with timing statistics
        """
        model.eval()
        latencies = []
        tokens_per_second = []

        with torch.no_grad():
            for _ in range(num_runs):
                generated = input_tensor.clone()

                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()

                for _ in range(num_tokens):
                    outputs = model(generated)
                    if isinstance(outputs, dict):
                        logits = outputs['logits']
                    else:
                        logits = outputs
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated = torch.cat([generated, next_token], dim=-1)

                if self.sync_cuda and torch.cuda.is_available():
                    torch.cuda.synchronize()

                end = time.perf_counter()
                latency = end - start
                latencies.append(latency)
                tokens_per_second.append(num_tokens / latency)

        return {
            "mean_latency_sec": statistics.mean(latencies),
            "mean_tokens_per_sec": statistics.mean(tokens_per_second),
            "max_tokens_per_sec": max(tokens_per_second),
            "min_tokens_per_sec": min(tokens_per_second),
            "time_per_token_ms": 1000 * statistics.mean(latencies) / num_tokens,
        }


class ThroughputBenchmark:
    """
    Benchmark training throughput.

    Measures samples/second and tokens/second during training.

    Example:
        benchmark = ThroughputBenchmark(model, optimizer, criterion)
        result = benchmark.run(dataloader, num_steps=100)
        print(f"Throughput: {result['samples_per_sec']:.2f} samples/sec")
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Optional[nn.Module] = None,
        gradient_accumulation_steps: int = 1,
    ):
        """
        Args:
            model: Model to benchmark
            optimizer: Optimizer
            criterion: Loss function (defaults to CrossEntropy)
            gradient_accumulation_steps: Gradient accumulation steps
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.gradient_accumulation_steps = gradient_accumulation_steps

    def run(
        self,
        dataloader,
        num_steps: int = 100,
        warmup_steps: int = 10,
    ) -> Dict[str, float]:
        """
        Run throughput benchmark.

        Args:
            dataloader: Training data loader
            num_steps: Number of steps to measure
            warmup_steps: Warmup steps

        Returns:
            Dictionary with throughput metrics
        """
        self.model.train()
        device = next(self.model.parameters()).device

        total_samples = 0
        total_tokens = 0
        step_times = []

        data_iter = iter(dataloader)

        # Warmup
        for _ in range(warmup_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)

            outputs = self.model(input_ids)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        start_time = time.perf_counter()

        for step in range(num_steps):
            step_start = time.perf_counter()

            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            input_ids = batch['input_ids'].to(device)
            labels = batch.get('labels', input_ids).to(device)

            outputs = self.model(input_ids)
            if isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            loss = loss / self.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            step_end = time.perf_counter()
            step_times.append(step_end - step_start)

            total_samples += input_ids.size(0)
            total_tokens += input_ids.numel()

        end_time = time.perf_counter()
        total_time = end_time - start_time

        return {
            "total_time_sec": total_time,
            "samples_per_sec": total_samples / total_time,
            "tokens_per_sec": total_tokens / total_time,
            "mean_step_time_ms": 1000 * statistics.mean(step_times),
            "std_step_time_ms": 1000 * statistics.stdev(step_times) if len(step_times) > 1 else 0,
            "total_samples": total_samples,
            "total_tokens": total_tokens,
            "num_steps": num_steps,
        }


def benchmark_inference(
    model: nn.Module,
    input_shape: Tuple[int, ...],
    num_runs: int = 100,
    warmup_runs: int = 10,
    device: Optional[torch.device] = None,
) -> ProfileResult:
    """
    Convenience function for inference benchmarking.

    Args:
        model: Model to benchmark
        input_shape: Input tensor shape
        num_runs: Number of measurement runs
        warmup_runs: Number of warmup runs
        device: Device for computation

    Returns:
        ProfileResult with latency statistics
    """
    if device is None:
        device = next(model.parameters()).device

    # Create dummy input
    input_tensor = torch.randint(0, 1000, input_shape, device=device)

    profiler = LatencyProfiler(warmup_runs=warmup_runs)
    return profiler.measure(model, input_tensor, num_runs=num_runs)


def benchmark_training(
    model: nn.Module,
    batch_size: int,
    seq_length: int,
    num_steps: int = 100,
    learning_rate: float = 1e-4,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Convenience function for training benchmarking.

    Args:
        model: Model to benchmark
        batch_size: Batch size
        seq_length: Sequence length
        num_steps: Number of training steps
        learning_rate: Learning rate
        device: Device for computation

    Returns:
        Dictionary with throughput metrics
    """
    if device is None:
        device = next(model.parameters()).device

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Create dummy dataloader
    class DummyDataset:
        def __init__(self, batch_size, seq_length, device):
            self.batch_size = batch_size
            self.seq_length = seq_length
            self.device = device

        def __iter__(self):
            while True:
                yield {
                    'input_ids': torch.randint(0, 1000, (self.batch_size, self.seq_length), device=self.device),
                    'labels': torch.randint(0, 1000, (self.batch_size, self.seq_length), device=self.device),
                }

    dataloader = DummyDataset(batch_size, seq_length, device)

    benchmark = ThroughputBenchmark(model, optimizer)
    return benchmark.run(dataloader, num_steps=num_steps)


def find_optimal_batch_size(
    model: nn.Module,
    seq_length: int,
    start_batch: int = 1,
    max_batch: int = 256,
    target_memory_fraction: float = 0.9,
) -> int:
    """
    Find optimal batch size through binary search.

    Args:
        model: Model to test
        seq_length: Sequence length
        start_batch: Starting batch size
        max_batch: Maximum batch size to try
        target_memory_fraction: Target GPU memory utilization

    Returns:
        Optimal batch size
    """
    if not torch.cuda.is_available():
        return start_batch

    device = next(model.parameters()).device
    model.eval()

    # Get available memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    target_memory = total_memory * target_memory_fraction

    optimal_batch = start_batch
    low, high = start_batch, max_batch

    while low <= high:
        mid = (low + high) // 2

        # Clear memory
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            with torch.no_grad():
                input_tensor = torch.randint(0, 1000, (mid, seq_length), device=device)
                _ = model(input_tensor)

            torch.cuda.synchronize()
            peak_memory = torch.cuda.max_memory_allocated()

            if peak_memory <= target_memory:
                optimal_batch = mid
                low = mid + 1
            else:
                high = mid - 1

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                high = mid - 1
                torch.cuda.empty_cache()
            else:
                raise

    return optimal_batch


class ProfilerContext:
    """
    Context manager for combined profiling.

    Example:
        with ProfilerContext(model) as profiler:
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

        print(profiler.report())
    """

    def __init__(
        self,
        model: nn.Module,
        profile_memory: bool = True,
        profile_time: bool = True,
    ):
        """
        Args:
            model: Model to profile
            profile_memory: Enable memory profiling
            profile_time: Enable time profiling
        """
        self.model = model
        self.profile_memory = profile_memory
        self.profile_time = profile_time

        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.peak_memory = None

    def __enter__(self):
        if self.profile_memory and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            self.start_memory = torch.cuda.memory_allocated()

        if self.profile_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start_time = time.perf_counter()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_time:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.end_time = time.perf_counter()

        if self.profile_memory and torch.cuda.is_available():
            self.peak_memory = torch.cuda.max_memory_allocated()

        return False

    @property
    def elapsed_time_ms(self) -> float:
        """Elapsed time in milliseconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0

    @property
    def memory_used_gb(self) -> float:
        """Memory used in GB."""
        if self.peak_memory:
            return self.peak_memory / 1e9
        return 0

    def report(self) -> str:
        """Generate profiling report."""
        lines = ["Profiling Results:"]

        if self.profile_time:
            lines.append(f"  Time: {self.elapsed_time_ms:.2f} ms")

        if self.profile_memory and self.peak_memory:
            lines.append(f"  Peak Memory: {self.memory_used_gb:.2f} GB")

        return "\n".join(lines)
