"""
Memory profiling utilities.

Provides:
- Memory tracking during forward/backward passes
- Peak memory analysis
- Memory breakdown by layer
- Memory estimation before allocation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
import gc


@dataclass
class MemoryStats:
    """Memory statistics snapshot."""
    allocated: int = 0           # Currently allocated
    reserved: int = 0            # Reserved by allocator
    peak_allocated: int = 0      # Peak allocated
    peak_reserved: int = 0       # Peak reserved
    num_allocs: int = 0          # Number of allocations

    def __str__(self) -> str:
        return (
            f"Allocated: {self.allocated / 1e9:.2f} GB, "
            f"Reserved: {self.reserved / 1e9:.2f} GB, "
            f"Peak: {self.peak_allocated / 1e9:.2f} GB"
        )


def memory_snapshot() -> MemoryStats:
    """Take a snapshot of current memory state."""
    if not torch.cuda.is_available():
        return MemoryStats()

    torch.cuda.synchronize()

    return MemoryStats(
        allocated=torch.cuda.memory_allocated(),
        reserved=torch.cuda.memory_reserved(),
        peak_allocated=torch.cuda.max_memory_allocated(),
        peak_reserved=torch.cuda.max_memory_reserved(),
    )


def get_memory_stats(device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Get current memory statistics.

    Returns:
        Dictionary with memory stats in GB
    """
    if not torch.cuda.is_available():
        return {
            "allocated_gb": 0,
            "reserved_gb": 0,
            "peak_allocated_gb": 0,
            "free_gb": 0,
        }

    device = device or torch.device('cuda')
    torch.cuda.synchronize(device)

    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    peak = torch.cuda.max_memory_allocated(device)

    # Get total memory
    total = torch.cuda.get_device_properties(device).total_memory
    free = total - reserved

    return {
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "peak_allocated_gb": peak / 1e9,
        "free_gb": free / 1e9,
        "total_gb": total / 1e9,
    }


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class MemoryTracker:
    """
    Track memory allocations during operations.

    Example:
        tracker = MemoryTracker()

        tracker.start()
        output = model(input)
        tracker.stop()

        print(f"Peak: {tracker.peak_memory_gb:.2f} GB")
    """

    def __init__(self, device: Optional[torch.device] = None):
        """
        Args:
            device: Device to track memory on
        """
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.reset()

    def reset(self):
        """Reset tracking state."""
        self.start_allocated = 0
        self.start_reserved = 0
        self.end_allocated = 0
        self.end_reserved = 0
        self.peak_allocated = 0
        self._tracking = False

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    def start(self):
        """Start memory tracking."""
        clear_memory()

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
            self.start_allocated = torch.cuda.memory_allocated(self.device)
            self.start_reserved = torch.cuda.memory_reserved(self.device)

        self._tracking = True

    def stop(self):
        """Stop memory tracking."""
        if not self._tracking:
            return

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)
            self.end_allocated = torch.cuda.memory_allocated(self.device)
            self.end_reserved = torch.cuda.memory_reserved(self.device)
            self.peak_allocated = torch.cuda.max_memory_allocated(self.device)

        self._tracking = False

    @property
    def allocated_memory(self) -> int:
        """Memory allocated during tracking (bytes)."""
        return self.end_allocated - self.start_allocated

    @property
    def allocated_memory_gb(self) -> float:
        """Memory allocated during tracking (GB)."""
        return self.allocated_memory / 1e9

    @property
    def peak_memory(self) -> int:
        """Peak memory during tracking (bytes)."""
        return self.peak_allocated

    @property
    def peak_memory_gb(self) -> float:
        """Peak memory during tracking (GB)."""
        return self.peak_memory / 1e9

    def report(self) -> str:
        """Generate memory report."""
        return (
            f"Memory Tracking Report:\n"
            f"  Start: {self.start_allocated / 1e9:.2f} GB\n"
            f"  End: {self.end_allocated / 1e9:.2f} GB\n"
            f"  Delta: {self.allocated_memory_gb:.2f} GB\n"
            f"  Peak: {self.peak_memory_gb:.2f} GB"
        )


class MemoryProfiler:
    """
    Comprehensive memory profiler.

    Tracks memory per layer and operation.

    Example:
        profiler = MemoryProfiler()

        with profiler.track():
            output = model(input)
            loss = criterion(output, target)
            loss.backward()

        print(profiler.report())
        profiler.plot()  # If matplotlib available
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        track_layers: bool = True,
    ):
        """
        Args:
            model: Model to profile (for layer tracking)
            track_layers: Track memory per layer
        """
        self.model = model
        self.track_layers = track_layers

        self.tracker = MemoryTracker()
        self.layer_memory: Dict[str, int] = {}
        self.hooks: List[Any] = []

        if model and track_layers:
            self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on model layers."""
        def make_hook(name):
            def hook(module, input, output):
                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated()
                    self.layer_memory[name] = mem
            return hook

        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules
                hook = module.register_forward_hook(make_hook(name))
                self.hooks.append(hook)

    def _remove_hooks(self):
        """Remove registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @contextmanager
    def track(self):
        """Context manager for memory tracking."""
        self.layer_memory = {}
        self.tracker.start()

        try:
            yield self
        finally:
            self.tracker.stop()

    def get_layer_memory(self) -> Dict[str, float]:
        """Get memory per layer in GB."""
        return {
            name: mem / 1e9
            for name, mem in self.layer_memory.items()
        }

    def report(self) -> str:
        """Generate detailed memory report."""
        lines = ["=" * 60, "Memory Profiling Report", "=" * 60]

        # Overall stats
        lines.append(f"\nOverall Memory:")
        lines.append(f"  Peak: {self.tracker.peak_memory_gb:.2f} GB")
        lines.append(f"  Delta: {self.tracker.allocated_memory_gb:.2f} GB")

        # Per-layer breakdown
        if self.layer_memory:
            lines.append(f"\nPer-Layer Memory (top 10):")
            sorted_layers = sorted(
                self.layer_memory.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]

            for name, mem in sorted_layers:
                lines.append(f"  {name}: {mem / 1e9:.4f} GB")

        lines.append("=" * 60)
        return "\n".join(lines)

    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


def estimate_model_memory(
    model: nn.Module,
    batch_size: int = 1,
    seq_length: int = 512,
    dtype: torch.dtype = torch.float16,
    include_gradients: bool = True,
    include_optimizer: bool = False,
    optimizer_type: str = "adam",
) -> Dict[str, float]:
    """
    Estimate memory requirements for a model.

    Args:
        model: Model to estimate
        batch_size: Batch size
        seq_length: Sequence length
        dtype: Parameter data type
        include_gradients: Include gradient memory
        include_optimizer: Include optimizer states
        optimizer_type: Type of optimizer (adam, sgd, etc.)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Count parameters
    param_bytes = 0
    for param in model.parameters():
        param_bytes += param.numel() * param.element_size()

    # Estimate for target dtype
    dtype_size = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
    }.get(dtype, 4)

    total_params = sum(p.numel() for p in model.parameters())
    param_memory = total_params * dtype_size

    # Gradient memory
    gradient_memory = param_memory if include_gradients else 0

    # Optimizer state memory
    optimizer_memory = 0
    if include_optimizer:
        if optimizer_type.lower() in ["adam", "adamw"]:
            # Adam stores m and v for each parameter
            optimizer_memory = param_memory * 2 * (4 / dtype_size)  # FP32 states
        elif optimizer_type.lower() == "sgd":
            # SGD with momentum stores v
            optimizer_memory = param_memory * (4 / dtype_size)

    # Activation memory (rough estimate)
    # Assumes transformer architecture
    config = getattr(model, 'config', None)
    if config:
        hidden_size = getattr(config, 'hidden_size', 768)
        num_layers = getattr(config, 'num_hidden_layers', 12)
        num_heads = getattr(config, 'num_attention_heads', 12)

        # Per-layer activations
        # attention: 4 * B * S * H (Q, K, V, output)
        # + attention scores: B * heads * S * S
        # + MLP: 2 * B * S * 4H
        per_layer = (
            4 * batch_size * seq_length * hidden_size * dtype_size +
            batch_size * num_heads * seq_length * seq_length * dtype_size +
            2 * batch_size * seq_length * 4 * hidden_size * dtype_size
        )
        activation_memory = per_layer * num_layers
    else:
        # Rough estimate
        activation_memory = param_memory * batch_size

    total = param_memory + gradient_memory + optimizer_memory + activation_memory

    return {
        "parameters_gb": param_memory / 1e9,
        "gradients_gb": gradient_memory / 1e9,
        "optimizer_gb": optimizer_memory / 1e9,
        "activations_gb": activation_memory / 1e9,
        "total_gb": total / 1e9,
        "total_params": total_params,
    }


class ActivationMemoryProfiler:
    """
    Profile activation memory during forward pass.

    Tracks memory consumed by activations at each layer.
    """

    def __init__(self, model: nn.Module):
        """
        Args:
            model: Model to profile
        """
        self.model = model
        self.activation_sizes: Dict[str, int] = {}
        self.hooks = []

    def _compute_tensor_memory(self, tensor: torch.Tensor) -> int:
        """Compute memory for a tensor."""
        return tensor.numel() * tensor.element_size()

    def _compute_output_memory(self, output: Any) -> int:
        """Compute memory for module output."""
        if isinstance(output, torch.Tensor):
            return self._compute_tensor_memory(output)
        elif isinstance(output, (tuple, list)):
            return sum(self._compute_output_memory(o) for o in output)
        elif isinstance(output, dict):
            return sum(self._compute_output_memory(v) for v in output.values())
        return 0

    def profile(self, input_tensor: torch.Tensor) -> Dict[str, int]:
        """
        Profile activation memory.

        Args:
            input_tensor: Input to the model

        Returns:
            Dictionary mapping layer names to activation memory
        """
        self.activation_sizes = {}

        def make_hook(name):
            def hook(module, input, output):
                mem = self._compute_output_memory(output)
                self.activation_sizes[name] = mem
            return hook

        # Register hooks
        for name, module in self.model.named_modules():
            hook = module.register_forward_hook(make_hook(name))
            self.hooks.append(hook)

        # Run forward pass
        with torch.no_grad():
            self.model(input_tensor)

        # Remove hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

        return self.activation_sizes

    def report(self) -> str:
        """Generate activation memory report."""
        if not self.activation_sizes:
            return "No profiling data. Run profile() first."

        lines = ["Activation Memory Report:", "-" * 40]

        total = 0
        for name, mem in sorted(self.activation_sizes.items(), key=lambda x: -x[1]):
            lines.append(f"{name}: {mem / 1e6:.2f} MB")
            total += mem

        lines.append("-" * 40)
        lines.append(f"Total: {total / 1e9:.2f} GB")

        return "\n".join(lines)
