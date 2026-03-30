"""
Profiling commands for the Complexity framework.
"""

import typer
from pathlib import Path
from typing import Optional

from ..utils import console, spinner, print_panel, print_table, success, error, warning, info

profile = typer.Typer(name="profile", help="Profiling and benchmarking commands")


@profile.command("model")
def profile_model(
    config: Path = typer.Argument(None, help="Model config YAML"),
    hidden_size: int = typer.Option(None, "--hidden", "-h", help="Hidden size (if no config)"),
    num_layers: int = typer.Option(None, "--layers", "-l", help="Number of layers"),
    num_heads: int = typer.Option(None, "--heads", help="Number of attention heads"),
    seq_length: int = typer.Option(2048, "--seq-length", "-s", help="Sequence length"),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size"),
    vocab_size: int = typer.Option(50304, "--vocab", help="Vocabulary size"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed breakdown"),
):
    """
    Profile model memory and compute.

    Examples:
        complexity profile model config.yaml
        complexity profile model --hidden 2048 --layers 24 --heads 16
    """
    import yaml

    if config:
        if not config.exists():
            console.print(error(f"Config not found: {config}"))
            raise typer.Exit(1)
        with open(config) as f:
            cfg = yaml.safe_load(f)
        model_cfg = cfg.get('model', cfg)
        hidden_size = model_cfg.get('hidden_size', 1024)
        num_layers = model_cfg.get('num_layers', 12)
        num_heads = model_cfg.get('num_heads', 8)
        vocab_size = model_cfg.get('vocab_size', 50304)
    elif not all([hidden_size, num_layers, num_heads]):
        console.print(error("Provide config file OR --hidden, --layers, --heads"))
        raise typer.Exit(1)

    try:
        from complexity.profiling import ModelProfiler, estimate_memory

        profiler = ModelProfiler()

        # Calculate parameters
        head_dim = hidden_size // num_heads
        ff_dim = hidden_size * 4

        # Parameter counts
        embed_params = vocab_size * hidden_size * 2  # input + output
        attn_params = num_layers * (4 * hidden_size * hidden_size)  # qkvo
        ff_params = num_layers * (2 * hidden_size * ff_dim)  # up + down
        norm_params = num_layers * (2 * hidden_size)  # 2 norms per layer
        total_params = embed_params + attn_params + ff_params + norm_params

        # Memory estimates (in bytes)
        param_memory = total_params * 2  # fp16
        grad_memory = total_params * 4  # fp32 grads
        optimizer_memory = total_params * 8  # Adam states

        # Activation memory (rough estimate)
        activation_per_layer = batch_size * seq_length * hidden_size * 4 * 2  # ~2 tensors per layer
        total_activation = activation_per_layer * num_layers

        # KV cache
        kv_cache_per_layer = 2 * batch_size * seq_length * hidden_size * 2  # K and V
        total_kv_cache = kv_cache_per_layer * num_layers

        print_panel(
            f"Model: {hidden_size}d, {num_layers}L, {num_heads}H\n"
            f"Vocab: {vocab_size:,}\n"
            f"Seq length: {seq_length}\n"
            f"Batch size: {batch_size}",
            title="Model Configuration",
            style="cyan"
        )

        print_table(
            "Parameter Counts",
            [
                {"name": "Component", "style": "cyan"},
                {"name": "Parameters", "style": "green"},
                {"name": "%", "style": "yellow"},
            ],
            [
                ("Embeddings", f"{embed_params:,}", f"{100*embed_params/total_params:.1f}%"),
                ("Attention", f"{attn_params:,}", f"{100*attn_params/total_params:.1f}%"),
                ("FFN", f"{ff_params:,}", f"{100*ff_params/total_params:.1f}%"),
                ("LayerNorm", f"{norm_params:,}", f"{100*norm_params/total_params:.1f}%"),
                ("", "", ""),
                ("TOTAL", f"{total_params:,}", "100%"),
            ]
        )

        def format_bytes(b):
            if b > 1e9:
                return f"{b/1e9:.2f} GB"
            return f"{b/1e6:.2f} MB"

        print_table(
            "Memory Estimates",
            [
                {"name": "Component", "style": "cyan"},
                {"name": "Memory", "style": "green"},
            ],
            [
                ("Parameters (fp16)", format_bytes(param_memory)),
                ("Gradients (fp32)", format_bytes(grad_memory)),
                ("Optimizer states", format_bytes(optimizer_memory)),
                ("Activations", format_bytes(total_activation)),
                ("KV Cache", format_bytes(total_kv_cache)),
                ("", ""),
                ("Training total", format_bytes(param_memory + grad_memory + optimizer_memory + total_activation)),
                ("Inference total", format_bytes(param_memory + total_kv_cache)),
            ]
        )

        # FLOPs estimate
        flops_per_token = 6 * total_params  # Rough estimate
        total_flops = flops_per_token * batch_size * seq_length

        console.print(f"\n[bold]Compute:[/bold]")
        console.print(f"  FLOPs/token: {flops_per_token/1e9:.2f}G")
        console.print(f"  Total FLOPs: {total_flops/1e12:.2f}T")

        if detailed:
            console.print(f"\n[bold]Per-Layer Breakdown:[/bold]")
            console.print(f"  Attention params: {attn_params//num_layers:,}")
            console.print(f"  FFN params: {ff_params//num_layers:,}")
            console.print(f"  KV cache/layer: {format_bytes(kv_cache_per_layer)}")
            console.print(f"  Activation/layer: {format_bytes(activation_per_layer)}")

    except Exception as e:
        console.print(error(f"Profiling failed: {e}"))
        raise typer.Exit(1)


@profile.command("gpu")
def profile_gpu():
    """
    Show GPU information and memory usage.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            console.print(warning("CUDA not available"))
            return

        num_gpus = torch.cuda.device_count()
        console.print(info(f"Found {num_gpus} GPU(s)"))

        rows = []
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            mem_total = props.total_memory / 1e9
            mem_used = torch.cuda.memory_allocated(i) / 1e9
            mem_cached = torch.cuda.memory_reserved(i) / 1e9

            rows.append((
                i,
                props.name,
                f"{mem_total:.1f} GB",
                f"{mem_used:.2f} GB",
                f"{mem_cached:.2f} GB",
                f"{props.multi_processor_count} SMs",
            ))

        print_table(
            "GPU Information",
            [
                {"name": "ID", "style": "cyan"},
                {"name": "Name", "style": "green"},
                {"name": "Total", "style": "yellow"},
                {"name": "Used", "style": "red"},
                {"name": "Cached", "style": "magenta"},
                {"name": "SMs", "style": "blue"},
            ],
            rows
        )

    except ImportError:
        console.print(error("PyTorch not installed"))
        raise typer.Exit(1)


@profile.command("throughput")
def profile_throughput(
    model_path: Path = typer.Argument(..., help="Model checkpoint"),
    batch_sizes: str = typer.Option("1,2,4,8,16", "--batch-sizes", "-b"),
    seq_lengths: str = typer.Option("512,1024,2048", "--seq-lengths", "-s"),
    iterations: int = typer.Option(10, "--iterations", "-n"),
    device: str = typer.Option("cuda", "--device", "-d"),
):
    """
    Profile model throughput across batch sizes and sequence lengths.

    Example:
        complexity profile throughput model.pt --batch-sizes 1,4,8
    """
    import time
    import torch

    batch_list = [int(b) for b in batch_sizes.split(",")]
    seq_list = [int(s) for s in seq_lengths.split(",")]

    try:
        with spinner("Loading model..."):
            from complexity.inference import InferenceEngine
            engine = InferenceEngine.from_checkpoint(model_path, device=device)

        console.print(success("Model loaded"))

        results = []

        for seq_len in seq_list:
            for bs in batch_list:
                # Create dummy input
                input_ids = torch.randint(0, 50000, (bs, seq_len), device=device)

                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        engine.model(input_ids)

                # Benchmark
                torch.cuda.synchronize()
                start = time.perf_counter()

                for _ in range(iterations):
                    with torch.no_grad():
                        engine.model(input_ids)

                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start

                avg_time = elapsed / iterations
                tokens_per_sec = (bs * seq_len) / avg_time

                results.append((
                    bs,
                    seq_len,
                    f"{avg_time*1000:.2f} ms",
                    f"{tokens_per_sec/1000:.1f}k",
                ))

        print_table(
            "Throughput Results",
            [
                {"name": "Batch", "style": "cyan"},
                {"name": "Seq Len", "style": "yellow"},
                {"name": "Latency", "style": "red"},
                {"name": "Tokens/s", "style": "green"},
            ],
            results
        )

    except Exception as e:
        console.print(error(f"Profiling failed: {e}"))
        raise typer.Exit(1)


@profile.command("attention")
def profile_attention(
    hidden_size: int = typer.Option(2048, "--hidden", "-h"),
    num_heads: int = typer.Option(16, "--heads"),
    seq_lengths: str = typer.Option("1024,2048,4096,8192", "--seq-lengths", "-s"),
    attention_type: str = typer.Option("flash", "--type", "-t", help="flash, sdpa, vanilla"),
    batch_size: int = typer.Option(1, "--batch-size", "-b"),
    iterations: int = typer.Option(50, "--iterations", "-n"),
):
    """
    Profile attention mechanisms.

    Example:
        complexity profile attention --seq-lengths 2048,4096,8192 --type flash
    """
    import time
    import torch

    seq_list = [int(s) for s in seq_lengths.split(",")]

    if not torch.cuda.is_available():
        console.print(error("CUDA required for attention profiling"))
        raise typer.Exit(1)

    device = "cuda"
    head_dim = hidden_size // num_heads

    results = []

    for seq_len in seq_list:
        # Create inputs
        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)

        if attention_type == "flash":
            try:
                from flash_attn import flash_attn_func
                attn_fn = lambda: flash_attn_func(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2))
            except ImportError:
                console.print(warning("Flash attention not installed, using SDPA"))
                attn_fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
        elif attention_type == "sdpa":
            attn_fn = lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            # Vanilla attention
            scale = head_dim ** -0.5
            attn_fn = lambda: torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1) @ v

        # Warmup
        for _ in range(10):
            attn_fn()

        # Benchmark
        torch.cuda.synchronize()
        start = time.perf_counter()

        for _ in range(iterations):
            attn_fn()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        flops = 4 * batch_size * num_heads * seq_len * seq_len * head_dim  # Rough FLOP count
        tflops = flops / avg_time / 1e12

        # Memory for this operation
        mem_bytes = batch_size * num_heads * seq_len * seq_len * 2  # Attention weights
        mem_mb = mem_bytes / 1e6

        results.append((
            seq_len,
            f"{avg_time*1000:.2f} ms",
            f"{tflops:.2f} TF/s",
            f"{mem_mb:.1f} MB",
        ))

    print_table(
        f"Attention Profile ({attention_type})",
        [
            {"name": "Seq Len", "style": "cyan"},
            {"name": "Latency", "style": "yellow"},
            {"name": "TFLOPs/s", "style": "green"},
            {"name": "Mem (attn)", "style": "red"},
        ],
        results
    )
