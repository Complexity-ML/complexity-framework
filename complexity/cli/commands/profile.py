"""
Profiling commands for the Complexity framework.
"""

from pathlib import Path
from typing import Optional

import typer

from ..utils import console, error, info, print_panel, print_table, warning

profile = typer.Typer(name="profile", help="Profiling and benchmarking commands")


@profile.command("model")
def profile_model(
    config: Optional[Path] = typer.Argument(None, help="Canonical model config JSON/YAML"),
    hidden_size: Optional[int] = typer.Option(None, "--hidden", "-h", min=1),
    num_layers: Optional[int] = typer.Option(None, "--layers", "-l", min=1),
    num_heads: Optional[int] = typer.Option(None, "--heads", min=1),
    seq_length: int = typer.Option(
        2048, "--seq-length", "-s", min=1, help="Recorded workload shape; no forward pass"
    ),
    batch_size: int = typer.Option(
        1, "--batch-size", "-b", min=1, help="Recorded workload shape; no forward pass"
    ),
    vocab_size: int = typer.Option(50304, "--vocab", min=1),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="List parameter shapes"),
    json_output: bool = typer.Option(
        False, "--json", help="Print machine-readable parameter counts"
    ),
):
    """Count the actual architecture on the meta device without allocating weights.

    Includes GQA, all routed/shared experts, and tied embeddings. No activation
    memory, execution speed, FLOPs or GPU requirements are inferred from counts.
    """
    import json
    import sys
    from contextlib import redirect_stdout

    from ..inspection import inspect_config

    try:
        # Optional kernel import notices belong on stderr so JSON stays parseable.
        with redirect_stdout(sys.stderr):
            import torch

            from complexity import ComplexityModel, ModelConfig

            if config is not None:
                parsed = inspect_config(config)
                model_config = ModelConfig.from_dict(parsed["model"])
            else:
                if None in (hidden_size, num_layers, num_heads):
                    raise ValueError("Provide a config file or --hidden, --layers and --heads")
                model_config = ModelConfig(
                    hidden_size=hidden_size,
                    num_hidden_layers=num_layers,
                    num_attention_heads=num_heads,
                    num_key_value_heads=num_heads,
                    intermediate_size=hidden_size * 4,
                    vocab_size=vocab_size,
                )
            if model_config.lexical_zipf_mode != "uniform":
                raise ValueError(
                    "Meta parameter counting does not load data-dependent lexical Zipf artifacts"
                )
            with torch.device("meta"):
                model = ComplexityModel(model_config)
            named = list(model.named_parameters())
            if any(not value.is_meta for _, value in named):
                raise ValueError(
                    "This architecture allocated real parameter weights; meta profiling is unsupported"
                )
        buffer_bytes = {}
        for value in model.buffers():
            device = str(value.device)
            buffer_bytes[device] = (
                buffer_bytes.get(device, 0) + value.numel() * value.element_size()
            )
        groups = {"embeddings_and_output": 0, "attention": 0, "mlp": 0, "other": 0}
        for name, value in named:
            component = (
                "embeddings_and_output"
                if "embed_tokens" in name or "lm_head" in name
                else "attention"
                if ".self_attn." in name
                else "mlp"
                if ".mlp." in name
                else "other"
            )
            groups[component] += value.numel()
        report = {
            "format": "meta_parameter_profile",
            "parameters": sum(value.numel() for _, value in named),
            "trainable_parameters": sum(value.numel() for _, value in named if value.requires_grad),
            "components": groups,
            "model": model_config.to_dict(),
            "parameter_tensors": len(named),
            "weights_allocated": False,
            "execution_measured": False,
            "memory_measured": False,
            "buffer_bytes_by_device": buffer_bytes,
            "requested_workload": {"batch_size": batch_size, "sequence_length": seq_length},
        }
        if detailed:
            report["parameter_shapes"] = [
                {"name": name, "shape": list(value.shape), "parameters": value.numel()}
                for name, value in named
            ]
    except (ImportError, ValueError, RuntimeError, TypeError, OSError, NotImplementedError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc)}), err=True)
        else:
            console.print(error(f"Parameter profiling failed: {exc}"))
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    print_panel(
        f"Architecture: {model_config.hidden_size}d, {model_config.num_hidden_layers} layers\n"
        f"MLP: {model_config.mlp_type}, experts: {model_config.num_experts}, top_k: {model_config.top_k}\n"
        f"Parameters: {report['parameters']:,}\nTrainable parameters: {report['trainable_parameters']:,}\n"
        f"Parameter tensors: {report['parameter_tensors']}\nWeights allocated: no",
        title="Architecture parameter count",
        style="cyan",
    )
    print_table(
        "Parameter components",
        [{"name": "Component"}, {"name": "Parameters"}],
        list(groups.items()),
    )
    console.print(info("No forward pass, memory measurement or throughput benchmark was run."))
    if detailed:
        for item in report["parameter_shapes"]:
            console.print(f"{item['name']}: {item['shape']}", markup=False)


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

            rows.append(
                (
                    i,
                    props.name,
                    f"{mem_total:.1f} GB",
                    f"{mem_used:.2f} GB",
                    f"{mem_cached:.2f} GB",
                    f"{props.multi_processor_count} SMs",
                )
            )

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
            rows,
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
    console.print(
        error(
            "Native checkpoint throughput profiling is not supported. Use complexity "
            "serve benchmark-server URL --model SERVED_MODEL for endpoint request latency, "
            "or the serving backend's own profiler for actual token throughput."
        )
    )
    raise typer.Exit(2)


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
        q = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
        )
        k = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
        )
        v = torch.randn(
            batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float16
        )

        if attention_type == "flash":
            try:
                from flash_attn import flash_attn_func

                def attn_fn():
                    return flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            except ImportError:
                console.print(warning("Flash attention not installed, using SDPA"))

                def attn_fn():
                    return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        elif attention_type == "sdpa":

            def attn_fn():
                return torch.nn.functional.scaled_dot_product_attention(q, k, v)
        else:
            # Vanilla attention
            scale = head_dim**-0.5

            def attn_fn():
                return torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1) @ v

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

        results.append(
            (
                seq_len,
                f"{avg_time * 1000:.2f} ms",
                f"{tflops:.2f} TF/s",
                f"{mem_mb:.1f} MB",
            )
        )

    print_table(
        f"Attention Profile ({attention_type})",
        [
            {"name": "Seq Len", "style": "cyan"},
            {"name": "Latency", "style": "yellow"},
            {"name": "TFLOPs/s", "style": "green"},
            {"name": "Mem (attn)", "style": "red"},
        ],
        results,
    )
