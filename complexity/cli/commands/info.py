"""
Info and utility commands for the Complexity framework.
"""

from pathlib import Path
from typing import Optional

import typer

from ..utils import (
    console,
    error,
    print_panel,
    success,
    warning,
)

info_app = typer.Typer(name="info", help="Information and utility commands")


@info_app.command("version")
def show_version(
    device_info: bool = typer.Option(
        False, "--device-info", help="Initialize PyTorch for CUDA device diagnostics"
    ),
):
    """Show installed versions; device discovery is an explicit operation."""
    from importlib.metadata import PackageNotFoundError, version

    from complexity import __version__

    console.print("[bold]Complexity Framework[/bold]\n")
    console.print(f"  Version: {__version__}")
    console.print("\n[bold]Dependencies:[/bold]")
    dependencies = (
        ("PyTorch", ("torch",)),
        ("Triton", ("triton", "pytorch-triton-rocm")),
        ("Flash Attention", ("flash-attn",)),
        ("Transformers", ("transformers",)),
        ("Tiktoken", ("tiktoken",)),
    )
    for name, distributions in dependencies:
        installed = None
        for distribution in distributions:
            try:
                installed = version(distribution)
                break
            except PackageNotFoundError:
                continue
        color = "green" if installed is not None else "yellow"
        console.print(f"  [{color}]{name}: {installed or 'not installed'}[/{color}]")
    if not device_info:
        return

    console.print("\n[bold]CUDA:[/bold]")
    try:
        import torch

        if torch.cuda.is_available():
            console.print(f"  [green]Available: {torch.cuda.device_count()} GPU(s)[/green]")
            console.print(f"  CUDA version: {torch.version.cuda}")
            for index in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(index)
                memory = torch.cuda.get_device_properties(index).total_memory / 1e9
                console.print(f"  GPU {index}: {name} ({memory:.1f} GB)")
        else:
            console.print("  [yellow]Not available[/yellow]")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")

@info_app.command("model")
def model_info(
    path: Path = typer.Argument(..., help="Model file/export directory or FSDP checkpoint"),
    detailed: bool = typer.Option(
        False, "--detailed", "-d", help="Show safetensors header keys and shapes"
    ),
    verify: bool = typer.Option(
        False, "--verify", help="Hash FSDP checkpoint manifest-listed files"
    ),
    recipe: Optional[Path] = typer.Option(
        None, "--recipe", help="Original FSDP recipe for identity checks"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable metadata"),
):
    """Inspect metadata without loading pickle or materializing model weights.

    Safetensors tensor element counts come from header shapes. For .pt/.bin files,
    this command reports file size only; it never executes torch.load.
    """
    import json
    import math

    from ..inspection import inspect_checkpoint, load_mapping

    try:
        if not path.exists():
            raise ValueError(f"Path not found: {path}")
        if path.is_dir() and (path / "complete.json").is_file():
            report = inspect_checkpoint(path, verify=verify, recipe_path=recipe)
        else:
            if verify or recipe is not None:
                raise ValueError("--verify and --recipe apply only to FSDP manifest checkpoints")
            directory = path if path.is_dir() else path.parent
            is_config = path.is_file() and path.suffix.lower() in {".json", ".yaml", ".yml"}
            config_path = (
                path
                if is_config
                else next(
                    (
                        directory / name
                        for name in ("config.json", "config.yaml", "config.yml")
                        if (directory / name).is_file()
                    ),
                    None,
                )
            )
            config = load_mapping(config_path) if config_path is not None else {}
            architecture = config.get("model", config)
            if not isinstance(architecture, dict):
                raise ValueError("Model config must be a mapping")
            model_files = (
                []
                if is_config
                else list(directory.glob("model*.safetensors"))
                if path.is_dir()
                else [path]
            )
            if not model_files and path.is_dir():
                model_files = [
                    directory / name
                    for name in ("model.pt", "pytorch_model.bin")
                    if (directory / name).is_file()
                ]
            tensors = []
            parameters = None
            dtypes = set()
            warnings = []
            if model_files and all(file.suffix == ".safetensors" for file in model_files):
                from safetensors import safe_open

                parameters = 0
                for file in sorted(model_files):
                    with safe_open(file, framework="numpy") as header:
                        for key in header.keys():
                            tensor = header.get_slice(key)
                            shape = tensor.get_shape()
                            parameters += math.prod(shape)
                            dtype = tensor.get_dtype()
                            dtypes.add(dtype)
                            if detailed:
                                tensors.append({"name": key, "shape": shape, "dtype": dtype})
            elif model_files:
                warnings.append(
                    "Pickle-based weights were not opened; parameter count and dtype remain unverified."
                )
            else:
                warnings.append("No standard model weight files found in this directory.")
            report = {
                "format": "model_metadata",
                "path": str(path),
                "files": [str(file) for file in model_files],
                "file_bytes": sum(file.stat().st_size for file in model_files),
                "tensor_elements_from_safetensors_headers": parameters,
                "dtypes": sorted(dtypes),
                "architecture": architecture,
                "weights_loaded": False,
                "checkpoint_restored": False,
                "warnings": warnings,
            }
            if detailed:
                report["tensor_headers"] = tensors
    except Exception as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "path": str(path)}), err=True)
        else:
            console.print(error(f"Cannot inspect model: {exc}"))
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    if report["format"] == "tr_hash_fsdp_checkpoint":
        text = (
            f"Path: {path}\nStep: {report['step']:,}\nTokens seen: {report['tokens_seen']:,}\n"
            f"Checked: {report['verification_scope']}\nRestored: no"
        )
    else:
        text = (
            f"Path: {path}\nFile bytes: {report['file_bytes']:,}\n"
            f"Tensor elements from safetensors headers: {report['tensor_elements_from_safetensors_headers']}\n"
            f"Dtypes: {', '.join(report['dtypes']) or 'unverified'}\nWeights loaded: no"
        )
        architecture = report["architecture"]
        if architecture:
            text += (
                f"\nHidden size: {architecture.get('hidden_size', 'unverified')}"
                f"\nLayers: {architecture.get('num_hidden_layers', architecture.get('num_layers', 'unverified'))}"
            )
    print_panel(text, title="Model metadata", style="cyan")
    for item in report["warnings"]:
        console.print(warning(item))
    if detailed:
        for tensor in report.get("tensor_headers", [])[:20]:
            console.print(f"{tensor['name']}: {tensor['shape']} {tensor['dtype']}", markup=False)


@info_app.command("config")
def show_config(
    config_path: Path = typer.Argument(..., help="Config file path"),
):
    """
    Display and validate a config file.

    Example:
        complexity info config training_config.yaml
    """
    from .train import validate_config

    validate_config(config_path, json_output=False)


@info_app.command("tokens")
def show_special_tokens():
    """
    Show Complexity special tokens.
    """
    from complexity.data import ComplexityTokens

    tokens = ComplexityTokens()

    categories = [
        ("Core (0-31)", ["unk_token", "bos_token", "eos_token", "pad_token"]),
        ("Chat (32-63)", ["turn_start", "user_start", "assistant_start", "system_start"]),
        ("Tools (64-127)", ["tool_call", "tool_args", "tool_result", "tools_start"]),
        ("Reasoning (128-191)", ["reason_start", "step_start", "conclude", "think_start"]),
        ("Code (192-255)", ["code_start", "output_start", "exec_start"]),
        ("Vision (256-287)", ["image", "vision_start", "bbox"]),
        ("Audio (288-319)", ["audio_start", "speech_start", "transcribe_start"]),
    ]

    console.print("[bold]Complexity Special Tokens[/bold]")
    console.print("[dim]2048 reserved token slots (0-2047)[/dim]\n")

    for cat_name, attrs in categories:
        console.print(f"[bold cyan]{cat_name}[/bold cyan]")
        for attr in attrs:
            if hasattr(tokens, attr):
                val = getattr(tokens, attr)
                console.print(f"  {attr:20s}: {val}")
        console.print()


@info_app.command("benchmark-env")
def benchmark_environment():
    """
    Benchmark the current environment.
    """
    import time

    console.print("[bold]Environment Benchmark[/bold]\n")

    # CPU info
    import platform

    console.print(f"[cyan]Platform:[/cyan] {platform.platform()}")
    console.print(f"[cyan]Python:[/cyan] {platform.python_version()}")

    # PyTorch benchmarks
    try:
        import torch

        console.print(f"\n[bold]PyTorch {torch.__version__}[/bold]")

        # CPU matrix multiply
        size = 2048
        a = torch.randn(size, size)
        b = torch.randn(size, size)

        # Warmup
        for _ in range(3):
            torch.mm(a, b)

        start = time.perf_counter()
        for _ in range(10):
            torch.mm(a, b)
        cpu_time = (time.perf_counter() - start) / 10

        console.print(f"  CPU MatMul ({size}x{size}): {cpu_time * 1000:.2f}ms")

        if torch.cuda.is_available():
            a_cuda = a.cuda()
            b_cuda = b.cuda()

            # Warmup
            for _ in range(10):
                torch.mm(a_cuda, b_cuda)
            torch.cuda.synchronize()

            start = time.perf_counter()
            for _ in range(100):
                torch.mm(a_cuda, b_cuda)
            torch.cuda.synchronize()
            gpu_time = (time.perf_counter() - start) / 100

            console.print(f"  GPU MatMul ({size}x{size}): {gpu_time * 1000:.3f}ms")
            console.print(f"  Speedup: {cpu_time / gpu_time:.1f}x")

            # Memory bandwidth test
            size_mb = 1024
            data = torch.randn(size_mb * 1024 * 1024 // 4, device="cuda")

            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                data.clone()
            torch.cuda.synchronize()
            bw_time = (time.perf_counter() - start) / 10

            bandwidth = (size_mb / bw_time) / 1000  # GB/s
            console.print(f"  Memory bandwidth: ~{bandwidth:.0f} GB/s")

    except ImportError:
        console.print(warning("PyTorch not installed"))

    console.print(success("\nBenchmark complete"))
