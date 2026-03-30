"""
Info and utility commands for the Complexity framework.
"""

import typer
from pathlib import Path
from typing import Optional

from ..utils import console, spinner, print_panel, print_table, success, error, warning, info, HAS_RICH

info_app = typer.Typer(name="info", help="Information and utility commands")


@info_app.command("version")
def show_version():
    """
    Show framework version and dependencies.
    """
    console.print("[bold]INL Complexity Framework[/bold]")
    console.print()

    # Framework version
    try:
        from complexity import __version__
        version = __version__
    except ImportError:
        version = "dev"

    console.print(f"  Version: {version}")

    # Check dependencies
    deps = []

    try:
        import torch
        deps.append(("PyTorch", torch.__version__, "green"))
    except ImportError:
        deps.append(("PyTorch", "not installed", "red"))

    try:
        import triton
        deps.append(("Triton", triton.__version__, "green"))
    except ImportError:
        deps.append(("Triton", "not installed", "yellow"))

    try:
        from flash_attn import __version__ as flash_version
        deps.append(("Flash Attention", flash_version, "green"))
    except ImportError:
        deps.append(("Flash Attention", "not installed", "yellow"))

    try:
        import transformers
        deps.append(("Transformers", transformers.__version__, "green"))
    except ImportError:
        deps.append(("Transformers", "not installed", "yellow"))

    try:
        import tiktoken
        deps.append(("Tiktoken", tiktoken.__version__, "green"))
    except ImportError:
        deps.append(("Tiktoken", "not installed", "yellow"))

    console.print("\n[bold]Dependencies:[/bold]")
    for name, ver, color in deps:
        console.print(f"  [{color}]{name}: {ver}[/{color}]")

    # CUDA info
    console.print("\n[bold]CUDA:[/bold]")
    try:
        import torch
        if torch.cuda.is_available():
            console.print(f"  [green]Available: {torch.cuda.device_count()} GPU(s)[/green]")
            console.print(f"  CUDA version: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                console.print(f"  GPU {i}: {name} ({mem:.1f} GB)")
        else:
            console.print("  [yellow]Not available[/yellow]")
    except ImportError:
        console.print("  [red]PyTorch not installed[/red]")


@info_app.command("model")
def model_info(
    path: Path = typer.Argument(..., help="Model path (checkpoint or directory)"),
    detailed: bool = typer.Option(False, "--detailed", "-d", help="Show detailed info"),
):
    """
    Show information about a model.

    Examples:
        complexity info model checkpoint.pt
        complexity info model ./model_dir --detailed
    """
    import yaml

    if not path.exists():
        console.print(error(f"Path not found: {path}"))
        raise typer.Exit(1)

    # Try to find config
    if path.is_dir():
        config_path = path / "config.yaml"
        model_path = path / "model.pt"
        if not model_path.exists():
            model_path = path / "model.safetensors"
    else:
        config_path = path.parent / "config.yaml"
        model_path = path

    # Load config if exists
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("model", config)

    # Get file info
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1e6
        size_str = f"{size_mb:.1f} MB" if size_mb < 1000 else f"{size_mb/1000:.2f} GB"
    else:
        size_str = "N/A"

    # Try to load and inspect
    param_count = "N/A"
    dtype = "N/A"

    if model_path.exists():
        try:
            import torch

            with spinner("Loading model info..."):
                if model_path.suffix == ".safetensors":
                    from safetensors import safe_open
                    with safe_open(model_path, framework="pt") as f:
                        keys = f.keys()
                        total_params = sum(f.get_tensor(k).numel() for k in keys)
                        first_key = next(iter(keys))
                        dtype = str(f.get_tensor(first_key).dtype)
                else:
                    state = torch.load(model_path, map_location="cpu")
                    if "model" in state:
                        state = state["model"]
                    elif "state_dict" in state:
                        state = state["state_dict"]
                    total_params = sum(p.numel() for p in state.values())
                    first_param = next(iter(state.values()))
                    dtype = str(first_param.dtype)

                if total_params > 1e9:
                    param_count = f"{total_params/1e9:.2f}B"
                else:
                    param_count = f"{total_params/1e6:.1f}M"

        except Exception as e:
            console.print(warning(f"Could not load model: {e}"))

    print_panel(
        f"Path: {path}\n"
        f"File size: {size_str}\n"
        f"Parameters: {param_count}\n"
        f"Dtype: {dtype}",
        title="Model Info",
        style="cyan"
    )

    if model_cfg:
        console.print("\n[bold]Architecture:[/bold]")
        console.print(f"  Hidden size: {model_cfg.get('hidden_size', 'N/A')}")
        console.print(f"  Layers: {model_cfg.get('num_layers', 'N/A')}")
        console.print(f"  Heads: {model_cfg.get('num_heads', 'N/A')}")
        console.print(f"  KV Heads: {model_cfg.get('num_kv_heads', 'N/A')}")
        console.print(f"  Vocab size: {model_cfg.get('vocab_size', 'N/A')}")
        console.print(f"  Max seq length: {model_cfg.get('max_seq_length', 'N/A')}")

    if detailed and model_path.exists():
        try:
            import torch

            state = torch.load(model_path, map_location="cpu")
            if "model" in state:
                state = state["model"]

            console.print("\n[bold]Weight Keys:[/bold]")
            for i, key in enumerate(list(state.keys())[:20]):
                shape = list(state[key].shape)
                console.print(f"  {key}: {shape}")

            if len(state) > 20:
                console.print(f"  ... ({len(state) - 20} more)")

        except Exception:
            pass


@info_app.command("config")
def show_config(
    config_path: Path = typer.Argument(..., help="Config file path"),
):
    """
    Display and validate a config file.

    Example:
        complexity info config training_config.yaml
    """
    import yaml

    if not config_path.exists():
        console.print(error(f"Config not found: {config_path}"))
        raise typer.Exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Pretty print
    if HAS_RICH:
        from rich.syntax import Syntax
        yaml_str = yaml.dump(config, default_flow_style=False, sort_keys=False)
        console.print(Syntax(yaml_str, "yaml", theme="monokai"))
    else:
        print(yaml.dump(config, default_flow_style=False))


@info_app.command("tokens")
def show_special_tokens():
    """
    Show INL Complexity special tokens.
    """
    from complexity.data import ComplexityTokens

    tokens = ComplexityTokens()

    categories = [
        ("Core (0-31)", ["unk_token", "bos_token", "eos_token", "pad_token"]),
        ("Chat (32-63)", ["turn_start", "user_start", "assistant_start", "system_start"]),
        ("Tools (64-127)", ["call_start", "args_start", "result_start", "tool_start"]),
        ("Reasoning (128-191)", ["reason_start", "step_start", "conclude", "think_start"]),
        ("Code (192-255)", ["code_start", "output_start", "exec_start"]),
        ("Vision (256-287)", ["image_start", "vision_start", "bbox_start"]),
        ("Audio (288-319)", ["audio_start", "speech_start", "transcribe_start"]),
        ("Robotics (384-511)", ["state_start", "action_start", "trajectory_start"]),
    ]

    console.print("[bold]INL Complexity Special Tokens[/bold]")
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

        console.print(f"  CPU MatMul ({size}x{size}): {cpu_time*1000:.2f}ms")

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

            console.print(f"  GPU MatMul ({size}x{size}): {gpu_time*1000:.3f}ms")
            console.print(f"  Speedup: {cpu_time/gpu_time:.1f}x")

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
