"""
Training commands for the Complexity framework.
"""

import typer
from pathlib import Path
from typing import Optional

from ..utils import console, spinner, print_panel, success, error, warning, info

train = typer.Typer(name="train", help="Training commands")


@train.command("run")
def run_training(
    config: Path = typer.Argument(..., help="Path to training config YAML"),
    checkpoint: Optional[Path] = typer.Option(None, "--checkpoint", "-c", help="Resume from checkpoint"),
    output_dir: Path = typer.Option(Path("outputs"), "--output", "-o", help="Output directory"),
    num_gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPUs"),
    distributed: bool = typer.Option(False, "--distributed", "-d", help="Use distributed training"),
    mixed_precision: str = typer.Option("bf16", "--precision", "-p", help="Mixed precision: fp32, fp16, bf16"),
    wandb_project: Optional[str] = typer.Option(None, "--wandb", help="W&B project name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config without training"),
):
    """
    Run training from a config file.

    Examples:
        complexity train run config.yaml
        complexity train run config.yaml --gpus 8 --distributed
        complexity train run config.yaml --checkpoint checkpoints/step-1000
    """
    import yaml

    if not config.exists():
        console.print(error(f"Config file not found: {config}"))
        raise typer.Exit(1)

    # Load config
    with spinner("Loading config..."):
        with open(config) as f:
            cfg = yaml.safe_load(f)

    console.print(success(f"Loaded config from {config}"))

    # Show config summary
    print_panel(
        f"Model: {cfg.get('model', {}).get('name', 'unknown')}\n"
        f"Hidden dim: {cfg.get('model', {}).get('hidden_size', 'N/A')}\n"
        f"Layers: {cfg.get('model', {}).get('num_layers', 'N/A')}\n"
        f"Batch size: {cfg.get('training', {}).get('batch_size', 'N/A')}\n"
        f"Learning rate: {cfg.get('training', {}).get('learning_rate', 'N/A')}\n"
        f"Max steps: {cfg.get('training', {}).get('max_steps', 'N/A')}",
        title="Training Config",
        style="cyan"
    )

    if dry_run:
        console.print(info("Dry run - skipping training"))
        return

    # Setup training
    try:
        from complexity.training import Trainer, TrainingConfig
        from complexity.core import TransformerConfig, Transformer
        from complexity.parallel import ParallelConfig

        # Build model config
        model_cfg = cfg.get('model', {})
        train_cfg = cfg.get('training', {})

        # Create trainer
        with spinner("Initializing trainer..."):
            trainer_config = TrainingConfig(
                max_steps=train_cfg.get('max_steps', 10000),
                batch_size=train_cfg.get('batch_size', 32),
                learning_rate=train_cfg.get('learning_rate', 1e-4),
                weight_decay=train_cfg.get('weight_decay', 0.01),
                warmup_steps=train_cfg.get('warmup_steps', 1000),
                gradient_accumulation_steps=train_cfg.get('gradient_accumulation_steps', 1),
                mixed_precision=mixed_precision,
                checkpoint_dir=str(output_dir / "checkpoints"),
                log_dir=str(output_dir / "logs"),
            )

            if distributed:
                parallel_config = ParallelConfig(
                    tensor_parallel_size=num_gpus,
                    pipeline_parallel_size=1,
                )
            else:
                parallel_config = None

        console.print(success("Trainer initialized"))

        # Start training
        console.print(info(f"Starting training with {num_gpus} GPU(s)..."))
        console.print(warning("Training loop not fully implemented - framework demonstration"))

    except ImportError as e:
        console.print(error(f"Missing dependency: {e}"))
        console.print(info("Install with: pip install complexity[training]"))
        raise typer.Exit(1)


@train.command("resume")
def resume_training(
    checkpoint_dir: Path = typer.Argument(..., help="Checkpoint directory to resume from"),
    steps: Optional[int] = typer.Option(None, "--steps", "-s", help="Additional steps to train"),
):
    """
    Resume training from a checkpoint.

    Example:
        complexity train resume outputs/checkpoints/step-5000 --steps 10000
    """
    if not checkpoint_dir.exists():
        console.print(error(f"Checkpoint not found: {checkpoint_dir}"))
        raise typer.Exit(1)

    console.print(info(f"Resuming from: {checkpoint_dir}"))
    console.print(warning("Resume not fully implemented - framework demonstration"))


@train.command("validate")
def validate_config(
    config: Path = typer.Argument(..., help="Path to training config YAML"),
):
    """
    Validate a training config without running.

    Example:
        complexity train validate config.yaml
    """
    import yaml

    if not config.exists():
        console.print(error(f"Config file not found: {config}"))
        raise typer.Exit(1)

    with open(config) as f:
        cfg = yaml.safe_load(f)

    # Validate required fields
    errors = []
    warnings_list = []

    if 'model' not in cfg:
        errors.append("Missing 'model' section")
    else:
        model = cfg['model']
        if 'hidden_size' not in model:
            errors.append("Missing model.hidden_size")
        if 'num_layers' not in model:
            errors.append("Missing model.num_layers")

    if 'training' not in cfg:
        errors.append("Missing 'training' section")
    else:
        train_cfg = cfg['training']
        if 'batch_size' not in train_cfg:
            warnings_list.append("No batch_size specified, will use default")
        if 'learning_rate' not in train_cfg:
            warnings_list.append("No learning_rate specified, will use default")

    if 'data' not in cfg:
        warnings_list.append("No 'data' section - will need to provide data separately")

    # Print results
    if errors:
        for e in errors:
            console.print(error(e))
        raise typer.Exit(1)

    for w in warnings_list:
        console.print(warning(w))

    console.print(success("Config is valid!"))


@train.command("create-config")
def create_config(
    output: Path = typer.Argument(Path("config.yaml"), help="Output config path"),
    model_size: str = typer.Option("small", "--size", "-s", help="Model size: tiny, small, medium, large, xl"),
    template: str = typer.Option("default", "--template", "-t", help="Config template"),
):
    """
    Create a template training config.

    Example:
        complexity train create-config my_config.yaml --size medium
    """
    import yaml

    # Model presets
    presets = {
        "tiny": {"hidden_size": 256, "num_layers": 6, "num_heads": 4, "ff_mult": 4},
        "small": {"hidden_size": 512, "num_layers": 12, "num_heads": 8, "ff_mult": 4},
        "medium": {"hidden_size": 1024, "num_layers": 24, "num_heads": 16, "ff_mult": 4},
        "large": {"hidden_size": 2048, "num_layers": 32, "num_heads": 32, "ff_mult": 4},
        "xl": {"hidden_size": 4096, "num_layers": 48, "num_heads": 64, "ff_mult": 4},
    }

    if model_size not in presets:
        console.print(error(f"Unknown size: {model_size}. Choose from: {list(presets.keys())}"))
        raise typer.Exit(1)

    preset = presets[model_size]

    config = {
        "model": {
            "name": f"complexity-{model_size}",
            "hidden_size": preset["hidden_size"],
            "num_layers": preset["num_layers"],
            "num_heads": preset["num_heads"],
            "ff_multiplier": preset["ff_mult"],
            "vocab_size": 50304,
            "max_seq_length": 2048,
            "attention_type": "gqa",
            "num_kv_heads": max(1, preset["num_heads"] // 4),
            "rope_theta": 10000,
            "use_sliding_window": False,
        },
        "training": {
            "max_steps": 100000,
            "batch_size": 32,
            "gradient_accumulation_steps": 4,
            "learning_rate": 3e-4,
            "weight_decay": 0.1,
            "warmup_steps": 2000,
            "lr_scheduler": "cosine",
            "min_lr_ratio": 0.1,
            "gradient_clip": 1.0,
            "mixed_precision": "bf16",
        },
        "data": {
            "train_path": "data/train.jsonl",
            "val_path": "data/val.jsonl",
            "seq_length": 2048,
            "num_workers": 4,
        },
        "checkpoint": {
            "save_steps": 1000,
            "save_dir": "checkpoints",
            "keep_last_n": 5,
        },
        "logging": {
            "log_steps": 10,
            "eval_steps": 500,
            "wandb_project": None,
        },
    }

    with open(output, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(success(f"Created config: {output}"))
    console.print(info(f"Model size: {model_size} ({preset['hidden_size']}d, {preset['num_layers']}L)"))
