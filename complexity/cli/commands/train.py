"""Training configuration tools and explicit boundaries around execution."""

import json
from pathlib import Path
from typing import Optional

import typer

from ..inspection import inspect_config, inspect_recipe, load_mapping
from ..utils import console, error, info, print_panel, success, warning

train = typer.Typer(
    name="train",
    help="Inspect plans and run explicit text training or SFT workflows",
    no_args_is_help=True,
)


@train.command("demo")
def local_training_demo(
    output: Path = typer.Option(..., "--output", "-o", help="New or empty artifact directory"),
    device: str = typer.Option("cpu", "--device", help="Explicit local device: cpu or mps"),
):
    """Train a tiny synthetic TR-HASH, restore, evaluate and verify native export."""
    import sys
    from contextlib import redirect_stdout

    try:
        with redirect_stdout(sys.stderr):
            from complexity.training.demo import run_training_demo

            report = run_training_demo(output, device=device)
    except ImportError as exc:
        typer.echo(
            json.dumps(
                {
                    "error": "Local training requires PyTorch and the core dependencies; install the CPU or MPS Torch backend first",
                    "dependency": getattr(exc, "name", None),
                }
            ),
            err=True,
        )
        raise typer.Exit(2) from exc
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


@train.command("lineage")
def text_lineage(
    config: Path = typer.Argument(..., help="Audited text pretraining/refinement JSON"),
    run: bool = typer.Option(
        False, "--run", help="Execute the declared stage with the native runner"
    ),
):
    """Preflight a bounded text lineage; execution and torchrun are explicit."""
    # Optional backend diagnostics belong on stderr; stdout remains JSON.
    import sys
    from contextlib import redirect_stdout

    try:
        with redirect_stdout(sys.stderr):
            from complexity.training.lineage_runner import execute_lineage_config

            report = execute_lineage_config(config, run=run)
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@train.command("sft")
def supervised_finetuning(
    config: Path = typer.Argument(..., help="Local SFT experiment JSON"),
    run: bool = typer.Option(False, "--run", help="Explicitly execute the reviewed experiment"),
    steps: Optional[int] = typer.Option(
        None,
        "--steps",
        min=1,
        help="Stop at this absolute optimizer update; keep the full schedule",
    ),
    resume: Optional[Path] = typer.Option(
        None, "--resume", help="Restore a local step directory or latest sentinel"
    ),
    distributed: bool = typer.Option(False, "--distributed", help="Use native DDP under torchrun"),
    migrate_single: bool = typer.Option(
        False,
        "--migrate-single",
        help="Explicitly migrate a portable single-device checkpoint to DDP",
    ),
):
    """Preflight an explicitly marked SFT experiment; execution is opt-in."""
    from complexity.training.sft import execute_sft_config

    try:
        if distributed:
            from complexity.training.sft_distributed import execute_distributed_sft_config

            report = execute_distributed_sft_config(
                config, run=run, steps=steps, resume=resume, migrate_single=migrate_single
            )
        else:
            if migrate_single:
                raise ValueError("--migrate-single requires --distributed --run")
            report = execute_sft_config(config, run=run, steps=steps, resume=resume)
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(2) from exc
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


@train.command("stack")
def staged_supervised_finetuning(
    config: Path = typer.Argument(..., help="File-backed LoRA curriculum JSON or YAML"),
    run: bool = typer.Option(False, "--run", help="Execute the declared SFT curriculum"),
    start_stage: Optional[str] = typer.Option(
        None, "--start-stage", help="Continue at this curriculum stage"
    ),
    stop_after: Optional[str] = typer.Option(
        None, "--stop-after", help="Stop after this curriculum stage"
    ),
):
    """Inspect or run a native LoRA curriculum with explicit source_stage."""
    from complexity.training.stack_sft import StackSFTRunner

    try:
        runner = StackSFTRunner.from_yaml(config, dry_run=not run)
        checkpoint = runner.run(start_stage=start_stage, stop_after=stop_after)
    except (ValueError, TypeError, RuntimeError, OSError, KeyError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(2) from exc
    if run:
        typer.echo(
            json.dumps(
                {"checkpoint": checkpoint, "training_started": True, "release_eligible": False}
            )
        )


def _report(report: dict, json_output: bool) -> None:
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    if report["format"] == "fsdp_recipe":
        summary = (
            f"Recipe: {report['path']}\n"
            f"World size: {report['world_size']} | microbatch/GPU: {report['microbatch']}\n"
            f"Context: {report['context']:,}\n"
            f"Optimizer updates: {report['max_updates']:,}\n"
            f"Tokens/update: {report['tokens_per_update']:,}\n"
            f"Scheduled tokens: {report['scheduled_tokens']:,}\n"
            f"Recipe SHA-256: {report['recipe_sha256']}"
        )
    else:
        model = report["model"]
        summary = (
            f"Config: {report['path']}\n"
            f"Hidden size: {model['hidden_size']}\n"
            f"Layers: {model['num_hidden_layers']}\n"
            f"Attention heads: {model['num_attention_heads']}\n"
            f"MLP: {model['mlp_type']}\n"
            f"Context: {model['max_position_embeddings']}"
        )
    print_panel(summary, title="Configuration inspection", style="cyan")
    for item in report["warnings"]:
        console.print(warning(item))
    console.print(info("No training started"))


def _read_config(config: Path, json_output: bool = False) -> dict:
    try:
        return inspect_config(config)
    except (ValueError, TypeError, OSError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "path": str(config)}), err=True)
        else:
            console.print(error(str(exc)))
        raise typer.Exit(1) from exc


@train.command("run")
def run_training(
    config: Path = typer.Argument(..., help="Model/training config JSON or YAML"),
    checkpoint: Optional[Path] = typer.Option(
        None, "--checkpoint", "-c", help="Original checkpoint (script-owned)"
    ),
    output_dir: Path = typer.Option(
        Path("outputs"), "--output", "-o", help="Output directory (script-owned)"
    ),
    num_gpus: int = typer.Option(1, "--gpus", "-g", min=1, help="Number of GPUs"),
    distributed: bool = typer.Option(
        False, "--distributed", "-d", help="Distributed execution (script-owned)"
    ),
    mixed_precision: str = typer.Option(
        "bf16", "--precision", "-p", help="Precision: fp32, fp16, bf16"
    ),
    wandb_project: Optional[str] = typer.Option(None, "--wandb", help="W&B project (script-owned)"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate config without constructing a model or training"
    ),
):
    """Validate a config; execution requires its dataloader-owning training script.

    The historical command remains available. Its dry run uses the same strict
    validation as ``train validate`` and never selects or initializes a GPU.
    """
    if mixed_precision not in {"fp32", "fp16", "bf16"}:
        console.print(error("Invalid precision. Choose one of: fp32, fp16, bf16"))
        raise typer.Exit(1)
    report = _read_config(config)
    _report(report, False)
    if dry_run:
        return
    console.print(
        error(
            "Generic CLI execution cannot infer a dataset/dataloader. Use the training "
            "script that owns this experiment or complexity.training.Trainer with explicit "
            "dataloaders. For the published 1B run, inspect scripts/pretrain_1b_fsdp/recipe.json "
            "and follow scripts/pretrain_1b_fsdp/TRAINING.md."
        )
    )
    raise typer.Exit(2)


@train.command("resume")
def resume_training(
    checkpoint_dir: Path = typer.Argument(..., help="Checkpoint directory to resume from"),
    steps: Optional[int] = typer.Option(
        None, "--steps", "-s", min=1, help="Additional steps (script-owned)"
    ),
):
    """Explain the original-script requirement for compatible resume."""
    if not checkpoint_dir.is_dir():
        console.print(error(f"Checkpoint directory not found: {checkpoint_dir}"))
        raise typer.Exit(1)
    console.print(
        error(
            "Resume requires the original model, dataloader, optimizer and RNG restore "
            "code. Use the training script that created this checkpoint. For a TR-HASH "
            "1B FSDP checkpoint, first use complexity checkpoint inspect PATH --verify "
            "--recipe scripts/pretrain_1b_fsdp/recipe.json, then follow its TRAINING.md."
        )
    )
    raise typer.Exit(2)


@train.command("validate")
def validate_config(
    config: Path = typer.Argument(..., help="Model config or model/training template (JSON/YAML)"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable results"),
):
    """Validate config fields and dimensions without constructing a model."""
    report = _read_config(config, json_output)
    _report(report, json_output)
    if not json_output:
        console.print(success("Configuration fields are valid"))


@train.command("inspect")
def inspect_training(
    config: Path = typer.Argument(..., help="Model config, FSDP recipe.json, or recipe directory"),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable results"),
):
    """Inspect architecture or FSDP token budget and linked asset identities."""
    try:
        path = config / "recipe.json" if config.is_dir() else config
        data = load_mapping(path)
        report = inspect_recipe(path) if "max_updates" in data else inspect_config(path)
    except (ValueError, TypeError, OSError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "path": str(config)}), err=True)
        else:
            console.print(error(str(exc)))
        raise typer.Exit(1) from exc
    _report(report, json_output)


@train.command("create-config")
def create_config(
    output: Path = typer.Argument(Path("config.yaml"), help="Output config JSON/YAML path"),
    model_size: str = typer.Option(
        "small", "--size", "-s", help="Model size: tiny, small, medium, large, xl"
    ),
    template: str = typer.Option("default", "--template", "-t", help="Template: default"),
    force: bool = typer.Option(False, "--force", help="Replace an existing configuration"),
):
    """Create a canonical model/training template for an explicit Trainer.

    This template is separate from the pinned TR-HASH 1B pretraining recipe.
    """
    import yaml

    presets = {
        "tiny": (256, 6, 4),
        "small": (512, 12, 4),
        "medium": (1024, 24, 4),
        "large": (2048, 32, 4),
        "xl": (4096, 48, 4),
    }
    if template != "default" or model_size not in presets:
        console.print(
            error("Choose --template default and --size tiny, small, medium, large, or xl")
        )
        raise typer.Exit(1)
    hidden, layers, heads = presets[model_size]
    config = {
        "model": {
            "hidden_size": hidden,
            "num_hidden_layers": layers,
            "num_attention_heads": heads,
            "num_key_value_heads": 1,
            "intermediate_size": hidden,
            "shared_intermediate_size": hidden * 2,
            "vocab_size": 50304,
            "max_position_embeddings": 2048,
            "attention_type": "gqa",
            "mlp_type": "tr_hash_engine",
            "num_experts": 4,
            "top_k": 2,
            "routing_strategy": "token_id_composed_modulo",
            "top_k_primary_weight": 0.5,
            "shared_output_scale": 1.0,
            "routed_output_scale": 2.0,
            "use_qk_norm": True,
            "rope_theta": 10000.0,
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
            "grad_clip": 1.0,
            "precision": "bf16",
            "checkpoint_dir": "checkpoints",
            "save_steps": 1000,
            "save_total_limit": 5,
            "log_steps": 10,
            "eval_steps": 500,
        },
        "data": {
            "train_path": "data/train.jsonl",
            "val_path": "data/val.jsonl",
            "seq_length": 2048,
        },
    }
    try:
        with output.open("w" if force else "x", encoding="utf-8") as stream:
            if output.suffix.lower() == ".json":
                json.dump(config, stream, indent=2)
                stream.write("\n")
            else:
                yaml.safe_dump(config, stream, sort_keys=False)
    except FileExistsError as exc:
        console.print(
            error(f"Config already exists: {output}. Choose another path or use --force.")
        )
        raise typer.Exit(1) from exc
    except OSError as exc:
        console.print(error(f"Cannot create config {output}: {exc}"))
        raise typer.Exit(1) from exc
    console.print(success(f"Created config: {output}"))
    console.print(info(f"Validate with: complexity train validate {output}"))
    console.print(
        warning(
            "Training requires explicit dataloaders. This does not replace the pinned 1B recipe."
        )
    )
