"""Checkpoint inspection and explicit model-only provenance collection."""

import json
from pathlib import Path
from typing import Optional

import typer

from ..inspection import inspect_checkpoint
from ..utils import console, error, print_panel, warning

checkpoint = typer.Typer(
    name="checkpoint",
    help="Inspect checkpoint manifests or collect local model-only artifacts",
    no_args_is_help=True,
)


@checkpoint.command("collect")
def collect(
    config: Path = typer.Argument(..., help="Explicit local artifact-collection JSON config"),
):
    """Collect weights and selected provenance into a new local directory.

    This explicitly loads model tensors; inspect remains read-only and never
    deserializes them. Collection does not publish or restore training state.
    """
    from complexity.export.provenance import collect_from_config

    try:
        manifest = collect_from_config(config)
    except (ValueError, TypeError, RuntimeError, OSError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(2) from exc
    typer.echo(str(manifest))


@checkpoint.command("inspect")
def inspect(
    path: Path = typer.Argument(..., help="TR-HASH FSDP checkpoint directory"),
    verify: bool = typer.Option(
        False, "--verify", help="Hash every manifest-listed file (can read many GB)"
    ),
    recipe: Optional[Path] = typer.Option(
        None, "--recipe", help="Original recipe.json or recipe directory"
    ),
    json_output: bool = typer.Option(False, "--json", help="Print machine-readable results"),
):
    """Inspect progress, file completeness, and optionally SHA-256 identities.

    This command never deserializes pickle metadata, model tensors, optimizer
    state or RNG files. Hash verification alone does not prove successful resume.
    """
    try:
        report = inspect_checkpoint(path, verify=verify, recipe_path=recipe)
    except (ValueError, TypeError, OSError) as exc:
        if json_output:
            typer.echo(json.dumps({"error": str(exc), "path": str(path)}), err=True)
        else:
            console.print(error(str(exc)))
        raise typer.Exit(1) from exc
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
        return
    print_panel(
        f"Checkpoint: {path}\n"
        f"Next optimizer update: {report['next_update']:,}\n"
        f"Tokens seen: {report['tokens_seen']:,}\n"
        f"World size: {report['world_size']}\n"
        f"Files: {report['manifest_files']} ({report['manifest_bytes']:,} bytes)\n"
        f"Checked: {report['verification_scope']}\n"
        f"Recipe match: {report['recipe_matches']}\n"
        "Restored: no",
        title="Checkpoint inspection",
        style="cyan",
    )
    for item in report["warnings"]:
        console.print(warning(item))
