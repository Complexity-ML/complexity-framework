"""Zero-shot choice evaluation and comparisons, with optional runtimes lazy."""

import json

import typer

evaluate = typer.Typer(
    name="eval",
    help="Score zero-shot choices or compare two local native checkpoints",
    no_args_is_help=True,
)

# Forward the existing argparse options verbatim. In particular, --help is
# handled by the lightweight evaluation module before any Torch/MLX import.
_FORWARD_OPTIONS = {"allow_extra_args": True, "ignore_unknown_options": True}


@evaluate.command("choices", context_settings=_FORWARD_OPTIONS, add_help_option=False)
def choices(ctx: typer.Context):
    """Score candidate continuations for a local native Torch or MLX bundle."""
    from complexity.evaluation.zero_shot import main

    _invoke(main, ctx.args)


@evaluate.command("compare", context_settings=_FORWARD_OPTIONS, add_help_option=False)
def compare(ctx: typer.Context):
    """Compare exactly two native bundles with the same data and tokenizer."""
    from complexity.evaluation.zero_shot import compare_main

    _invoke(compare_main, ctx.args)


def _invoke(entrypoint, args):
    try:
        entrypoint(args)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as exc:
        typer.echo(json.dumps({"error": str(exc)}), err=True)
        raise typer.Exit(2) from exc
