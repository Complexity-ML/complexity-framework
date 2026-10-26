"""One command tree for visual models and operations."""

import typer

from complexity.vision.workflows import WORKFLOWS, run_workflow

vision = typer.Typer(help="Detection, images, video and visual-language workflows.", no_args_is_help=True)
_CONTEXT = {"allow_extra_args": True, "ignore_unknown_options": True}


@vision.command("list")
def list_workflows():
    """List operations; runtime help is available with TASK -- --help."""
    for operation, tasks in WORKFLOWS.items():
        for task in tasks:
            typer.echo(f"complexity vision {operation} {task} -- --help")


def _dispatch(ctx, operation, task):
    try:
        run_workflow(operation, task, list(ctx.args))
    except ImportError as exc:
        typer.echo(f"Missing vision dependency: {exc}. Install the appropriate vision profile.", err=True)
        raise typer.Exit(1) from exc
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(1) from exc


@vision.command("train", context_settings=_CONTEXT)
def train(ctx: typer.Context, task: str = typer.Argument(..., help="detector, image, image-edit, image-text")):
    """Train a selected visual model with its existing runtime options."""
    _dispatch(ctx, "train", task)


@vision.command("pretrain", context_settings=_CONTEXT)
def pretrain(ctx: typer.Context, task: str = typer.Argument("backbone")):
    """Pretrain the visual backbone."""
    _dispatch(ctx, "pretrain", task)


@vision.command("evaluate", context_settings=_CONTEXT)
def evaluate(ctx: typer.Context, task: str = typer.Argument("detector")):
    """Evaluate native or ONNX detectors with the COCO protocol."""
    _dispatch(ctx, "evaluate", task)


@vision.command("export", context_settings=_CONTEXT)
def export(ctx: typer.Context, task: str = typer.Argument("detector")):
    """Export detector predictions to ONNX or build a TensorRT engine."""
    _dispatch(ctx, "export", task)


@vision.command("check", context_settings=_CONTEXT)
def check(ctx: typer.Context, task: str = typer.Argument(...)):
    """Validate COCO reports or native/quantized ONNX parity."""
    _dispatch(ctx, "check", task)


@vision.command("merge", context_settings=_CONTEXT)
def merge(ctx: typer.Context, task: str = typer.Argument("coco")):
    """Merge COCO branch and precision reports."""
    _dispatch(ctx, "merge", task)


@vision.command("infer", context_settings=_CONTEXT)
def infer(ctx: typer.Context, task: str = typer.Argument("onnx")):
    """Run ONNX detector inference."""
    _dispatch(ctx, "infer", task)


@vision.command("benchmark", context_settings=_CONTEXT)
def benchmark(ctx: typer.Context, task: str = typer.Argument("onnx")):
    """Measure an ONNX artifact with an explicit provider."""
    _dispatch(ctx, "benchmark", task)


@vision.command("quantize", context_settings=_CONTEXT)
def quantize(ctx: typer.Context, task: str = typer.Argument("onnx")):
    """Prepare FP16 or calibrated INT8 ONNX artifacts."""
    _dispatch(ctx, "quantize", task)


@vision.command("release", context_settings=_CONTEXT)
def release(ctx: typer.Context, task: str = typer.Argument("onnx")):
    """Build or verify pinned ONNX artifacts and their manifest."""
    _dispatch(ctx, "release", task)


@vision.command("generate", context_settings=_CONTEXT)
def generate(ctx: typer.Context, task: str = typer.Argument("image")):
    """Generate images from a trained visual checkpoint."""
    _dispatch(ctx, "generate", task)


@vision.command("edit", context_settings=_CONTEXT)
def edit(ctx: typer.Context, task: str = typer.Argument("image")):
    """Edit images from a trained image-editor checkpoint."""
    _dispatch(ctx, "edit", task)


@vision.command("serve", context_settings=_CONTEXT)
def serve(ctx: typer.Context, task: str = typer.Argument("detector")):
    """Serve a detector using the optional HTTP runtime."""
    _dispatch(ctx, "serve", task)
