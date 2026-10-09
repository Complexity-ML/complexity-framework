"""
Complexity Framework CLI - Main Application Entry Point.

Usage:
    complexity train validate configs/examples/tr_gqa_cpu.json
    complexity train inspect scripts/pretrain_1b_fsdp/recipe.json
    complexity checkpoint inspect /path/to/checkpoint --verify
    complexity inference generate served-model --prompt "Hello"
    complexity inference chat served-model
    complexity tokenize encode "Hello world"
    complexity profile model config.json
    complexity vision list
    complexity jobs list
    complexity info version
"""

import sys

try:
    import typer

    HAS_TYPER = True
except ImportError:
    HAS_TYPER = False
    typer = None


if HAS_TYPER:
    from .commands import (
        checkpoint,
        convert,
        data,
        evaluate,
        inference,
        info,
        jobs,
        profile,
        serve,
        tokenize,
        train,
        vision,
    )

    # Create main app
    app = typer.Typer(
        name="complexity",
        help="Complexity Framework - TR-Hash language and vision model research",
        add_completion=False,
        no_args_is_help=True,
    )

    # Register command groups
    app.add_typer(train, name="train")
    app.add_typer(inference, name="inference")
    app.add_typer(tokenize, name="tokenize")
    app.add_typer(profile, name="profile")
    app.add_typer(convert, name="convert")
    app.add_typer(serve, name="serve")
    app.add_typer(info, name="info")
    app.add_typer(jobs, name="jobs")
    app.add_typer(checkpoint, name="checkpoint")
    app.add_typer(vision, name="vision")
    app.add_typer(data, name="data")
    app.add_typer(evaluate, name="eval")

    @app.callback(invoke_without_command=True)
    def main_callback(
        version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    ):
        """
        Complexity Framework CLI.

        Inspect configurations and checkpoints, build explicit training
        workflows, and use external text-serving runtimes.
        """
        if version:
            try:
                from complexity import __version__

                typer.echo(f"complexity {__version__}")
            except ImportError:
                typer.echo("complexity dev")
            raise typer.Exit()

    @app.command("quickstart")
    def quickstart():
        """
        Show quickstart guide.
        """
        from .utils import print_markdown

        guide = """
# Complexity Framework - Quickstart

Run these commands from the repository root. Install PyTorch separately for the
intended backend; the helper installs the framework afterwards:

```bash
python3 -m venv .venv
source .venv/bin/activate
./scripts/install_backend.sh cpu
complexity --version
```

Use `cuda` or `rocm` instead of `cpu` on the corresponding training host.

## Validate an architecture without training

```bash
complexity train validate configs/examples/tr_gqa_cpu.json
complexity train inspect configs/examples/tr_gqa_cpu.json --json
```

Create a separate Trainer template if you need one:

```bash
complexity train create-config my_config.yaml --size tiny
complexity train validate my_config.yaml
```

Templates need explicit dataloaders through `complexity.training.Trainer`.
Generic `train run` and `train resume` cannot reconstruct those dataloaders;
`train run CONFIG --dry-run` validates only.

## Inspect the pinned TR-HASH 1B pretraining recipe

```bash
complexity train inspect scripts/pretrain_1b_fsdp/recipe.json
complexity checkpoint inspect /path/to/step_0051000 --verify \
  --recipe scripts/pretrain_1b_fsdp/recipe.json
```

Inspection reads local files only. Follow `scripts/pretrain_1b_fsdp/TRAINING.md`
for original-script resume and evaluation. The pinned recipe stays separate from
new experiments. File integrity alone does not establish successful restore.

## Generate through an existing serving endpoint

```bash
complexity inference generate my-served-model --backend vllm \
  --base-url http://localhost:8000 --prompt "Hello"
complexity inference chat my-served-model --backend vllm \
  --base-url http://localhost:8000
```

vLLM/SGLang runs separately; these commands send requests to its existing server.
A training checkpoint must be exported for the serving backend before use.

## Discover other tools

```bash
complexity --help
complexity tokenize --help
complexity jobs --help
complexity info version
```
"""
        print_markdown(guide)

    @app.command("shell")
    def interactive_shell():
        """
        Start an interactive Python shell with framework loaded.
        """
        from .utils import console, info

        console.print(info("Starting interactive shell..."))
        console.print("Available imports: complexity.*, torch, numpy")

        try:
            import code

            import numpy as np
            import torch

            # Import framework components
            try:
                import complexity
                from complexity.config import ModelConfig
                from complexity.data import ComplexityTemplate, ComplexityTokenizer
                from complexity.models import ComplexityModel
                from complexity.training import Trainer, TrainingConfig

                has_complexity = True
            except ImportError:
                complexity = None
                has_complexity = False

            banner = """
Complexity Interactive Shell
============================
Available:
  - torch, np (numpy)
  - complexity module
  - ComplexityModel, ModelConfig
  - ComplexityTokenizer, ComplexityTemplate
  - Trainer, TrainingConfig

Type 'exit()' or Ctrl+D to quit.
"""
            local_vars = {
                "torch": torch,
                "np": np,
                "complexity": complexity,
            }

            if has_complexity:
                local_vars.update(
                    {
                        "ComplexityModel": ComplexityModel,
                        "ModelConfig": ModelConfig,
                        "ComplexityTokenizer": ComplexityTokenizer,
                        "ComplexityTemplate": ComplexityTemplate,
                        "Trainer": Trainer,
                        "TrainingConfig": TrainingConfig,
                    }
                )

            code.interact(banner=banner, local=local_vars)

        except Exception as e:
            console.print(f"Shell error: {e}")

else:
    app = None


def main():
    """Entry point for the CLI."""
    if app is None:
        print("=" * 50)
        print("Complexity CLI")
        print("=" * 50)
        print()
        print("Error: typer not installed.")
        print()
        print("Install with:")
        print("  pip install typer[all] rich")
        print()
        print("Or use the Python API directly:")
        print("  from complexity.cli import ComplexityCLI")
        print("  cli = ComplexityCLI()")
        print()
        sys.exit(1)

    app()


if __name__ == "__main__":
    main()
