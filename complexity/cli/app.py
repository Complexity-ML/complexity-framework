"""
INL Complexity Framework CLI - Main Application Entry Point.

Usage:
    complexity train run config.yaml
    complexity inference generate model.pt --prompt "Hello"
    complexity inference chat model.pt
    complexity tokenize encode "Hello world"
    complexity profile model --hidden 2048 --layers 24
    complexity convert to-safetensors model.pt
    complexity serve start model.pt --port 8000
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
    from .commands import train, inference, tokenize, profile, convert, serve, info

    # Create main app
    app = typer.Typer(
        name="complexity",
        help="INL Complexity Framework - Deep Learning Training & Inference",
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

    @app.callback()
    def main_callback(
        version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    ):
        """
        INL Complexity Framework CLI.

        A comprehensive deep learning framework for training and deploying
        large language models with advanced attention mechanisms.
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
        from .utils import console, print_markdown

        guide = """
# INL Complexity Framework - Quickstart

## Installation
```bash
pip install complexity
# or with all extras
pip install complexity[all]
```

## Training a Model

1. Create a config:
```bash
complexity train create-config my_config.yaml --size small
```

2. Start training:
```bash
complexity train run my_config.yaml --gpus 1
```

## Inference

Generate text:
```bash
complexity inference generate model.pt --prompt "Hello, world"
```

Interactive chat:
```bash
complexity inference chat model.pt --system "You are a helpful assistant"
```

## Serving

Start an API server:
```bash
complexity serve start model.pt --port 8000
```

## Tokenization

Encode text:
```bash
complexity tokenize encode "Hello, world!" --show
```

## Model Conversion

To safetensors:
```bash
complexity convert to-safetensors model.pt
```

From HuggingFace:
```bash
complexity convert from-hf meta-llama/Llama-2-7b ./llama2
```

## Profiling

Profile model:
```bash
complexity profile model --hidden 2048 --layers 24 --heads 16
```

## More Info

```bash
complexity info version      # Show version & deps
complexity info model model.pt  # Inspect model
complexity info tokens       # Show special tokens
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
            import torch
            import numpy as np

            # Import framework components
            try:
                import complexity
                from complexity.core import Transformer, TransformerConfig
                from complexity.data import ComplexityTokenizer, ComplexityTemplate
                from complexity.training import Trainer, TrainingConfig
                has_complexity = True
            except ImportError:
                complexity = None
                has_complexity = False

            banner = """
INL Complexity Interactive Shell
================================
Available:
  - torch, np (numpy)
  - complexity module
  - Transformer, TransformerConfig
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
                local_vars.update({
                    "Transformer": Transformer,
                    "TransformerConfig": TransformerConfig,
                    "ComplexityTokenizer": ComplexityTokenizer,
                    "ComplexityTemplate": ComplexityTemplate,
                    "Trainer": Trainer,
                    "TrainingConfig": TrainingConfig,
                })

            code.interact(banner=banner, local=local_vars)

        except Exception as e:
            console.print(f"Shell error: {e}")

else:
    app = None


def main():
    """Entry point for the CLI."""
    if app is None:
        print("=" * 50)
        print("INL Complexity CLI")
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
