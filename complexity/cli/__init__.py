"""
Complexity Framework CLI.

Provides command-line interface for:
- Inspecting model and training configurations
- Inspecting checkpoints without restoring them
- Using external vLLM/SGLang text inference
- Tokenization
- Profiling
- Model conversion
- Vision training, export and serving workflows

Usage:
    complexity --help
    complexity train validate config.json
    complexity inference chat served-model
    complexity vision list
"""

from .app import app, main


def __getattr__(name):
    if name == "ComplexityCLI":
        from .core import ComplexityCLI

        globals()[name] = ComplexityCLI
        return ComplexityCLI
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "app",
    "main",
    "ComplexityCLI",
]
