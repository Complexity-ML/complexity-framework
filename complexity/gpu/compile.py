"""
torch.compile configuration and utilities.

Complexity-ML — 2026
"""

import torch
import logging
import os
import shutil

logger = logging.getLogger(__name__)


def compile_model(
    model: torch.nn.Module,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
    disable: bool = False,
) -> torch.nn.Module:
    """
    Compile model with torch.compile, with proper cache management.

    Args:
        model: Model to compile
        mode: "default", "reduce-overhead" (CUDA graphs), "max-autotune"
        fullgraph: Require full graph capture (fails if graph break)
        dynamic: Allow dynamic shapes (slower compile, flexible batch)
        disable: Skip compilation entirely

    Returns:
        Compiled model (or original if compile unavailable/disabled)
    """
    if disable or not hasattr(torch, 'compile'):
        return model

    # Clear inductor cache to avoid stale compiled kernels
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", "/tmp/torchinductor_root")
    if os.path.isdir(cache_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        logger.info(f"[gpu] Cleared inductor cache: {cache_dir}")

    try:
        compiled = torch.compile(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )
        logger.info(f"[gpu] torch.compile enabled (mode={mode}, fullgraph={fullgraph})")
        return compiled
    except Exception as e:
        logger.warning(f"[gpu] torch.compile failed: {e}")
        return model


def set_compile_env(
    max_autotune: bool = False,
    coordinate_descent_tuning: bool = False,
) -> None:
    """
    Set environment variables for torch.compile/inductor optimization.

    Args:
        max_autotune: Enable exhaustive kernel search (slower compile, faster runtime)
        coordinate_descent_tuning: Fine-tune tile sizes after initial selection
    """
    if max_autotune:
        os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
    if coordinate_descent_tuning:
        os.environ["TORCHINDUCTOR_COORDINATE_DESCENT_TUNING"] = "1"
