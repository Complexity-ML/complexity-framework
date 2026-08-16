"""Create a multi-hash TR-Hash model without mutating its base checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import ModelConfig
from ..models import ComplexityModel
from .token_routed_conversion import (
    _CONVERTIBLE_MLP_TYPES,
    convert_token_routed_state_dict,
)

_ROUTE_BUFFER_SUFFIXES = (
    ".engine.route_table",
    ".engine.fused_route_codes",
    ".engine.fused_expert_pairs",
)


def multi_hash_config(
    raw_config: Mapping[str, Any],
    *,
    route_hash_count: int = 2,
) -> ModelConfig:
    """Return a separate four-expert/top-2 multi-hash model config."""

    patched = dict(raw_config)
    if patched.get("mlp_type") in _CONVERTIBLE_MLP_TYPES:
        patched["mlp_type"] = "tr_hash_engine"
    if patched.get("mlp_type", "tr_hash_engine") not in {
        "tr_hash_engine",
        "tr_hash_moe",
    }:
        raise ValueError("multi-hash conversion requires a TR-Hash MLP checkpoint")
    if int(patched.get("num_experts", 1)) != 4:
        raise ValueError("the current multi-hash text recipe requires exactly 4 experts")
    if int(patched.get("top_k", 1)) != 2:
        raise ValueError("the current multi-hash text recipe requires top_k=2")
    patched["routing_strategy"] = "token_id_multi_hash"
    patched["route_hash_count"] = int(route_hash_count)
    return ModelConfig.from_dict(patched)


def convert_checkpoint_to_multi_hash(
    state_dict: Mapping[str, torch.Tensor],
    raw_config: Mapping[str, Any],
    *,
    route_hash_count: int = 2,
) -> ComplexityModel:
    """Copy trained tensors while intentionally regenerating route buffers.

    Expert parameters are transferred exactly. The source routing table is
    excluded because loading it would silently disable the requested routing
    upgrade. Continued pretraining is required afterwards so the experts can
    adapt to their new token assignments.
    """

    config = multi_hash_config(raw_config, route_hash_count=route_hash_count)
    if raw_config.get("mlp_type") in _CONVERTIBLE_MLP_TYPES:
        converted, _ = convert_token_routed_state_dict(dict(state_dict))
    else:
        converted = dict(state_dict)
    transferable = {
        key: value
        for key, value in converted.items()
        if not key.endswith(_ROUTE_BUFFER_SUFFIXES)
    }
    floating_dtype = next(
        (value.dtype for value in transferable.values() if value.is_floating_point()),
        torch.float32,
    )
    model = ComplexityModel(config).to(dtype=floating_dtype)
    missing, unexpected = model.load_state_dict(transferable, strict=False)
    tolerated_missing = (*_ROUTE_BUFFER_SUFFIXES, ".rotary_emb.inv_freq")
    unexplained = [
        key for key in missing if not key.endswith(tolerated_missing)
    ]
    if unexplained or unexpected:
        raise RuntimeError(
            "multi-hash conversion left mismatched tensors: "
            f"missing={unexplained}, unexpected={list(unexpected)}"
        )
    return model


def convert_checkpoint_dir_to_multi_hash(
    checkpoint_dir: str | Path,
    *,
    route_hash_count: int = 2,
) -> ComplexityModel:
    """Load an HF-style source directory and return a multi-hash model."""

    checkpoint_dir = Path(checkpoint_dir)
    raw_config = json.loads((checkpoint_dir / "config.json").read_text())
    safetensors_path = checkpoint_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state_dict = load_file(str(safetensors_path), device="cpu")
    else:
        state_dict = torch.load(
            checkpoint_dir / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
        )
    return convert_checkpoint_to_multi_hash(
        state_dict,
        raw_config,
        route_hash_count=route_hash_count,
    )
