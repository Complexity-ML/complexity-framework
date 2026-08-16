"""Function-preserving expert expansion for pretrained TR-Hash checkpoints."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch

from ..config import ModelConfig
from ..models import ComplexityModel
from ..tr_hash.routing import expand_route_table_hierarchically
from .token_routed_conversion import (
    _CONVERTIBLE_MLP_TYPES,
    convert_token_routed_state_dict,
)

_ENGINE_ROUTE_KEY = re.compile(
    r"^layers\.(?P<layer>\d+)\.mlp\.engine\.route_table$"
)
_EXPERT_PARAMETER_SUFFIXES = (
    ".engine.expert_gate",
    ".engine.expert_up",
    ".engine.expert_down",
)
_ROUTE_BUFFER_SUFFIXES = (
    ".engine.route_table",
    ".engine.fused_route_codes",
    ".engine.fused_expert_pairs",
)


def expanded_expert_config(
    raw_config: Mapping[str, Any],
    *,
    target_num_experts: int = 8,
) -> ModelConfig:
    """Return an expanded config that preserves the source expert width."""

    patched = dict(raw_config)
    source_num_experts = int(patched.get("num_experts", 1))
    if source_num_experts != 4:
        raise ValueError("expert expansion currently requires exactly 4 source experts")
    if target_num_experts != 8:
        raise ValueError("expert expansion currently targets exactly 8 experts")
    if int(patched.get("top_k", 1)) != 2:
        raise ValueError("function-preserving expert expansion requires top_k=2")
    mlp_type = patched.get("mlp_type", "tr_hash_engine")
    if mlp_type in _CONVERTIBLE_MLP_TYPES:
        patched["mlp_type"] = "tr_hash_engine"
    elif mlp_type not in {"tr_hash_engine", "tr_hash_moe"}:
        raise ValueError("expert expansion requires a TR-Hash MLP checkpoint")

    routed_width = int(patched["intermediate_size"])
    if routed_width % source_num_experts:
        raise ValueError("source routed width must be divisible by its expert count")
    expert_width = routed_width // source_num_experts
    patched["intermediate_size"] = expert_width * target_num_experts
    patched["num_experts"] = target_num_experts
    patched["routing_strategy"] = "token_id_hierarchical_hash"
    patched["route_hash_count"] = 2
    patched["active_num_experts"] = None
    patched["active_expert_width"] = None
    return ModelConfig.from_dict(patched)


def _canonical_state_and_routes(
    state_dict: Mapping[str, torch.Tensor],
    raw_config: Mapping[str, Any],
) -> tuple[dict[str, torch.Tensor], dict[int, torch.Tensor]]:
    if raw_config.get("mlp_type") in _CONVERTIBLE_MLP_TYPES:
        return convert_token_routed_state_dict(dict(state_dict))

    state = dict(state_dict)
    routes: dict[int, torch.Tensor] = {}
    for key in tuple(state):
        match = _ENGINE_ROUTE_KEY.match(key)
        if match:
            routes[int(match.group("layer"))] = state.pop(key)
        elif key.endswith(_ROUTE_BUFFER_SUFFIXES[1:]):
            state.pop(key)
    return state, routes


def convert_checkpoint_to_expanded_experts(
    state_dict: Mapping[str, torch.Tensor],
    raw_config: Mapping[str, Any],
    *,
    target_num_experts: int = 8,
) -> ComplexityModel:
    """Clone four trained experts into eight while preserving model outputs.

    The old route selects an already-trained parent family. A second token-ID
    hash selects one of its identical clones. Only later continued pretraining
    makes the clones diverge and turns the extra stored capacity into useful
    specialization.
    """

    source_num_experts = int(raw_config.get("num_experts", 1))
    config = expanded_expert_config(
        raw_config,
        target_num_experts=target_num_experts,
    )
    canonical, source_routes = _canonical_state_and_routes(state_dict, raw_config)
    expected_layers = set(range(config.num_hidden_layers))
    if set(source_routes) != expected_layers:
        missing = sorted(expected_layers - set(source_routes))
        raise ValueError(f"source checkpoint is missing exact route tables for layers {missing}")

    clone_count = target_num_experts // source_num_experts
    expanded: dict[str, torch.Tensor] = {}
    for key, value in canonical.items():
        if key.endswith(_ROUTE_BUFFER_SUFFIXES):
            continue
        if key.endswith(_EXPERT_PARAMETER_SUFFIXES):
            if value.ndim != 3 or value.size(0) != source_num_experts:
                raise ValueError(f"invalid expert tensor shape for {key}: {tuple(value.shape)}")
            value = torch.cat([value.clone() for _ in range(clone_count)], dim=0)
        expanded[key] = value

    floating_dtype = next(
        (value.dtype for value in expanded.values() if value.is_floating_point()),
        torch.float32,
    )
    model = ComplexityModel(config).to(dtype=floating_dtype)
    missing, unexpected = model.load_state_dict(expanded, strict=False)
    tolerated_missing = (*_ROUTE_BUFFER_SUFFIXES, ".rotary_emb.inv_freq")
    unexplained = [key for key in missing if not key.endswith(tolerated_missing)]
    if unexplained or unexpected:
        raise RuntimeError(
            "expert expansion left mismatched tensors: "
            f"missing={unexplained}, unexpected={list(unexpected)}"
        )

    for layer_index, source_route_table in source_routes.items():
        expanded_routes = expand_route_table_hierarchically(
            source_route_table,
            source_num_experts=source_num_experts,
            target_num_experts=target_num_experts,
            layer_index=layer_index,
        )
        model.layers[layer_index].mlp.engine.load_route_table(expanded_routes)
    return model


def convert_checkpoint_dir_to_expanded_experts(
    checkpoint_dir: str | Path,
    *,
    target_num_experts: int = 8,
) -> ComplexityModel:
    """Load an HF-style checkpoint and return its expanded equivalent."""

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
    return convert_checkpoint_to_expanded_experts(
        state_dict,
        raw_config,
        target_num_experts=target_num_experts,
    )
