"""Read local native/HF SafeTensors checkpoints with explicit MoE topology.

This loader recovers omitted aggregate expert widths from stored tensors.
Only the composed TR-HASH architecture is accepted. The loader does not
download, rewrite exports or initialize a training run.
The frozen SFT runner remains a separate loader with its own input contract.
"""

from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file

from complexity.config import ModelConfig
from complexity.models import ComplexityModel


def normalize_native_checkpoint_config(raw: Mapping, weights: Mapping) -> ModelConfig:
    """Infer shape-backed widths and preserve HF routing aliases, failing closed."""
    if not isinstance(raw, Mapping) or not isinstance(weights, Mapping):
        raise ValueError("checkpoint config and weights must be mappings")
    patched = dict(raw)
    if patched.get("mlp_type") not in {"tr_hash_engine", "tr_hash_moe"} or patched.get("routing_strategy") != "token_id_composed_modulo":
        raise ValueError("Native TR-HASH checkpoints must declare the composed modulo architecture.")
    for alias, canonical in (
        ("num_experts_per_tok", "top_k"),
        ("rms_norm_eps", "norm_eps"),
    ):
        if alias in patched:
            if canonical in patched and patched[canonical] != patched[alias]:
                raise ValueError(f"checkpoint {canonical} and {alias} disagree")
            patched[canonical] = patched.pop(alias)
    widths, shared_widths = set(), set()
    for name, tensor in weights.items():
        if name.endswith(".expert_gate"):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
                raise ValueError(f"invalid expert tensor layout: {name}")
            if missing := {"hidden_size", "num_experts"} - patched.keys():
                raise ValueError(f"expert topology requires explicit fields: {sorted(missing)}")
            experts, hidden, width = tensor.shape
            if experts != patched["num_experts"] or hidden != patched["hidden_size"]:
                raise ValueError(f"expert tensor differs from checkpoint topology: {name}")
            widths.add(experts * width)
        elif name.endswith(".shared_gate.weight"):
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 2:
                raise ValueError(f"invalid shared expert tensor layout: {name}")
            if "hidden_size" not in patched:
                raise ValueError("shared expert topology requires explicit hidden_size")
            if tensor.shape[1] != patched["hidden_size"]:
                raise ValueError(f"shared expert hidden width differs: {name}")
            shared_widths.add(tensor.shape[0])
    for key, observed in (
        ("intermediate_size", widths),
        ("shared_intermediate_size", shared_widths),
    ):
        if len(observed) > 1:
            raise ValueError(f"checkpoint layers disagree on {key}")
        if observed:
            inferred = next(iter(observed))
            if patched.get(key) is not None and patched[key] != inferred:
                raise ValueError(f"checkpoint {key} disagrees with stored expert tensors")
            patched[key] = inferred
    # Native configuration is strict. Ordinary HF export metadata is excluded
    # from the architecture only when it is a known serialization field.
    hf_metadata = {
        "architectures",
        "model_type",
        "torch_dtype",
        "transformers_version",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "auto_map",
    }
    known = {field.name for field in fields(ModelConfig)}
    if unknown := patched.keys() - known - hf_metadata:
        raise ValueError(f"unknown checkpoint architecture fields: {sorted(unknown)}")
    return ModelConfig.from_dict(
        {key: value for key, value in patched.items() if key in known}, strict=True
    )


def load_native_checkpoint(path: str | Path) -> ComplexityModel:
    """Load complete local SafeTensors on CPU; preserve every trained route."""
    path = Path(path)
    raw = json.loads((path / "config.json").read_text(encoding="utf-8"))
    weights = load_file(str(path / "model.safetensors"), device="cpu")
    config = normalize_native_checkpoint_config(raw, weights)
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("native checkpoint loading requires the default dtype float32")
    del weights
    with torch.device("cpu"):
        model = ComplexityModel(config)
    model._load_pretrained_weights(path / "model.safetensors")
    return model.float()
