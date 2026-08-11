"""Small, framework-native LoRA adapters for ``torch.nn.Linear`` modules.

Checkpoints keep a canonical, merged model state for normal inference and a
separate adapter state for exact training resume.  This avoids making the
runtime or exported model depend on LoRA wrapper classes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int
    alpha: float
    dropout: float
    targets: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "targets": list(self.targets),
        }


class LoRALinear(nn.Module):
    """Low-rank residual around an existing frozen linear layer."""

    is_lora_adapter = True

    def __init__(self, base: nn.Linear, *, rank: int, alpha: float, dropout: float):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout else nn.Identity()
        self.lora_A = nn.Parameter(
            torch.empty(
                self.rank,
                base.in_features,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                base.out_features,
                self.rank,
                device=base.weight.device,
                dtype=base.weight.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        self.base.requires_grad_(False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base_output = self.base(inputs)
        low_rank = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return base_output + low_rank * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


def _split_parent(model: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parent_name, _, child_name = module_name.rpartition(".")
    parent = model.get_submodule(parent_name) if parent_name else model
    return parent, child_name


def apply_lora(
    model: nn.Module,
    *,
    rank: int,
    alpha: float,
    dropout: float,
    targets: Iterable[str],
) -> dict[str, int]:
    """Freeze ``model`` and wrap matching linear projections with LoRA."""

    target_names = tuple(dict.fromkeys(str(name).strip() for name in targets if str(name).strip()))
    if not target_names:
        raise ValueError("LoRA needs at least one target module suffix")
    model.requires_grad_(False)
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target_names
    ]
    if not matches:
        raise ValueError(f"no linear modules matched LoRA targets {target_names}")
    for name, module in matches:
        parent, child_name = _split_parent(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "modules": len(matches),
        "trainable": trainable,
        "total": total,
        "frozen": total - trainable,
    }


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model).items():
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    modules = lora_modules(model)
    expected = {
        f"{name}.{parameter}"
        for name in modules
        for parameter in ("lora_A", "lora_B")
    }
    if set(state) != expected:
        missing = sorted(expected.difference(state))
        unexpected = sorted(set(state).difference(expected))
        raise ValueError(f"LoRA adapter state mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, module in modules.items():
            module.lora_A.copy_(state[f"{name}.lora_A"])
            module.lora_B.copy_(state[f"{name}.lora_B"])


def unmerge_adapter_from_base(model: nn.Module) -> None:
    """Turn a merged base plus adapter into the equivalent unmerged pair."""

    with torch.no_grad():
        for module in lora_modules(model).values():
            module.base.weight.sub_(module.delta_weight().to(module.base.weight.dtype))


def merged_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return an ordinary model state with all adapters folded into weights."""

    adapters = lora_modules(model)
    source = model.state_dict()
    merged: dict[str, torch.Tensor] = {}
    adapter_names = set(adapters)
    for key, value in source.items():
        if key.endswith(".lora_A") or key.endswith(".lora_B"):
            continue
        if key.endswith(".base.weight"):
            module_name = key[: -len(".base.weight")]
            canonical_key = f"{module_name}.weight"
            delta = adapters[module_name].delta_weight().detach()
            merged[canonical_key] = (value.detach() + delta.to(value.dtype)).cpu()
            continue
        if key.endswith(".base.bias"):
            module_name = key[: -len(".base.bias")]
            if module_name in adapter_names:
                merged[f"{module_name}.bias"] = value.detach().cpu()
                continue
        merged[key] = value.detach().cpu()
    return merged
