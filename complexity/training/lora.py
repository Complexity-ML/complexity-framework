"""Framework-native LoRA adapters for linear and TR-Hash expert tensors.

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
from torch.nn.utils import parametrize

from complexity.tr_hash.engine import TRHashEngine


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
        return self.base(inputs) + self.lora_residual(inputs)

    def lora_residual(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return only the trainable residual branch.

        Attention can fuse the frozen K/Q/V base projections while reusing
        this exact branch for mixed or otherwise unsupported adapter shapes.
        """

        low_rank = F.linear(F.linear(self.dropout(inputs), self.lora_A), self.lora_B)
        return low_rank * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return (self.lora_B @ self.lora_A) * self.scaling


class LoRAExpertTensor(nn.Module):
    """Per-expert low-rank update for one ``[experts, input, output]`` tensor.

    TR-Hash stores its routed projections as three-dimensional parameters so
    they can be consumed directly by the grouped Triton kernels. PyTorch
    parametrization keeps that exact public tensor contract: fused kernels see
    an ordinary merged tensor while only the low-rank factors are trainable.
    """

    is_lora_adapter = True

    def __init__(
        self,
        base: torch.Tensor,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if base.ndim != 3:
            raise ValueError("expert LoRA expects [experts, input, output]")
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")
        experts, input_size, output_size = base.shape
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = float(dropout)
        self.lora_A = nn.Parameter(
            torch.empty(
                experts,
                input_size,
                self.rank,
                device=base.device,
                dtype=base.dtype,
            )
        )
        self.lora_B = nn.Parameter(
            torch.zeros(
                experts,
                self.rank,
                output_size,
                device=base.device,
                dtype=base.dtype,
            )
        )
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def delta_weight(self, *, regularized: bool = False) -> torch.Tensor:
        left = self.lora_A
        if regularized and self.dropout:
            # Expert weights are consumed as tensors by grouped kernels, not
            # through an nn.Linear call where input dropout could be inserted.
            # Factor dropout provides the corresponding training-only
            # regularization while preserving the fused tensor interface.
            left = F.dropout(left, p=self.dropout, training=True)
        return torch.bmm(left, self.lora_B) * self.scaling

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        left = self.lora_A
        if self.training and self.dropout:
            left = F.dropout(left, p=self.dropout, training=True)
        # Fuse the low-rank batched GEMM, scaling, and addition to the frozen
        # expert tensor. This avoids materializing an intermediate dense delta
        # plus two pointwise kernels before the routed CGGR projection.
        return torch.baddbmm(
            base,
            left,
            self.lora_B,
            beta=1.0,
            alpha=self.scaling,
        )


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
    """Freeze ``model`` and adapt matching linear/expert projections."""

    target_names = tuple(dict.fromkeys(str(name).strip() for name in targets if str(name).strip()))
    if not target_names:
        raise ValueError("LoRA needs at least one target module suffix")
    model.requires_grad_(False)
    linear_matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and name.rsplit(".", 1)[-1] in target_names
    ]
    for name, module in linear_matches:
        parent, child_name = _split_parent(model, name)
        setattr(
            parent,
            child_name,
            LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout),
        )
    expert_matches: list[tuple[str, TRHashEngine, str]] = []
    expert_targets = {"expert_gate", "expert_up", "expert_down"}.intersection(target_names)
    if expert_targets:
        for module_name, module in model.named_modules():
            if not isinstance(module, TRHashEngine):
                continue
            for tensor_name in sorted(expert_targets):
                expert_matches.append((module_name, module, tensor_name))
                parametrize.register_parametrization(
                    module,
                    tensor_name,
                    LoRAExpertTensor(
                        getattr(module, tensor_name),
                        rank=rank,
                        alpha=alpha,
                        dropout=dropout,
                    ),
                )
                module.parametrizations[tensor_name].original.requires_grad_(False)
    if not linear_matches and not expert_matches:
        raise ValueError(f"no modules matched LoRA targets {target_names}")
    for module in model.modules():
        if isinstance(module, TRHashEngine):
            module.refresh_fused_shared_lora_weight()
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "modules": len(linear_matches) + len(expert_matches),
        "linear_modules": len(linear_matches),
        "expert_tensors": len(expert_matches),
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


def expert_lora_modules(
    model: nn.Module,
) -> dict[str, tuple[TRHashEngine, str, LoRAExpertTensor]]:
    adapters: dict[str, tuple[TRHashEngine, str, LoRAExpertTensor]] = {}
    for module_name, module in model.named_modules():
        if not isinstance(module, TRHashEngine) or not parametrize.is_parametrized(module):
            continue
        for tensor_name, parametrizations in module.parametrizations.items():
            if len(parametrizations) != 1 or not isinstance(
                parametrizations[0], LoRAExpertTensor
            ):
                continue
            canonical_name = f"{module_name}.{tensor_name}" if module_name else tensor_name
            adapters[canonical_name] = (module, tensor_name, parametrizations[0])
    return adapters


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {}
    for name, module in lora_modules(model).items():
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    for name, (_, _, module) in expert_lora_modules(model).items():
        state[f"{name}.lora_A"] = module.lora_A.detach().cpu()
        state[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return state


def load_adapter_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    modules = lora_modules(model)
    expert_modules = expert_lora_modules(model)
    expected = {
        f"{name}.{parameter}"
        for name in modules
        for parameter in ("lora_A", "lora_B")
    } | {
        f"{name}.{parameter}"
        for name in expert_modules
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
        for name, (_, _, module) in expert_modules.items():
            module.lora_A.copy_(state[f"{name}.lora_A"])
            module.lora_B.copy_(state[f"{name}.lora_B"])


def unmerge_adapter_from_base(model: nn.Module) -> None:
    """Turn a merged base plus adapter into the equivalent unmerged pair."""

    with torch.no_grad():
        for module in lora_modules(model).values():
            module.base.weight.sub_(module.delta_weight().to(module.base.weight.dtype))
        for parent, tensor_name, module in expert_lora_modules(model).values():
            original = parent.parametrizations[tensor_name].original
            original.sub_(module.delta_weight().to(original.dtype))
    for module in model.modules():
        if isinstance(module, TRHashEngine):
            module.refresh_fused_shared_lora_weight()


def merged_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return an ordinary model state with all adapters folded into weights."""

    adapters = lora_modules(model)
    expert_adapters = expert_lora_modules(model)
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
        if "parametrizations." in key:
            continue
        merged[key] = value.detach().cpu()
    for name, (parent, tensor_name, module) in expert_adapters.items():
        original = parent.parametrizations[tensor_name].original
        merged[name] = (
            original.detach() + module.delta_weight().detach().to(original.dtype)
        ).cpu()
    return merged
