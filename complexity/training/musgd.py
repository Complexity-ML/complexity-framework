"""MuSGD optimizer used by the current TR-HASH Vision training pipeline."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def zeroth_power_newton_schulz(matrix: torch.Tensor) -> torch.Tensor:
    """Approximate the orthogonal factor of one matrix."""

    if matrix.ndim != 2:
        raise ValueError("MuSGD orthogonalization expects a 2D matrix")
    transpose = matrix.shape[0] > matrix.shape[1]
    work = matrix.mT if transpose else matrix
    dtype = torch.bfloat16 if matrix.device.type == "cuda" else torch.float32
    work = work.to(dtype)
    work = work / work.norm().clamp_min(1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(5):
        gram = work @ work.mT
        work = a * work + (b * gram + c * (gram @ gram)) @ work
    work = work.mT if transpose else work
    return work.to(matrix.dtype)


class MuSGD(Optimizer):
    """Hybrid Muon+SGD optimizer used by YOLO26-style long vision runs.

    Groups marked ``use_muon`` receive an orthogonalized momentum update plus
    the standard SGD update. Other groups receive SGD only. No Adam/AdamW
    state or update is used.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon_weight: float = 0.2,
        sgd_weight: float = 1.0,
    ) -> None:
        if lr < 0 or momentum < 0 or weight_decay < 0:
            raise ValueError("MuSGD learning rate, momentum and decay must be non-negative")
        if nesterov and momentum <= 0:
            raise ValueError("Nesterov MuSGD requires positive momentum")
        if muon_weight < 0 or sgd_weight < 0:
            raise ValueError("MuSGD component weights must be non-negative")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "use_muon": use_muon,
        }
        super().__init__(params, defaults)
        self.muon_weight = float(muon_weight)
        self.sgd_weight = float(sgd_weight)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            momentum = float(group["momentum"])
            weight_decay = float(group["weight_decay"])
            nesterov = bool(group["nesterov"])
            use_muon = bool(group["use_muon"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                state = self.state[parameter]

                if use_muon:
                    muon_buffer = state.setdefault("muon_momentum", torch.zeros_like(parameter))
                    muon_buffer.lerp_(gradient, 1.0 - momentum)
                    direction = gradient.lerp(muon_buffer, momentum) if nesterov else muon_buffer
                    original_shape = direction.shape
                    matrix = direction.reshape(direction.shape[0], -1)
                    matrix = zeroth_power_newton_schulz(matrix)
                    matrix.mul_(math.sqrt(max(1.0, original_shape[-2] / original_shape[-1])))
                    parameter.add_(
                        matrix.reshape(original_shape),
                        alpha=-(lr * self.muon_weight),
                    )
                    sgd_buffer_name = "sgd_momentum"
                    sgd_lr = lr * self.sgd_weight
                else:
                    sgd_buffer_name = "momentum_buffer"
                    sgd_lr = lr

                sgd_gradient = gradient
                if weight_decay:
                    sgd_gradient = gradient.add(parameter, alpha=weight_decay)
                sgd_buffer = state.setdefault(sgd_buffer_name, torch.zeros_like(parameter))
                sgd_buffer.mul_(momentum).add_(sgd_gradient)
                sgd_direction = (
                    sgd_gradient.add(sgd_buffer, alpha=momentum) if nesterov else sgd_buffer
                )
                parameter.add_(sgd_direction, alpha=-sgd_lr)
        return loss


def build_musgd_parameter_groups(
    model: nn.Module,
    *,
    learning_rate: Callable[[str], tuple[float, str]],
    momentum: float,
    weight_decay: float,
) -> list[dict[str, Any]]:
    """Split vision-model parameters into named MuSGD and SGD groups."""

    norm_types = tuple(
        module_type
        for name, module_type in vars(nn).items()
        if isinstance(module_type, type) and "Norm" in name
    )
    grouped: dict[tuple[float, str, bool, float], list[nn.Parameter]] = {}
    for module_name, module in model.named_modules():
        for parameter_name, parameter in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
            lr, group_name = learning_rate(full_name)
            use_muon = parameter.ndim in {2, 4}
            no_decay = (
                parameter_name == "bias"
                or isinstance(module, norm_types)
                or "logit_scale" in full_name
            )
            decay = 0.0 if no_decay else weight_decay
            grouped.setdefault((lr, group_name, use_muon, decay), []).append(parameter)

    return [
        {
            "params": parameters,
            "lr": lr,
            "momentum": momentum,
            "nesterov": True,
            "weight_decay": decay,
            "use_muon": use_muon,
            "group_name": group_name,
        }
        for (lr, group_name, use_muon, decay), parameters in grouped.items()
    ]


def named_learning_rates(optimizer: Optimizer) -> dict[str, float]:
    """Return the current LR of every logical parameter scope."""

    result: dict[str, float] = {}
    for group in optimizer.param_groups:
        name = str(group.get("group_name", "default"))
        lr = float(group["lr"])
        previous = result.setdefault(name, lr)
        if not math.isclose(previous, lr, rel_tol=1e-12, abs_tol=0.0):
            raise RuntimeError(f"inconsistent learning rates for MuSGD group {name}")
    return result
