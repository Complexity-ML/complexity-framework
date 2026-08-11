"""Exponential moving average weights for detector evaluation and export."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn


class ModelEMA:
    """Maintain a non-trainable moving copy without altering optimizer state."""

    def __init__(self, model: nn.Module, decay: float, updates: int = 0):
        if not 0.0 < decay < 1.0:
            raise ValueError("EMA decay must be in (0, 1)")
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay = float(decay)
        self.updates = int(updates)

    def _decay(self) -> float:
        return self.decay * (1.0 - math.exp(-self.updates / 2_000.0))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self._decay()
        model_state = model.state_dict()
        for name, value in self.module.state_dict().items():
            source = model_state[name].detach()
            if value.is_floating_point():
                value.mul_(decay).add_(source, alpha=1.0 - decay)
            else:
                value.copy_(source)

    def load_state_dict(self, state_dict: dict[str, torch.Tensor], updates: int) -> None:
        self.module.load_state_dict(state_dict)
        self.updates = int(updates)
