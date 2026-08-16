"""Dense SwiGLU with dispatch-free lexical channel modulation."""

import math
from typing import Optional

import torch
import torch.nn as nn

from ..registry import register_mlp
from .base import MLPBase, MLPConfig
from .fused_activations import fused_silu_mul


@register_mlp("lexical_channel_modulated")
@register_mlp("lexical_channel_object")
class LexicalChannelModulatedMLP(MLPBase):
    """Modulate dense SwiGLU channels using a compact lexical object table."""

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.object_rank = int(config.lexical_object_rank)
        self.shared_gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.shared_up = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.shared_down = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.token_scale = nn.Embedding(config.vocab_size, self.object_rank)
        self.object_output_gate = nn.Parameter(
            torch.tensor(float(config.lexical_object_gate_init))
        )
        nn.init.zeros_(self.token_scale.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError("token_ids are required for lexical_channel_modulated MLP")
        if token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError(
                f"token_ids shape {tuple(token_ids.shape)} must match hidden state prefix "
                f"shape {tuple(hidden_states.shape[:-1])}"
            )

        channels = fused_silu_mul(
            self.shared_gate(hidden_states), self.shared_up(hidden_states)
        )
        repeats = math.ceil(self.intermediate_size / self.object_rank)
        lexical_scale = self.token_scale(token_ids).repeat_interleave(repeats, dim=-1)
        lexical_scale = lexical_scale[..., : self.intermediate_size]
        channels = channels * (1.0 + self.object_output_gate * lexical_scale)
        return self.shared_down(channels)