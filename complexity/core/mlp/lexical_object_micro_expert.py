"""Lexical low-rank object plus dispatch-free deterministic micro-experts."""

from typing import Optional

import torch
import torch.nn as nn

from ..registry import register_mlp
from .base import MLPConfig
from .fused_activations import fused_silu_mul
from .lexical_modulated import LexicalModulatedMLP


@register_mlp("lexical_object_micro_expert")
class LexicalObjectMicroExpertMLP(LexicalModulatedMLP):
    """Add a tiny top-1 expert residual without token sorting or scatter."""

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.micro_num_experts = int(config.micro_num_experts)
        self.micro_expert_width = int(config.micro_expert_width)
        total_width = self.micro_num_experts * self.micro_expert_width
        self.micro_gate = nn.Linear(self.hidden_size, total_width, bias=False)
        self.micro_up = nn.Linear(self.hidden_size, total_width, bias=False)
        self.micro_down = nn.Parameter(
            torch.empty(
                self.micro_num_experts,
                self.micro_expert_width,
                self.hidden_size,
            )
        )
        nn.init.kaiming_uniform_(self.micro_down, a=5**0.5)
        self.micro_output_gate = nn.Parameter(
            torch.tensor(float(config.micro_expert_gate_init))
        )
        mapping = (
            torch.arange(config.vocab_size, dtype=torch.long) + int(config.layer_idx)
        ) % self.micro_num_experts
        self.register_buffer("token_to_micro_expert", mapping)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        output = super().forward(hidden_states, token_ids=token_ids, **kwargs)
        assert token_ids is not None
        shape = hidden_states.shape[:-1] + (
            self.micro_num_experts,
            self.micro_expert_width,
        )
        micro_hidden = fused_silu_mul(
            self.micro_gate(hidden_states), self.micro_up(hidden_states)
        ).view(shape)
        all_outputs = torch.einsum("...ew,ewd->...ed", micro_hidden, self.micro_down)
        expert_ids = self.token_to_micro_expert[token_ids]
        selected = torch.gather(
            all_outputs,
            -2,
            expert_ids[..., None, None].expand(*expert_ids.shape, 1, self.hidden_size),
        ).squeeze(-2)
        return output + self.micro_output_gate * selected