"""Dense SwiGLU with a dispatch-free lexical low-rank residual."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_mlp
from .base import MLPBase, MLPConfig
from .fused_activations import fused_silu_mul


@register_mlp("lexical_modulated")
@register_mlp("lexical_object")
class LexicalModulatedMLP(MLPBase):
    """Shared dense SwiGLU plus a token-modulated low-rank residual.

    The object branch uses two regular dense projections and an embedding
    lookup. It deliberately avoids expert grouping, sorting, and scatter.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.object_rank = int(config.lexical_object_rank)

        self.shared_gate = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.shared_up = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.shared_down = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)

        self.object_up = nn.Linear(self.hidden_size, self.object_rank, bias=False)
        self.object_down = nn.Linear(self.object_rank, self.hidden_size, bias=False)
        self.token_scale = nn.Embedding(config.vocab_size, self.object_rank)
        self.object_output_gate = nn.Parameter(
            torch.tensor(float(config.lexical_object_gate_init))
        )
        nn.init.zeros_(self.token_scale.weight)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        lexical_token_scale_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError("token_ids are required for lexical_modulated MLP")
        if token_ids.shape != hidden_states.shape[:-1]:
            raise ValueError(
                f"token_ids shape {tuple(token_ids.shape)} must match hidden state prefix "
                f"shape {tuple(hidden_states.shape[:-1])}"
            )

        shared = self.shared_down(
            fused_silu_mul(self.shared_gate(hidden_states), self.shared_up(hidden_states))
        )
        scale = 1.0 + (
            lexical_token_scale_values
            if lexical_token_scale_values is not None
            else self.token_scale(token_ids)
        )
        object_hidden = F.silu(self.object_up(hidden_states)) * scale
        object_residual = self.object_down(object_hidden)
        return shared + self.object_output_gate * object_residual

    def training_control_capabilities(self) -> frozenset[str]:
        return frozenset({"lexical_object_gate"})

    def training_telemetry(self) -> dict[str, float]:
        return {"object_gate": float(self.object_output_gate.detach().float().item())}