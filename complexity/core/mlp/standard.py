"""
Standard MLP implementations: GELU FFN and SwiGLU.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import MLPBase, MLPConfig
from ..registry import register_mlp


@register_mlp("standard")
@register_mlp("gelu")
class StandardMLP(MLPBase):
    """
    Standard Feed-Forward Network with GELU activation.

    Architecture:
        out = down_proj(gelu(up_proj(x)))

    Used in: GPT-2, BERT, original Transformer
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act_fn = self.get_activation(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.up_proj(hidden_states)))


@register_mlp("swiglu")
@register_mlp("silu")
@register_mlp("llama")
class SwiGLUMLP(MLPBase):
    """
    SwiGLU MLP (Gated Linear Unit with Swish activation).

    Architecture:
        out = down_proj(swish(gate_proj(x)) * up_proj(x))

    Benefits vs standard FFN:
    - More expressive gating mechanism
    - Same parameter count (gate splits intermediate)
    - Better training dynamics

    Used in: Llama, Mistral, Gemma, and most modern LLMs

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        # Three linear projections (no bias for efficiency)
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act_fn = self.get_activation(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # SwiGLU: swish(gate) * up
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)


@register_mlp("geglu")
class GeGLUMLP(MLPBase):
    """
    GeGLU MLP (Gated Linear Unit with GELU activation).

    Architecture:
        out = down_proj(gelu(gate_proj(x)) * up_proj(x))

    Similar to SwiGLU but uses GELU instead of SiLU/Swish.
    Sometimes preferred for specific domains.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(self, config: MLPConfig):
        # Force GELU activation
        config.hidden_act = "gelu"
        super().__init__(config)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.act_fn = F.gelu

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        gate = self.act_fn(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        return self.down_proj(gate * up)
