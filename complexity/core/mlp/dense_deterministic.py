"""
Dense SwiGLU MLP with a marker type for deterministic initialisation.

Forward pass is identical to ``SwiGLUMLP``. The class exists only so
``ComplexityModel._init_dense_deterministic`` can detect, via
``isinstance``, that the model is opting into RNG-free initialisation
for every weight matrix (embeddings + attention + FFN).

Prior versions of this module shipped a Hadamard-based init scheme on
the FFN projections. We dropped it after an ablation (1000 steps,
72.9M params, FineWeb-Edu) showed Hadamard ±s weights consistently
underperform a deterministic Gaussian of the same standard deviation
by ~0.15 loss. The current scheme is deterministic Gaussian
everywhere, which matches the framework default's training trajectory
while removing the dependence on global RNG state.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import MLPBase, MLPConfig
from ..registry import register_mlp


@register_mlp("dense_deterministic")
@register_mlp("dense_hadamard")        # legacy alias (no Hadamard in the weights)
@register_mlp("hadamard_swiglu")       # legacy alias
class DenseDeterministicMLP(MLPBase):
    """
    Marker class for deterministic-init dense SwiGLU FFN.

    Architecture matches ``SwiGLUMLP`` bit-for-bit (three Linear
    projections, SiLU-gated GLU). The actual deterministic init is
    performed by ``ComplexityModel._init_dense_deterministic`` during
    model construction, which walks every block of ``isinstance`` of
    this class and re-initialises its weights — together with the
    attention projections and the embedding table — via a locally-
    seeded Generator. See ``deterministic_init.py``.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.bias
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=config.bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.bias
        )
        self.act_fn = self.get_activation(config.hidden_act)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        if self.act_fn is F.silu:
            from .fused_activations import fused_silu_mul
            inter = fused_silu_mul(gate, up)
        else:
            inter = self.act_fn(gate) * up
        return self.down_proj(inter)
