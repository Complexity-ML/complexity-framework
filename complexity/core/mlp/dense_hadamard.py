"""
Dense SwiGLU MLP with deterministic Hadamard initialisation.

Architecturally identical to `SwiGLUMLP` — three Linear projections,
SiLU-gated GLU — but weights are initialised through the RNG-free
Hadamard scheme in `hadamard_init.py`. Two instantiations of
`DenseHadamardMLP` with the same config produce bit-identical weights
on any hardware, in any environment, without seed management.

Use cases:
  - Reproducibility-first science: guarantees identical init across
    runs, machines, and checkpoints.
  - Baseline ablation against `SwiGLUMLP` (same forward, different init)
    to isolate the effect of the initialisation distribution on final
    training quality.
  - FSDP / multi-rank setups where seeded RNG is brittle: every rank
    derives the exact same weights from the same `(shape, layer_idx)`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import MLPBase, MLPConfig
from .hadamard_init import hadamard_init_
from ..registry import register_mlp


@register_mlp("dense_hadamard")
@register_mlp("dense_deterministic")
@register_mlp("hadamard_swiglu")
class DenseHadamardMLP(MLPBase):
    """
    SwiGLU MLP with deterministic Hadamard-based weight initialisation.

    Forward pass is identical to `SwiGLUMLP`:
        out = down_proj(swish(gate_proj(x)) * up_proj(x))

    Init is entirely deterministic: the three weight matrices are each
    set by `hadamard_init_` with a per-matrix offset so the three
    projections do not share identical sign patterns.
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

        self._init_hadamard(config.layer_idx)

    def _init_hadamard(self, layer_idx: int) -> None:
        """
        Apply Hadamard init to the three Linear weights.

        Each projection gets a distinct `layer_idx` offset (×4 + k) so the
        gate / up / down projections within a single block do not end up
        with identical sign patterns — which would be degenerate.
        """
        base = int(layer_idx) * 4
        hadamard_init_(self.gate_proj.weight, layer_idx=base + 1)
        hadamard_init_(self.up_proj.weight,   layer_idx=base + 2)
        hadamard_init_(self.down_proj.weight, layer_idx=base + 3)
        if self.config.bias:
            nn.init.zeros_(self.gate_proj.bias)
            nn.init.zeros_(self.up_proj.bias)
            nn.init.zeros_(self.down_proj.bias)

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
