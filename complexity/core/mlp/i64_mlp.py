"""
I64 Integer MLP — INT8 matmuls + LUT SiLU activation.

SwiGLU MLP where every matmul is INT8 and SiLU is a LUT lookup.
Forward (quantized): y = down_i8(silu_lut(gate_i8(x)) * up_i8(x))

Three INT8 matmuls + one LUT lookup. Zero float compute in the MLP.

Registered as "i64_swiglu" / "integer_swiglu" in the MLP registry.

INL 2025 — ported from complexity-i64.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from .base import MLPBase, MLPConfig
from ..registry import register_mlp
from ..integer_ops import (
    int8_linear, int8_fused_gate_up, silu_multiply_integer,
    quantize_weight_int8,
)


@register_mlp("i64_swiglu")
@register_mlp("integer_swiglu")
class I64SwiGLUMLP(MLPBase):
    """
    Integer SwiGLU MLP.

    Weights stored as float for training. Call quantize() to convert
    to INT8 with fused gate+up and LUT SiLU for inference.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        if hasattr(self, 'gate_up_int8'):
            return self._forward_int8(hidden_states)
        # Float fallback (training)
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))

    def _forward_int8(self, x: torch.Tensor) -> torch.Tensor:
        """Full INT8 path: fused gate+up -> LUT SiLU*up -> INT8 down."""
        gate, up = int8_fused_gate_up(
            x, self.gate_up_int8, self.gate_up_scale, self.intermediate_size,
        )
        inter = silu_multiply_integer(gate, up)
        return int8_linear(inter, self.down_int8, self.down_scale)

    def quantize(self):
        """Convert float weights to INT8 for inference."""
        gq, gs = quantize_weight_int8(self.gate_proj.weight.data)
        uq, us = quantize_weight_int8(self.up_proj.weight.data)
        dq, ds = quantize_weight_int8(self.down_proj.weight.data)

        # Fused gate+up
        self.register_buffer("gate_up_int8", torch.cat([gq, uq], dim=0))
        self.register_buffer("gate_up_scale", torch.cat([gs, us]))
        self.register_buffer("down_int8", dq)
        self.register_buffer("down_scale", ds)

        # Free float weights
        del self.gate_proj
        del self.up_proj
        del self.down_proj


@register_mlp("i64_token_routed")
@register_mlp("integer_moe")
class I64TokenRoutedMLP(MLPBase):
    """
    Integer token-routed MLP. i64 routing + INT8 expert compute.

    Routing: expert_id = token_id % num_experts (pure integer, no gate)
    Expert compute: INT8 SwiGLU per expert

    Call quantize() after training to convert expert weights to INT8.
    """

    def __init__(self, config: MLPConfig):
        super().__init__(config)

        self.num_experts = config.num_experts
        self.vocab_size = config.vocab_size
        self.expert_inter = self.intermediate_size // self.num_experts

        # Expert weights (will be quantized to INT8)
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, 2 * self.expert_inter)
        )
        self.down_proj_weight = nn.Parameter(
            torch.empty(self.num_experts, self.expert_inter, self.hidden_size)
        )

        # i64 routing table
        self.register_buffer(
            "token_to_expert",
            torch.arange(self.vocab_size, dtype=torch.long) % self.num_experts,
        )

        nn.init.kaiming_uniform_(self.gate_up_proj, a=5**0.5)
        nn.init.kaiming_uniform_(self.down_proj_weight, a=5**0.5)

    def route(self, token_ids: Optional[torch.Tensor], num_tokens: int,
              device: torch.device, mu: Optional[torch.Tensor] = None) -> torch.Tensor:
        if token_ids is None:
            return torch.zeros(num_tokens, dtype=torch.long, device=device)
        ids = token_ids.clamp(0, self.token_to_expert.shape[0] - 1)
        return self.token_to_expert[ids]

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        flat_hidden = hidden_states.view(-1, self.hidden_size)

        flat_token_ids = token_ids.view(-1) if token_ids is not None else None
        expert_ids = self.route(flat_token_ids, flat_hidden.shape[0], flat_hidden.device, mu=mu)
        if expert_ids.dim() > 1:
            expert_ids = expert_ids.view(-1)

        output = self._expert_forward(flat_hidden, expert_ids)
        return output.view(batch_size, seq_len, self.hidden_size)

    def _expert_forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Dispatch tokens to experts, compute, gather."""
        if hasattr(self, 'gate_up_int8_experts'):
            return self._expert_forward_int8(x, expert_ids)

        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            xe = x[mask]
            gu = torch.mm(xe, self.gate_up_proj[e])
            gate, up = gu.split(self.expert_inter, dim=-1)
            inter = F.silu(gate) * up
            output[mask] = torch.mm(inter, self.down_proj_weight[e])
        return output

    def _expert_forward_int8(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """INT8 expert forward with LUT SiLU."""
        output = torch.zeros_like(x)
        for e in range(self.num_experts):
            mask = expert_ids == e
            if not mask.any():
                continue
            xe = x[mask]
            gu = int8_linear(xe, self.gate_up_int8_experts[e], self.gate_up_scale_experts[e])
            gate, up = gu.split(self.expert_inter, dim=-1)
            inter = silu_multiply_integer(gate, up)
            output[mask] = int8_linear(inter, self.down_int8_experts[e], self.down_scale_experts[e])
        return output

    def quantize(self):
        """Quantize expert weights to INT8."""
        gu_q, gu_s, dn_q, dn_s = [], [], [], []
        for e in range(self.num_experts):
            gq, gs = quantize_weight_int8(self.gate_up_proj[e].t())
            dq, ds = quantize_weight_int8(self.down_proj_weight[e].t())
            gu_q.append(gq); gu_s.append(gs)
            dn_q.append(dq); dn_s.append(ds)

        self.register_buffer("gate_up_int8_experts", torch.stack(gu_q))
        self.register_buffer("gate_up_scale_experts", torch.stack(gu_s))
        self.register_buffer("down_int8_experts", torch.stack(dn_q))
        self.register_buffer("down_scale_experts", torch.stack(dn_s))

        # Free float weights
        self.gate_up_proj = None
        self.down_proj_weight = None
