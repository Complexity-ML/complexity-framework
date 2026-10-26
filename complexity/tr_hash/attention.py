"""Canonical TR-HASH attention and MLP construction for non-language domains.

Spatial and cross-modal identity ordering are task adaptations. This module
provides architecture behavior; it makes no vision or audio quality claim.
"""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .composed_routing import bind_value_channels, composed_head_routes
from .config import TRHashBackend, TRHashEngineConfig, TRHashPrecision, TRHashStrategy
from .engine import TRHashEngine


def resolve_domain_widths(
    config, *, hidden="hidden_size", shared="shared_width", expert="expert_width"
) -> None:
    """Resolve omitted capacity while preserving explicit positive widths."""
    width = getattr(config, hidden)
    if getattr(config, shared) is None:
        object.__setattr__(config, shared, 2 * width)
    if getattr(config, expert) is None:
        object.__setattr__(config, expert, max(1, width // 4))
    if getattr(config, shared) <= 0 or getattr(config, expert) <= 0:
        raise ValueError("TR-HASH shared and expert widths must be positive")


def resolve_domain_heads(config, *, heads="num_attention_heads", kv="num_key_value_heads") -> None:
    if getattr(config, kv) is None:
        object.__setattr__(config, kv, 1)
    if getattr(config, heads) != 4 or getattr(config, kv) != 1:
        raise ValueError("TR-HASH requires exactly four query heads and one key/value head")


def make_domain_mlp(
    *,
    hidden_size: int,
    vocab_size: int,
    num_experts: int,
    top_k: int,
    shared_width: int,
    expert_width: int,
    layer_index: int,
    precision: TRHashPrecision | str = TRHashPrecision.BF16,
) -> TRHashEngine:
    return TRHashEngine(
        TRHashEngineConfig(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_experts=num_experts,
            top_k=top_k,
            shared_width=shared_width,
            expert_width=expert_width,
            routing_strategy=TRHashStrategy.COMPOSED_MODULO,
            route_weights=(0.5, 0.5),
            shared_output_scale=1.0,
            routed_output_scale=2.0,
            layer_index=layer_index,
            precision=TRHashPrecision(precision),
            backend=TRHashBackend.AUTO,
        )
    )


class DomainSelfAttention(nn.Module):
    """Non-causal four-query/one-KV attention with identity-based head selection.

    SDPA boolean/additive masks, relative spatial biases and local windows are
    supplied by the caller. The first layer binds value-input channel quarters.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        *,
        num_key_value_heads: int | None = None,
        layer_index: int = 0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = 1 if num_key_value_heads is None else num_key_value_heads
        if num_heads != 4 or self.num_key_value_heads != 1:
            raise ValueError("TR-HASH requires exactly four query heads and one key/value head")
        if hidden_size <= 0 or hidden_size % 4:
            raise ValueError("TR-HASH hidden_size must be positive and divisible by four")
        self.layer_index = layer_index
        self.head_dim = hidden_size // 4
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, hidden_size + 2 * self.head_dim, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        if route_ids is None:
            raise ValueError("TR-HASH attention requires original route_ids")
        q_weight, k_weight, v_weight = self.qkv.weight.split(
            (width, self.head_dim, self.head_dim), dim=0
        )
        q = F.linear(x, q_weight).view(batch, length, 4, self.head_dim).transpose(1, 2)
        k = F.linear(x, k_weight).view(batch, length, 1, self.head_dim).transpose(1, 2)
        values = bind_value_channels(x, route_ids, self.layer_index)
        v = F.linear(values, v_weight).view(batch, length, 1, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        y = F.scaled_dot_product_attention(
            q,
            k.repeat_interleave(4, dim=1),
            v.repeat_interleave(4, dim=1),
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        routes = composed_head_routes(route_ids, self.layer_index, 4)
        selection = F.one_hot(routes, num_classes=4).sum(dim=-2)
        y = y * selection.permute(0, 2, 1).unsqueeze(-1).to(y.dtype) * math.sqrt(2)
        return self.out(y.transpose(1, 2).reshape(batch, length, width))
