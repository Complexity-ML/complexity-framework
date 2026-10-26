"""Shared Source67 attention and MLP construction for non-language domains.

The four-head recipe matches the text reference. Other head counts, spatial
attention and cross-modal use are architectural extensions without measured
vision/audio accuracy. ``legacy`` retains the former MHA parameter layout.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from .baseline import bind_source67_values, source67_head_routes
from .config import TRHashBackend, TRHashEngineConfig, TRHashPrecision, TRHashStrategy
from .engine import TRHashEngine


def validate_domain_profile(profile: str, version: int = 1) -> None:
    if profile not in {"source67", "legacy"}:
        raise ValueError("architecture_profile must be 'source67' or 'legacy'")
    if version != 1:
        raise ValueError("unsupported architecture_profile_version")


def resolve_domain_widths(
    config,
    *,
    hidden: str = "hidden_size",
    shared: str = "shared_width",
    expert: str = "expert_width",
    legacy_shared: int = 128,
    legacy_expert: int = 48,
) -> None:
    """Resolve omitted capacity while retaining every explicit width."""
    source = config.architecture_profile == "source67"
    width = getattr(config, hidden)
    if getattr(config, shared) is None:
        object.__setattr__(config, shared, 2 * width if source else legacy_shared)
    if getattr(config, expert) is None:
        object.__setattr__(config, expert, max(1, width // 4) if source else legacy_expert)


def resolve_domain_heads(
    config, *, heads: str = "num_attention_heads", kv: str = "num_key_value_heads"
) -> None:
    count = getattr(config, heads)
    if getattr(config, kv) is None:
        object.__setattr__(config, kv, 1 if config.architecture_profile == "source67" else count)
    if count <= 0 or getattr(config, kv) <= 0 or count % getattr(config, kv):
        raise ValueError(f"{heads} must be positive and divisible by {kv}")


def make_domain_mlp(
    *,
    hidden_size: int,
    vocab_size: int,
    num_experts: int,
    top_k: int,
    shared_width: int,
    expert_width: int,
    layer_index: int,
    route_seed: int,
    architecture_profile: str = "source67",
    precision: TRHashPrecision | str = TRHashPrecision.BF16,
) -> TRHashEngine:
    validate_domain_profile(architecture_profile)
    source = architecture_profile == "source67"
    return TRHashEngine(
        TRHashEngineConfig(
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            num_experts=num_experts,
            top_k=top_k,
            shared_width=shared_width,
            expert_width=expert_width,
            routing_strategy=(
                TRHashStrategy.COMPOSED_MODULO if source else TRHashStrategy.BALANCED_HASH
            ),
            route_weights=(0.5, 0.5) if source else (),
            shared_output_scale=1.0,
            routed_output_scale=2.0 if source else 1.0,
            layer_index=layer_index,
            route_seed=route_seed,
            precision=TRHashPrecision(precision),
            backend=TRHashBackend.AUTO,
        )
    )


class DomainSelfAttention(nn.Module):
    """Non-causal GQA with Source67 V0 and output-head selection.

    ``attention_mask`` uses SDPA's bool/additive semantics. Relative spatial
    bias and local-window masks remain the caller's responsibility.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.0,
        *,
        num_key_value_heads: int | None = None,
        architecture_profile: str = "source67",
        layer_index: int = 0,
    ):
        super().__init__()
        validate_domain_profile(architecture_profile)
        self.architecture_profile = architecture_profile
        self.layer_index = layer_index
        self.num_heads = num_heads
        self.num_key_value_heads = (
            (1 if architecture_profile == "source67" else num_heads)
            if num_key_value_heads is None
            else num_key_value_heads
        )
        if hidden_size % num_heads or num_heads % self.num_key_value_heads:
            raise ValueError("hidden size and query heads must be divisible by their head counts")
        if architecture_profile == "source67" and hidden_size % 4:
            raise ValueError("source67 V0 requires hidden_size divisible by four")
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        bias = architecture_profile == "legacy"
        self.qkv = nn.Linear(
            hidden_size, hidden_size + 2 * self.num_key_value_heads * self.head_dim, bias=bias
        )
        self.out = nn.Linear(hidden_size, hidden_size, bias=bias)
        if architecture_profile == "source67":
            self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)

    def forward(
        self,
        x: torch.Tensor,
        route_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, length, width = x.shape
        kv_width = self.num_key_value_heads * self.head_dim
        if self.architecture_profile == "source67":
            if route_ids is None:
                raise ValueError("source67 attention requires original route_ids")
            q_weight, k_weight, v_weight = self.qkv.weight.split((width, kv_width, kv_width), dim=0)
            q = F.linear(x, q_weight)
            k = F.linear(x, k_weight)
            v = F.linear(bind_source67_values(x, route_ids, self.layer_index), v_weight)
        else:
            q, k, v = self.qkv(x).split((width, kv_width, kv_width), dim=-1)
        q = q.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, length, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        if self.architecture_profile == "source67":
            q, k = self.q_norm(q), self.k_norm(k)
        groups = self.num_heads // self.num_key_value_heads
        k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0
        )
        if self.architecture_profile == "source67":
            routes = source67_head_routes(route_ids, self.layer_index, self.num_heads)
            selection = F.one_hot(routes, num_classes=self.num_heads).sum(dim=-2)
            mask = selection.permute(0, 2, 1).unsqueeze(-1).to(y.dtype)
            y = y * mask * math.sqrt(self.num_heads / min(2, self.num_heads))
        return self.out(y.transpose(1, 2).reshape(batch, length, width))
