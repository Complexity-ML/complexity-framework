"""Standalone grouped GQA and contextual Write/Read/Value attention."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype)


def _rotary(x: torch.Tensor, offset: int, theta: float) -> torch.Tensor:
    head_dim = x.shape[-1]
    half = head_dim // 2
    positions = torch.arange(offset, offset + x.shape[2], device=x.device, dtype=torch.float32)
    frequencies = theta ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angles = positions[:, None] * frequencies[None, :]
    cos = angles.cos()[None, None].to(x.dtype)
    sin = angles.sin()[None, None].to(x.dtype)
    first, second = x[..., :half], x[..., half : 2 * half]
    rotated = torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1)
    return torch.cat((rotated, x[..., 2 * half :]), dim=-1) if head_dim > 2 * half else rotated


def _sdpa(
    reads: torch.Tensor,
    writes: torch.Tensor,
    values: torch.Tensor,
    *,
    past_length: int,
    dropout_p: float,
    scale: float,
) -> torch.Tensor:
    query_length = reads.shape[2]
    key_length = writes.shape[2]
    if past_length == 0 and query_length == key_length:
        mask = None
        is_causal = True
    elif query_length == 1:
        mask = None
        is_causal = False
    else:
        query_positions = past_length + torch.arange(query_length, device=reads.device)
        key_positions = torch.arange(key_length, device=reads.device)
        mask = key_positions[None, :] <= query_positions[:, None]
        mask = mask[None, None]
        is_causal = False
    return F.scaled_dot_product_attention(
        reads,
        writes,
        values,
        attn_mask=mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
    )


class GroupedQueryAttention(nn.Module):
    """Matched GQA control with per-head Q/K RMS normalization and RoPE."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_read_heads = config.num_read_heads
        self.num_write_heads = config.num_write_heads
        self.head_dim = self.hidden_size // self.num_read_heads
        if self.hidden_size % self.num_read_heads or self.num_read_heads % self.num_write_heads:
            raise ValueError("invalid grouped-head dimensions")
        write_width = self.num_write_heads * self.head_dim
        self.key_proj = nn.Linear(self.hidden_size, write_width, bias=False)
        self.read_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, write_width, bias=False)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.read_norm = RMSNorm(self.head_dim)
        self.write_norm = RMSNorm(self.head_dim)
        self.dropout = config.attention_dropout
        self.rope_theta = config.rope_theta
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_ids: torch.Tensor | None = None,
        lexical_scale: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        del token_ids, lexical_scale
        batch, length, _ = hidden_states.shape
        write_width = self.num_write_heads * self.head_dim
        fused = torch.cat((self.key_proj.weight, self.read_proj.weight, self.value_proj.weight), dim=0)
        writes, reads, values = F.linear(hidden_states, fused).split(
            (write_width, self.hidden_size, write_width), dim=-1
        )
        reads = reads.view(batch, length, self.num_read_heads, self.head_dim).transpose(1, 2)
        writes = writes.view(batch, length, self.num_write_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch, length, self.num_write_heads, self.head_dim).transpose(1, 2)
        reads, writes = self.read_norm(reads), self.write_norm(writes)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        reads = _rotary(reads, past_length, self.rope_theta)
        writes = _rotary(writes, past_length, self.rope_theta)
        if past_key_value is not None:
            writes = torch.cat((past_key_value[0], writes), dim=2)
            values = torch.cat((past_key_value[1], values), dim=2)
        cache = (writes, values) if use_cache else None
        repeat = self.num_read_heads // self.num_write_heads
        retrieved = _sdpa(
            reads,
            writes.repeat_interleave(repeat, dim=1),
            values.repeat_interleave(repeat, dim=1),
            past_length=past_length,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        output = retrieved.transpose(1, 2).reshape(batch, length, self.hidden_size)
        return self.output_proj(output), cache


class ContextualWRVAttention(nn.Module):
    """Contextual W/R/V attention using PyTorch SDPA/FlashAttention dispatch."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_read_heads = config.num_read_heads
        self.num_write_heads = config.num_write_heads
        self.head_dim = self.hidden_size // self.num_read_heads
        if self.hidden_size % self.num_read_heads or self.num_read_heads % self.num_write_heads:
            raise ValueError("invalid grouped-head dimensions")
        write_width = self.num_write_heads * self.head_dim
        self.read_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.write_context_proj = nn.Linear(self.hidden_size, write_width, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, write_width, bias=False)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.read_norm = RMSNorm(self.head_dim)
        self.write_norm = RMSNorm(self.head_dim)
        self.lexical_forge = nn.Linear(config.lexical_object_rank, write_width, bias=False)
        self.lexical_gate = nn.Parameter(torch.zeros(self.num_write_heads))
        self.lexical_write_residual = config.lexical_write_residual
        self.use_read_write_norm = config.use_read_write_norm
        if not self.lexical_write_residual:
            self.lexical_gate.requires_grad_(False)
        if not self.use_read_write_norm:
            self.read_norm.requires_grad_(False)
            self.write_norm.requires_grad_(False)
        self.dropout = config.attention_dropout
        self.rope_theta = config.rope_theta
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _lexical_writes(self, token_ids: torch.Tensor, lexical_scale: torch.Tensor | None) -> torch.Tensor:
        width = self.num_write_heads * self.head_dim
        dimensions = torch.arange(1, width + 1, device=token_ids.device, dtype=torch.float64)
        phases = (token_ids.to(torch.float64)[..., None] + 1.0) * torch.pi * torch.sqrt(dimensions)
        writes = torch.sin(phases).float().view(*token_ids.shape, self.num_write_heads, self.head_dim)
        if lexical_scale is not None:
            learned = self.lexical_forge(lexical_scale).view_as(writes)
            writes = writes + learned.float()
        return F.normalize(writes, dim=-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        token_ids: torch.Tensor,
        lexical_scale: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, length, _ = hidden_states.shape
        write_width = self.num_write_heads * self.head_dim
        fused = torch.cat(
            (self.read_proj.weight, self.write_context_proj.weight, self.value_proj.weight), dim=0
        )
        reads, writes, values = F.linear(hidden_states, fused).split(
            (self.hidden_size, write_width, write_width), dim=-1
        )
        reads = reads.view(batch, length, self.num_read_heads, self.head_dim).transpose(1, 2)
        writes = writes.view(batch, length, self.num_write_heads, self.head_dim)
        values = values.view(batch, length, self.num_write_heads, self.head_dim).transpose(1, 2)
        if self.lexical_write_residual:
            lexical = self._lexical_writes(token_ids, lexical_scale)
            writes = (
                writes.float()
                + torch.tanh(self.lexical_gate.float())[None, None, :, None] * lexical
            ).to(hidden_states.dtype)
        writes = writes.transpose(1, 2)
        if self.use_read_write_norm:
            reads, writes = self.read_norm(reads), self.write_norm(writes)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        reads = _rotary(reads, past_length, self.rope_theta)
        writes = _rotary(writes, past_length, self.rope_theta)
        if past_key_value is not None:
            writes = torch.cat((past_key_value[0], writes), dim=2)
            values = torch.cat((past_key_value[1], values), dim=2)
        cache = (writes, values) if use_cache else None
        repeat = self.num_read_heads // self.num_write_heads
        retrieved = _sdpa(
            reads,
            writes.repeat_interleave(repeat, dim=1),
            values.repeat_interleave(repeat, dim=1),
            past_length=past_length,
            dropout_p=self.dropout if self.training else 0.0,
            scale=self.scale,
        )
        output = retrieved.transpose(1, 2).reshape(batch, length, self.hidden_size)
        return self.output_proj(output), cache
