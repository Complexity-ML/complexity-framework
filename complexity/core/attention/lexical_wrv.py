"""Lexically addressed W/R/V attention executed by PyTorch SDPA."""

import math
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils.device import sdpa_kernel_context
from ..registry import register_attention
from ..normalization.norms import RMSNorm
from .base import AttentionBase, AttentionConfig


@register_attention("lexical_wrv")
class LexicalWRVAttention(AttentionBase):
    """Multi-read causal attention with shared contextual W/V heads."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        if not config.use_qk_norm:
            raise ValueError("lexical_wrv requires per-head read/write RMSNorm")
        if not config.use_sdpa:
            raise ValueError("lexical_wrv requires PyTorch SDPA")
        if config.sliding_window is not None:
            raise ValueError("lexical_wrv does not support sliding_window")
        if config.rope_type not in {"standard", "rope"}:
            raise ValueError("lexical_wrv supports only standard RoPE")
        self.num_read_heads = int(config.num_attention_heads)
        self.num_write_heads = int(config.num_key_value_heads)
        if self.hidden_size % self.num_read_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_read_heads % self.num_write_heads != 0:
            raise ValueError("read heads must be divisible by write heads")
        self.head_dim = self.hidden_size // self.num_read_heads
        self.read_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        write_width = self.num_write_heads * self.head_dim
        self.write_context_proj = nn.Linear(self.hidden_size, write_width, bias=False)
        self.value_proj = nn.Linear(self.hidden_size, write_width, bias=False)
        self.output_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.read_norm = RMSNorm(self.head_dim, eps=1e-6)
        self.write_norm = RMSNorm(self.head_dim, eps=1e-6)

        self.lexical_forge = nn.Linear(
            int(config.lexical_object_rank), write_width, bias=False
        )
        self.lexical_gate = nn.Parameter(torch.zeros(self.num_write_heads))
        self.attention_dropout = float(config.attention_dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.rope_theta = float(config.rope_theta)


    def _apply_rotary(
        self, tensor: torch.Tensor, position_offset: int
    ) -> torch.Tensor:
        half = self.head_dim // 2
        positions = torch.arange(
            position_offset,
            position_offset + tensor.shape[2],
            device=tensor.device,
            dtype=torch.float32,
        )
        frequencies = 1.0 / (
            self.rope_theta
            ** (torch.arange(half, device=tensor.device, dtype=torch.float32) / half)
        )
        angles = positions[:, None] * frequencies[None, :]
        cos = angles.cos()[None, None].to(tensor.dtype)
        sin = angles.sin()[None, None].to(tensor.dtype)
        first, second = tensor[..., :half], tensor[..., half : 2 * half]
        rotated = torch.cat(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        )
        if self.head_dim > 2 * half:
            rotated = torch.cat((rotated, tensor[..., 2 * half :]), dim=-1)
        return rotated

    def lexical_base_writes(self, token_ids: torch.Tensor) -> torch.Tensor:
        write_width = self.num_write_heads * self.head_dim
        dimensions = torch.arange(
            1, write_width + 1, device=token_ids.device, dtype=torch.float64
        )
        phases = (
            (token_ids.to(torch.float64)[..., None] + 1.0)
            * torch.pi
            * torch.sqrt(dimensions)
        )
        return torch.sin(phases).float().view(
            *token_ids.shape, self.num_write_heads, self.head_dim
        )

    def _lexical_writes(
        self,
        token_ids: torch.Tensor,
        lexical_token_scale_values: Optional[torch.Tensor] = None,
        lexical_base_writes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        writes = (
            lexical_base_writes
            if lexical_base_writes is not None
            else self.lexical_base_writes(token_ids)
        )
        if lexical_token_scale_values is not None:
            learned = self.lexical_forge(lexical_token_scale_values).view(
                *token_ids.shape, self.num_write_heads, self.head_dim
            )
            writes = writes + learned.float()
        return F.normalize(writes, dim=-1)

    def _compose_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        batch_size: int,
        query_length: int,
        key_length: int,
        past_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], bool]:
        if attention_mask is None and query_length == key_length:
            return None, True
        if attention_mask is None and query_length == 1:
            return None, False

        query_positions = past_length + torch.arange(query_length, device=device)
        key_positions = torch.arange(key_length, device=device)
        causal = key_positions[None, :] <= query_positions[:, None]
        causal = causal[None, None, :, :]
        if attention_mask is None:
            return causal, False

        mask = attention_mask.to(device=device)
        if mask.ndim == 2:
            if mask.shape[0] != batch_size:
                raise ValueError("attention_mask batch size does not match hidden states")
            if mask.shape[1] == query_length and past_length:
                prefix_shape = (batch_size, past_length)
                if mask.dtype == torch.bool:
                    prefix = torch.ones(prefix_shape, device=device, dtype=torch.bool)
                else:
                    prefix = torch.zeros(prefix_shape, device=device, dtype=mask.dtype)
                mask = torch.cat((prefix, mask), dim=-1)
            if mask.shape[1] != key_length:
                raise ValueError("attention_mask key length does not match W/V cache")
            mask = mask[:, None, None, :]
        elif mask.ndim not in {3, 4}:
            raise ValueError("attention_mask must have 2, 3, or 4 dimensions")
        elif mask.shape[-1] != key_length:
            raise ValueError("attention_mask key length does not match W/V cache")

        if mask.dtype == torch.bool:
            return mask & causal, False
        causal_bias = torch.zeros(
            causal.shape, device=device, dtype=dtype
        ).masked_fill(~causal, float("-inf"))
        return mask.to(dtype=dtype) + causal_bias, False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        lexical_token_scale_values: Optional[torch.Tensor] = None,
        lexical_base_writes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> tuple[
        torch.Tensor, Optional[tuple[torch.Tensor, torch.Tensor]]
    ]:
        del kwargs
        if token_ids is None:
            raise ValueError("token_ids are required for lexical W/R/V attention")
        batch_size, sequence_length, _ = hidden_states.shape
        lexical = self._lexical_writes(
            token_ids, lexical_token_scale_values, lexical_base_writes
        )
        contextual_write = self.write_context_proj(hidden_states).view(
            batch_size, sequence_length, self.num_write_heads, self.head_dim
        )
        writes = (
            contextual_write.float()
            + torch.tanh(self.lexical_gate.float())[None, None, :, None] * lexical
        ).to(hidden_states.dtype)
        reads = self.read_proj(hidden_states).view(
            batch_size, sequence_length, self.num_read_heads, self.head_dim
        )
        reads = reads.to(hidden_states.dtype)
        values = self.value_proj(hidden_states).view(
            batch_size, sequence_length, self.num_write_heads, self.head_dim
        )
        writes = writes.transpose(1, 2)
        reads = reads.transpose(1, 2)
        values = values.transpose(1, 2)
        writes = self.write_norm(writes)
        reads = self.read_norm(reads)
        position_offset = 0 if past_key_value is None else past_key_value[0].shape[2]
        writes = self._apply_rotary(writes, position_offset)
        reads = self._apply_rotary(reads, position_offset)
        if past_key_value is not None:
            writes = torch.cat((past_key_value[0], writes), dim=2)
            values = torch.cat((past_key_value[1], values), dim=2)
        new_cache = (writes, values) if use_cache else None
        repeat_factor = self.num_read_heads // self.num_write_heads
        attention_writes = writes.repeat_interleave(repeat_factor, dim=1)
        attention_values = values.repeat_interleave(repeat_factor, dim=1)
        dropout_p = self.attention_dropout if self.training else 0.0
        attention_mask, is_causal = self._compose_attention_mask(
            attention_mask,
            batch_size=batch_size,
            query_length=reads.shape[2],
            key_length=writes.shape[2],
            past_length=position_offset,
            device=reads.device,
            dtype=reads.dtype,
        )
        with sdpa_kernel_context():
            retrieved = F.scaled_dot_product_attention(
                reads,
                attention_writes,
                attention_values,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
                scale=self.scale,
            )
        retrieved = retrieved.transpose(1, 2).reshape(
            batch_size, sequence_length, self.hidden_size
        )
        output = self.output_proj(retrieved)
        return output, new_cache
