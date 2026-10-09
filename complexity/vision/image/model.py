"""Latent rectified-flow transformer with deterministic TR-Hash experts."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    TRHashStrategy,
)

from .config import TRHashImageConfig


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale[:, None]) + shift[:, None]


def _timestep_embedding(timesteps: torch.Tensor, width: int) -> torch.Tensor:
    half = width // 2
    frequency = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timesteps.float()[:, None] * frequency[None] * 1_000.0
    embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
    if embedding.size(-1) < width:
        embedding = F.pad(embedding, (0, width - embedding.size(-1)))
    return embedding


class _Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).reshape(batch, length, width))


class _CrossAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key_value = nn.Linear(hidden_size, 2 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, query_length, width = query.shape
        context_length = context.size(1)
        q = self.query(query).view(batch, query_length, self.num_heads, self.head_dim)
        k, v = self.key_value(context).view(
            batch, context_length, 2, self.num_heads, self.head_dim
        ).unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        mask = context_mask[:, None, None, :]
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).reshape(batch, query_length, width))


class _TextEncoder(nn.Module):
    def __init__(self, config: TRHashImageConfig):
        super().__init__()
        self.max_length = config.max_text_length
        self.token_embedding = nn.Embedding(config.vocab_size, config.text_hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.text_hidden_size)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.text_hidden_size,
            nhead=config.text_heads,
            dim_feedforward=4 * config.text_hidden_size,
            dropout=config.attention_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=config.text_layers)
        self.norm = nn.LayerNorm(config.text_hidden_size, eps=config.layer_norm_eps)
        self.projection = nn.Linear(config.text_hidden_size, config.hidden_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if token_ids.ndim != 2 or attention_mask.shape != token_ids.shape:
            raise ValueError("caption ids and mask must be matching [batch, length] tensors")
        if token_ids.size(1) > self.max_length:
            token_ids = token_ids[:, : self.max_length]
            attention_mask = attention_mask[:, : self.max_length]
        mask = attention_mask.bool().clone()
        empty = ~mask.any(dim=1)
        if empty.any():
            mask[empty, 0] = True
        x = self.token_embedding(token_ids) + self.position_embedding[:, : token_ids.size(1)]
        x = self.layers(x, src_key_padding_mask=~mask)
        x = self.projection(self.norm(x))
        x = x * mask.unsqueeze(-1).to(x.dtype)
        return x, mask


class _TRHashDiTBlock(nn.Module):
    def __init__(self, config: TRHashImageConfig, layer_index: int):
        super().__init__()
        self.eps = config.layer_norm_eps
        self.self_attention = _Attention(
            config.hidden_size, config.num_attention_heads, config.attention_dropout
        )
        self.cross_attention = _CrossAttention(
            config.hidden_size, config.num_attention_heads, config.attention_dropout
        )
        self.condition = nn.Linear(config.hidden_size, 6 * config.hidden_size)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.route_vocab_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                shared_width=config.shared_width,
                expert_width=config.expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=layer_index,
                route_seed=config.route_seed,
                precision=TRHashPrecision.BF16,
                backend=TRHashBackend.AUTO,
            )
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), eps=self.eps)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        route_ids: torch.Tensor,
    ) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.condition(condition).chunk(
            6, dim=-1
        )
        attention_input = _modulate(self._norm(x), shift_a, scale_a)
        x = x + gate_a[:, None].tanh() * self.self_attention(attention_input)
        x = x + self.cross_attention(self._norm(x), text, text_mask)
        mlp_input = _modulate(self._norm(x), shift_m, scale_m)
        x = x + gate_m[:, None].tanh() * self.mlp(mlp_input, route_ids)
        return x


class TRHashTextToImage(nn.Module):
    """Caption-conditioned latent flow model using the canonical TR-Hash engine."""

    def __init__(self, config: Optional[TRHashImageConfig] = None):
        super().__init__()
        self.config = config or TRHashImageConfig()
        self.gradient_checkpointing = False
        config = self.config
        self.text_encoder = _TextEncoder(config)
        self.latent_in = nn.Linear(config.latent_patch_features, config.hidden_size)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.image_token_count, config.hidden_size)
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.SiLU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )
        self.blocks = nn.ModuleList(
            _TRHashDiTBlock(config, layer_index=index) for index in range(config.num_layers)
        )
        self.final_condition = nn.Linear(config.hidden_size, 2 * config.hidden_size)
        self.latent_out = nn.Linear(config.hidden_size, config.latent_patch_features)
        nn.init.normal_(self.position_embedding, std=0.02)

    def num_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _patchify(self, latents: torch.Tensor) -> torch.Tensor:
        config = self.config
        expected = (config.latent_channels, config.latent_resolution, config.latent_resolution)
        if tuple(latents.shape[1:]) != expected:
            raise ValueError(f"expected latent shape [batch, {expected}], got {tuple(latents.shape)}")
        patch = config.latent_patch_size
        return (
            latents.unfold(2, patch, patch)
            .unfold(3, patch, patch)
            .permute(0, 2, 3, 1, 4, 5)
            .reshape(latents.size(0), config.image_token_count, config.latent_patch_features)
        )

    def _unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        config = self.config
        batch = patches.size(0)
        grid = config.latent_grid_size
        patch = config.latent_patch_size
        return (
            patches.view(batch, grid, grid, config.latent_channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, config.latent_channels, config.latent_resolution, config.latent_resolution)
        )

    def build_image_route_ids(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            raise ValueError("timesteps must be [batch]")
        buckets = (timesteps.float().clamp(0, 1) * self.config.time_buckets).long()
        buckets = buckets.clamp_max(self.config.time_buckets - 1)
        positions = torch.arange(self.config.image_token_count, device=timesteps.device)
        return buckets[:, None] * self.config.image_token_count + positions[None]

    def forward(
        self,
        latents: torch.Tensor,
        timesteps: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
    ) -> torch.Tensor:
        if timesteps.shape != (latents.size(0),):
            raise ValueError("one timestep is required per latent sample")
        text, text_mask = self.text_encoder(caption_ids, caption_mask)
        masked = text * text_mask.unsqueeze(-1).to(text.dtype)
        pooled_text = masked.sum(dim=1) / text_mask.sum(dim=1, keepdim=True).clamp_min(1)
        time = self.time_mlp(
            _timestep_embedding(timesteps, self.config.hidden_size).to(latents.dtype)
        )
        condition = pooled_text + time
        x = self.latent_in(self._patchify(latents)) + self.position_embedding
        route_ids = self.build_image_route_ids(timesteps)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    condition,
                    text,
                    text_mask,
                    route_ids,
                    use_reentrant=False,
                )
            else:
                x = block(x, condition, text, text_mask, route_ids)
        shift, scale = self.final_condition(condition).chunk(2, dim=-1)
        x = _modulate(F.layer_norm(x, (x.size(-1),)), shift, scale)
        return self._unpatchify(self.latent_out(x))

    def flow_matching_loss(
        self,
        clean_latents: torch.Tensor,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        batch = clean_latents.size(0)
        noise = torch.randn(
            clean_latents.shape,
            dtype=clean_latents.dtype,
            device=clean_latents.device,
            generator=generator,
        )
        timesteps = torch.rand(
            batch, dtype=clean_latents.dtype, device=clean_latents.device, generator=generator
        )
        if self.training and self.config.caption_dropout:
            dropped = torch.rand(batch, device=clean_latents.device, generator=generator)
            caption_mask = caption_mask.clone()
            caption_mask[dropped < self.config.caption_dropout] = False
        shape = (batch,) + (1,) * (clean_latents.ndim - 1)
        t = timesteps.view(shape)
        noisy = (1.0 - t) * clean_latents + t * noise
        target_velocity = noise - clean_latents
        prediction = self(noisy, timesteps, caption_ids, caption_mask)
        return F.mse_loss(prediction.float(), target_velocity.float())

    @torch.no_grad()
    def sample(
        self,
        caption_ids: torch.Tensor,
        caption_mask: torch.Tensor,
        *,
        steps: int = 30,
        guidance_scale: float = 4.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        if steps <= 0:
            raise ValueError("steps must be positive")
        batch = caption_ids.size(0)
        config = self.config
        x = torch.randn(
            batch,
            config.latent_channels,
            config.latent_resolution,
            config.latent_resolution,
            device=caption_ids.device,
            dtype=self.position_embedding.dtype,
            generator=generator,
        )
        empty_ids = torch.zeros_like(caption_ids)
        empty_mask = torch.zeros_like(caption_mask, dtype=torch.bool)
        empty_mask[:, 0] = True
        step_size = 1.0 / steps
        for index in range(steps):
            timestep = torch.full(
                (batch,), 1.0 - index / steps, device=x.device, dtype=x.dtype
            )
            conditional = self(x, timestep, caption_ids, caption_mask)
            if guidance_scale == 1.0:
                velocity = conditional
            else:
                unconditional = self(x, timestep, empty_ids, empty_mask)
                velocity = unconditional + guidance_scale * (conditional - unconditional)
            x = x - step_size * velocity
        return x
