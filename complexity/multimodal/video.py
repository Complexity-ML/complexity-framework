"""
Video encoder — tubelet embedding + factorised spatio-temporal transformer.

Architecture choices
--------------------
* TubeletEmbedding
    Conv3d patch tokeniser (VideoMAE / ViViT style).
    Kernel: (temporal_patch_size, patch_size, patch_size).
    Produces T_p * S_p tokens where T_p = num_frames / temporal_patch_size
    and S_p = (image_size / patch_size) ** 2.

* VideoTokenRoutedMLP
    Token-Routed MLP for video — same BMM pattern as TokenRoutedMLPParallel
    but routes by *spatial patch position* (position % num_experts).
    Position-based routing is precomputed → fullgraph=True safe, zero overhead.
    Each spatial position always hits the same expert → spatial specialisation.

* SpatioTemporalBlock
    Factorised attention — ViViT "Factorised Encoder" (Model 3):
    1. Spatial attention  : [B*T_p, S_p, H]
    2. Temporal attention : [B*S_p, T_p, H]
    3. VideoTokenRoutedMLP (or plain MLP if num_experts=1)

* VideoEncoder
    Thin wrapper projecting to a target hidden_size, same interface as
    VisionEncoder / AudioEncoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Config
# =============================================================================

@dataclass
class VideoConfig:
    """Configuration for the video encoder."""
    image_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    temporal_patch_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    # Token-routed MLP: 1 = standard MLP, >1 = spatial-position routing
    num_experts: int = 4

    @property
    def num_spatial_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_temporal_patches(self) -> int:
        return self.num_frames // self.temporal_patch_size

    @property
    def num_patches(self) -> int:
        return self.num_spatial_patches * self.num_temporal_patches


# =============================================================================
# Patch tokeniser
# =============================================================================

class TubeletEmbedding(nn.Module):
    """
    3-D tubelet tokeniser via Conv3d.

    Input : [B, C, T, H, W]
    Output: [B, T_p * S_p, hidden_size]
    """

    def __init__(self, config: VideoConfig):
        super().__init__()
        self.T = config.num_temporal_patches
        self.S = config.num_spatial_patches

        self.projection = nn.Conv3d(
            in_channels=config.num_channels,
            out_channels=config.hidden_size,
            kernel_size=(config.temporal_patch_size, config.patch_size, config.patch_size),
            stride=(config.temporal_patch_size, config.patch_size, config.patch_size),
        )

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W]

        Returns:
            tokens: [B, T_p * S_p, hidden_size]
        """
        x = self.projection(video)            # [B, H, T_p, h_p, w_p]
        B, H, T_p, h_p, w_p = x.shape
        x = x.flatten(3).permute(0, 2, 3, 1)  # [B, T_p, S_p, H]
        return x.reshape(B, T_p * h_p * w_p, H)


# =============================================================================
# Attention
# =============================================================================

class VideoAttention(nn.Module):
    """Multi-head self-attention (shared by spatial and temporal ops)."""

    def __init__(self, config: VideoConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        return self.proj(out)


# =============================================================================
# Token-Routed MLP (spatial-position routing)
# =============================================================================

class VideoTokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP for video tokens.

    Routing key: *spatial patch position* (s = token_id % num_spatial_patches)
    Expert assignment: s % num_experts

    This is deterministic and position-only — no content dependency.
    The position_ids tensor is precomputed once in VideoTransformer and
    reused across all layers → zero overhead, fullgraph=True safe.

    Uses the same fused BMM pattern as TokenRoutedMLPParallel:
        gate+up BMM  →  SwiGLU  →  down BMM
    """

    def __init__(self, config: VideoConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.expert_intermediate_size = config.intermediate_size // config.num_experts

        # Fused gate+up: [E, H, 2*I_e]
        self.gate_up_proj = nn.Parameter(
            torch.randn(
                config.num_experts,
                config.hidden_size,
                self.expert_intermediate_size * 2,
            ) * 0.02
        )
        # Down: [E, I_e, H]
        self.down_proj = nn.Parameter(
            torch.randn(
                config.num_experts,
                self.expert_intermediate_size,
                config.hidden_size,
            ) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,           # [B, T*S, H]
        expert_ids: torch.Tensor,  # [T*S]  precomputed, broadcast over B
    ) -> torch.Tensor:
        B, N, H = x.shape

        flat = x.view(B * N, H)
        # Expand expert_ids for batch: [T*S] → [B*T*S]
        eids = expert_ids.unsqueeze(0).expand(B, -1).reshape(B * N)

        gu_w = self.gate_up_proj[eids]   # [B*N, H, 2*I_e]
        down_w = self.down_proj[eids]    # [B*N, I_e, H]

        gu = torch.bmm(flat.unsqueeze(1), gu_w).squeeze(1)   # [B*N, 2*I_e]
        gate, up = gu.split(self.expert_intermediate_size, dim=-1)
        inter = F.silu(gate) * up                             # [B*N, I_e]
        out = torch.bmm(inter.unsqueeze(1), down_w).squeeze(1)  # [B*N, H]

        return out.view(B, N, H)


class VideoPlainMLP(nn.Module):
    """Standard feed-forward block (fallback when num_experts=1)."""

    def __init__(self, config: VideoConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor, expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(F.gelu(self.fc1(x)))))


# =============================================================================
# Factorised spatio-temporal block
# =============================================================================

class SpatioTemporalBlock(nn.Module):
    """
    ViViT Factorised Encoder block with optional token-routed MLP.

    1. Spatial self-attention  — patches within each frame.
    2. Temporal self-attention — frames at each spatial position.
    3. Token-Routed MLP (or plain MLP if num_experts == 1).
    """

    def __init__(self, config: VideoConfig):
        super().__init__()
        self.spatial_attn = VideoAttention(config)
        self.temporal_attn = VideoAttention(config)

        if config.num_experts > 1:
            self.mlp: nn.Module = VideoTokenRoutedMLP(config)
        else:
            self.mlp = VideoPlainMLP(config)

        self.norm_spatial = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_temporal = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_mlp = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,           # [B, T*S, H]
        T: int,
        S: int,
        expert_ids: torch.Tensor,  # [T*S]
    ) -> torch.Tensor:
        B = x.shape[0]

        # 1. Spatial attention
        residual = x
        x_s = self.norm_spatial(x).view(B * T, S, x.shape[-1])
        x_s = self.spatial_attn(x_s).view(B, T * S, x.shape[-1])
        x = residual + x_s

        # 2. Temporal attention
        residual = x
        H = x.shape[-1]
        x_t = self.norm_temporal(x).view(B, T, S, H).permute(0, 2, 1, 3).reshape(B * S, T, H)
        x_t = self.temporal_attn(x_t)
        x_t = x_t.view(B, S, T, H).permute(0, 2, 1, 3).reshape(B, T * S, H)
        x = residual + x_t

        # 3. Token-routed (or plain) MLP
        x = x + self.mlp(self.norm_mlp(x), expert_ids)

        return x


# =============================================================================
# Full video transformer
# =============================================================================

class VideoTransformer(nn.Module):
    """
    Full spatio-temporal video transformer.

    Input : [B, C, T, H, W]   (T = num_frames)
    Output: dict  last_hidden_state [B, T_p*S_p, H],  pooler_output [B, H]
    """

    def __init__(self, config: VideoConfig):
        super().__init__()
        self.config = config
        self.T = config.num_temporal_patches
        self.S = config.num_spatial_patches

        self.tubelet_embed = TubeletEmbedding(config)

        # Factorised positional embeddings (spatial + temporal, learnable)
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, config.num_spatial_patches, config.hidden_size)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_temporal_patches, 1, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Precompute expert_ids for the MLP: route by spatial position
        # expert_id[t*S + s] = s % num_experts  (s is the spatial patch index)
        S = config.num_spatial_patches
        spatial_ids = torch.arange(S) % config.num_experts  # [S]
        expert_ids = spatial_ids.unsqueeze(0).expand(config.num_temporal_patches, -1).reshape(-1)
        self.register_buffer("expert_ids", expert_ids)  # [T*S], long

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(
        self,
        video: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        Args:
            video: [B, C, T, H, W]
            output_hidden_states: return all block outputs

        Returns:
            dict:
                last_hidden_state : [B, T_p*S_p, H]
                pooler_output     : [B, H]
                hidden_states     : list[Tensor]  (if output_hidden_states)
        """
        T, S = self.T, self.S

        x = self.tubelet_embed(video)  # [B, T*S, H]

        # Factorised positional embedding: broadcast over T and S
        pos = (self.spatial_pos_embed + self.temporal_pos_embed)   # [1, T, S, H]
        pos = pos.reshape(1, T * S, self.config.hidden_size)
        x = self.dropout(x + pos)

        all_hidden_states = [] if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(x)
            x = block(x, T, S, self.expert_ids)

        x = self.norm(x)
        if output_hidden_states:
            all_hidden_states.append(x)

        result = {
            "last_hidden_state": x,
            "pooler_output": x.mean(dim=1),
        }
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states

        return result


# =============================================================================
# Convenience wrapper
# =============================================================================

class VideoEncoder(nn.Module):
    """
    Generic video encoder wrapper — same interface as VisionEncoder / AudioEncoder.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        temporal_patch_size: int = 2,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 4,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        config = VideoConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_experts=num_experts,
        )

        self.encoder = VideoTransformer(config)

        if output_dim is not None and output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = None

    def forward(
        self,
        video: torch.Tensor,
        return_all_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W]
            return_all_features: all patch tokens vs pooled

        Returns:
            [B, hidden_size] or [B, T_p*S_p, hidden_size]
        """
        outputs = self.encoder(video)
        features = outputs["last_hidden_state"] if return_all_features else outputs["pooler_output"]
        if self.proj is not None:
            features = self.proj(features)
        return features
