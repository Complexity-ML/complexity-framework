"""Canonical TR-HASH modality blocks and task-specific connector components. Original position IDs are preserved for head/value routing and contextual expert assignment."""

from __future__ import annotations
from dataclasses import asdict, dataclass, fields, field
from typing import Optional
import torch
import torch.nn as nn
from complexity.tr_hash.attention import DomainSelfAttention, make_domain_mlp, resolve_domain_heads


@dataclass
class VideoConfig:
    """Configuration for the video encoder."""

    image_size: int = 224
    patch_size: int = 16
    num_frames: int = 16
    temporal_patch_size: int = 2
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    num_key_value_heads: int | None = field(default=None, kw_only=True)
    intermediate_size: int | None = None
    shared_intermediate_size: Optional[int] = None
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    num_channels: int = 3
    layer_norm_eps: float = 1e-06
    num_experts: int = 4

    def __post_init__(self):
        resolve_domain_heads(self)
        if self.num_experts != 4:
            raise ValueError("TR-HASH requires exactly four experts")
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size
        if self.shared_intermediate_size is None:
            self.shared_intermediate_size = 2 * self.hidden_size

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown multimodal config fields: {sorted(unknown)}")
        return cls(**values)

    @property
    def num_spatial_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_temporal_patches(self) -> int:
        return self.num_frames // self.temporal_patch_size

    @property
    def num_patches(self) -> int:
        return self.num_spatial_patches * self.num_temporal_patches


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
        x = self.projection(video)
        B, H, T_p, h_p, w_p = x.shape
        x = x.flatten(3).permute(0, 2, 3, 1)
        return x.reshape(B, T_p * h_p * w_p, H)


class VideoAttention(nn.Module):
    """Multi-head self-attention (shared by spatial and temporal ops)."""

    def __init__(self, config: VideoConfig, layer_index: int = 0):
        super().__init__()
        self.source_attention = DomainSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout_prob,
            num_key_value_heads=config.num_key_value_heads,
            layer_index=layer_index,
        )

    def forward(self, x: torch.Tensor, route_ids=None) -> torch.Tensor:
        return self.source_attention(x, route_ids)


class VideoTokenRoutedMLP(nn.Module):
    """Shared SwiGLU plus two of four experts. Routes use original position IDs and their two preceding IDs; no learned gate."""

    def __init__(self, config: VideoConfig, layer_index: int = 0):
        super().__init__()
        self.engine = make_domain_mlp(
            hidden_size=config.hidden_size,
            vocab_size=config.num_patches,
            num_experts=4,
            top_k=2,
            shared_width=config.shared_intermediate_size,
            expert_width=config.intermediate_size // 4,
            layer_index=layer_index,
        )

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.arange(x.size(1), device=x.device)
        route_ids = (
            expert_ids.unsqueeze(0).expand(x.size(0), -1) if expert_ids.ndim == 1 else expert_ids
        )
        return self.engine(x, route_ids)


class SpatioTemporalBlock(nn.Module):
    """
    ViViT Factorised Encoder block with contextual TR-HASH MLP.

    1. Spatial self-attention  — patches within each frame.
    2. Temporal self-attention — frames at each spatial position.
    3. Token-Routed MLP.
    """

    def __init__(self, config: VideoConfig, layer_index: int = 0):
        super().__init__()
        self.spatial_attn = VideoAttention(config, layer_index)
        self.temporal_attn = VideoAttention(config, layer_index)
        self.mlp: nn.Module = VideoTokenRoutedMLP(config, layer_index)
        self.norm_spatial = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_temporal = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm_mlp = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, T: int, S: int, expert_ids: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        route_ids = expert_ids.unsqueeze(0).expand(B, -1)
        residual = x
        x_s = self.norm_spatial(x).view(B * T, S, x.shape[-1])
        x_s = self.spatial_attn(x_s, route_ids.reshape(B * T, S)).view(B, T * S, x.shape[-1])
        x = residual + x_s
        residual = x
        H = x.shape[-1]
        x_t = self.norm_temporal(x).view(B, T, S, H).permute(0, 2, 1, 3).reshape(B * S, T, H)
        x_t = self.temporal_attn(x_t, route_ids.reshape(B, T, S).permute(0, 2, 1).reshape(B * S, T))
        x_t = x_t.view(B, S, T, H).permute(0, 2, 1, 3).reshape(B, T * S, H)
        x = residual + x_t
        x = x + self.mlp(self.norm_mlp(x), expert_ids)
        return x


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
        self.spatial_pos_embed = nn.Parameter(
            torch.zeros(1, 1, config.num_spatial_patches, config.hidden_size)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.zeros(1, config.num_temporal_patches, 1, config.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.blocks = nn.ModuleList(
            [SpatioTemporalBlock(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        expert_ids = torch.arange(config.num_patches)
        self.register_buffer("expert_ids", expert_ids)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)

    def forward(self, video: torch.Tensor, output_hidden_states: bool = False) -> dict:
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
        T, S = (self.T, self.S)
        x = self.tubelet_embed(video)
        pos = self.spatial_pos_embed + self.temporal_pos_embed
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
        result = {"last_hidden_state": x, "pooler_output": x.mean(dim=1)}
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        return result


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
        num_heads: int = 4,
        num_experts: int = 4,
        output_dim: Optional[int] = None,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        config = VideoConfig(
            num_key_value_heads=num_key_value_heads,
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

    def forward(self, video: torch.Tensor, return_all_features: bool = False) -> torch.Tensor:
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
