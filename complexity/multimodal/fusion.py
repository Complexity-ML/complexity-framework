"""
Multi-modal fusion mechanisms.

Implements various fusion strategies:
- Cross-attention fusion
- Gated fusion
- Concatenation with projection
- Perceiver-style resampling

v2: FusionTokenRoutedMLP — query-position routing (pos % num_experts).
    CrossAttentionBlock and PerceiverResampler use it in their MLP step.
    Same fused BMM pattern as TokenRoutedMLPParallel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
from dataclasses import dataclass
import math


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_layers: int = 2
    dropout: float = 0.1
    num_latents: int = 64  # For Perceiver
    layer_norm_eps: float = 1e-6
    # Token-routed MLP: 1 = standard, >1 = query-position routing
    num_experts: int = 4


# =============================================================================
# Token-Routed MLP (query-position routing)
# =============================================================================

class FusionTokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP for fusion query tokens.

    Routing key: query position in the output sequence.
    Expert assignment: pos % num_experts (computed on-the-fly).

    For PerceiverResampler the query length equals num_latents (fixed);
    for CrossAttentionFusion it equals the text/query sequence length
    (variable but deterministic). Both paths are fullgraph=True safe.

    Fused BMM: gate+up → SwiGLU → down.
    """

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.expert_intermediate_size = (config.hidden_size * 4) // config.num_experts

        self.gate_up_proj = nn.Parameter(
            torch.randn(config.num_experts, config.hidden_size, self.expert_intermediate_size * 2) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(config.num_experts, self.expert_intermediate_size, config.hidden_size) * 0.02
        )

    def forward(
        self,
        x: torch.Tensor,                             # [B, N, H]
        expert_ids: Optional[torch.Tensor] = None,   # [N] or None
    ) -> torch.Tensor:
        B, N, H = x.shape

        if expert_ids is None:
            expert_ids = torch.arange(N, device=x.device) % self.num_experts

        flat = x.view(B * N, H)
        eids = expert_ids.unsqueeze(0).expand(B, -1).reshape(B * N)

        gu_w = self.gate_up_proj[eids]
        down_w = self.down_proj[eids]

        gu = torch.bmm(flat.unsqueeze(1), gu_w).squeeze(1)
        gate, up = gu.split(self.expert_intermediate_size, dim=-1)
        inter = F.silu(gate) * up
        out = torch.bmm(inter.unsqueeze(1), down_w).squeeze(1)

        return out.view(B, N, H)


class FusionPlainMLP(nn.Module):
    """Standard MLP block (fallback when num_experts == 1)."""

    def __init__(self, config: FusionConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, _expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.net(x)


# =============================================================================
# Cross-attention
# =============================================================================

class CrossAttention(nn.Module):
    """Cross-attention between two modalities."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: [batch, q_len, hidden]
            key_value: [batch, kv_len, hidden]
            attention_mask: Optional mask

        Returns:
            [batch, q_len, hidden]
        """
        batch_size, q_len, _ = query.shape
        kv_len = key_value.size(1)

        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, -1)
        return self.out_proj(attn_output)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with token-routed feedforward."""

    def __init__(self, config: FusionConfig):
        super().__init__()

        self.cross_attn = CrossAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.dropout,
        )

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        if config.num_experts > 1:
            self.mlp: nn.Module = FusionTokenRoutedMLP(config)
        else:
            self.mlp = FusionPlainMLP(config)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention with residual
        residual = query
        query = self.norm1(query)
        query = self.cross_attn(query, key_value, attention_mask)
        query = residual + query

        # Token-routed MLP with residual
        residual = query
        query = self.norm2(query)
        query = self.mlp(query, expert_ids)
        query = residual + query

        return query


# =============================================================================
# Fusion modules
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """
    Fuse modalities using cross-attention.

    Text attends to image/audio features.
    Expert IDs computed on-the-fly from query length.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_experts: int = 4,
    ):
        super().__init__()

        config = FusionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_experts=num_experts,
        )

        self.layers = nn.ModuleList([
            CrossAttentionBlock(config)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        text_features: torch.Tensor,
        other_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            text_features: [batch, text_len, hidden]
            other_features: [batch, other_len, hidden]
            attention_mask: Optional attention mask

        Returns:
            Fused features [batch, text_len, hidden]
        """
        fused = text_features

        for layer in self.layers:
            fused = layer(fused, other_features, attention_mask)

        return self.norm(fused)


class GatedFusion(nn.Module):
    """
    Gated fusion of multiple modalities.

    Learns to weight contributions from each modality.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_modalities: int = 2,
    ):
        super().__init__()

        self.num_modalities = num_modalities

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * num_modalities, hidden_size),
                nn.Sigmoid(),
            )
            for _ in range(num_modalities)
        ])

        self.proj = nn.Linear(hidden_size * num_modalities, hidden_size)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *features: Variable number of feature tensors [batch, seq, hidden]

        Returns:
            Fused features [batch, seq, hidden]
        """
        assert len(features) == self.num_modalities

        concat = torch.cat(features, dim=-1)

        gated_features = []
        for feat, gate in zip(features, self.gates):
            gated_features.append(gate(concat) * feat)

        return self.proj(torch.cat(gated_features, dim=-1))


class ConcatProjection(nn.Module):
    """
    Simple concatenation and projection fusion.

    Concatenates features and projects back to hidden dimension.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_modalities: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        pooled = []
        for feat in features:
            pooled.append(feat.mean(dim=1) if feat.dim() == 3 else feat)
        return self.proj(torch.cat(pooled, dim=-1))


class PerceiverResampler(nn.Module):
    """
    Perceiver-style resampler for multi-modal fusion.

    Uses learned latent queries to resample variable-length
    features to fixed-length representations.

    Latent expert IDs precomputed: latent_i → expert i % num_experts.

    Reference: Perceiver IO (https://arxiv.org/abs/2107.14795)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_latents: int = 64,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_experts: int = 4,
    ):
        super().__init__()

        self.num_latents = num_latents

        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)

        config = FusionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_latents=num_latents,
            num_experts=num_experts,
        )

        self.layers = nn.ModuleList([
            CrossAttentionBlock(config)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        # Precompute expert_ids for latents (fixed count)
        expert_ids = torch.arange(num_latents) % num_experts
        self.register_buffer("expert_ids", expert_ids)  # [num_latents]

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: [batch, seq_len, hidden]
            attention_mask: Optional mask

        Returns:
            Resampled features [batch, num_latents, hidden]
        """
        batch_size = features.size(0)
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        for layer in self.layers:
            latents = layer(latents, features, attention_mask, self.expert_ids)

        return self.norm(latents)


class MultimodalFusion(nn.Module):
    """
    Unified multi-modal fusion module.

    Supports multiple fusion strategies.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        fusion_type: str = "cross_attention",
        num_latents: int = 64,
        dropout: float = 0.1,
        num_experts: int = 4,
    ):
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                num_experts=num_experts,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(hidden_size=hidden_size)
        elif fusion_type == "concat":
            self.fusion = ConcatProjection(hidden_size=hidden_size, dropout=dropout)
        elif fusion_type == "perceiver":
            self.fusion = PerceiverResampler(
                hidden_size=hidden_size,
                num_latents=num_latents,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                num_experts=num_experts,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        text_features: torch.Tensor,
        other_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.fusion_type == "cross_attention":
            return self.fusion(text_features, other_features, attention_mask)
        elif self.fusion_type == "gated":
            text_pooled = text_features.mean(dim=1, keepdim=True) if text_features.dim() == 3 else text_features.unsqueeze(1)
            other_pooled = other_features.mean(dim=1, keepdim=True) if other_features.dim() == 3 else other_features.unsqueeze(1)
            return self.fusion(text_pooled, other_pooled).squeeze(1)
        elif self.fusion_type == "concat":
            return self.fusion(text_features, other_features)
        elif self.fusion_type == "perceiver":
            return self.fusion(other_features, attention_mask)


class VisionLanguageConnector(nn.Module):
    """
    Connect vision encoder to language model.

    Used in vision-language models like LLaVA.
    """

    def __init__(
        self,
        vision_hidden_size: int = 1024,
        language_hidden_size: int = 4096,
        num_tokens: int = 576,  # (384/14)^2 for SigLIP
        connector_type: str = "mlp",
    ):
        super().__init__()

        self.num_tokens = num_tokens
        self.connector_type = connector_type

        if connector_type == "mlp":
            self.connector = nn.Sequential(
                nn.Linear(vision_hidden_size, language_hidden_size),
                nn.GELU(),
                nn.Linear(language_hidden_size, language_hidden_size),
            )
        elif connector_type == "resampler":
            self.connector = PerceiverResampler(
                hidden_size=vision_hidden_size,
                num_latents=64,
                num_heads=16,
                num_layers=2,
            )
            self.proj = nn.Linear(vision_hidden_size, language_hidden_size)
        else:
            raise ValueError(f"Unknown connector type: {connector_type}")

    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vision_features: [batch, num_patches, vision_hidden]

        Returns:
            Language-compatible features [batch, num_tokens, language_hidden]
        """
        if self.connector_type == "mlp":
            return self.connector(vision_features)
        elif self.connector_type == "resampler":
            resampled = self.connector(vision_features)
            return self.proj(resampled)
