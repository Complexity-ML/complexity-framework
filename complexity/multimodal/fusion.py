"""
Multi-modal fusion mechanisms.

Implements various fusion strategies:
- Cross-attention fusion
- Gated fusion
- Concatenation with projection
- Perceiver-style resampling
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
            query: Query tensor [batch, q_len, hidden]
            key_value: Key/Value tensor [batch, kv_len, hidden]
            attention_mask: Optional mask

        Returns:
            Cross-attended output [batch, q_len, hidden]
        """
        batch_size, q_len, _ = query.shape
        kv_len = key_value.size(1)

        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key_value)
        v = self.v_proj(key_value)

        # Reshape for multi-head attention
        q = q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, q_len, -1)

        return self.out_proj(attn_output)


class CrossAttentionBlock(nn.Module):
    """Cross-attention block with feedforward."""

    def __init__(self, config: FusionConfig):
        super().__init__()

        self.cross_attn = CrossAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.dropout,
        )

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Cross-attention with residual
        residual = query
        query = self.norm1(query)
        query = self.cross_attn(query, key_value, attention_mask)
        query = residual + query

        # MLP with residual
        residual = query
        query = self.norm2(query)
        query = self.mlp(query)
        query = residual + query

        return query


class CrossAttentionFusion(nn.Module):
    """
    Fuse modalities using cross-attention.

    Text attends to image/audio features.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        config = FusionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
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
            text_features: Text features [batch, text_len, hidden]
            other_features: Image/audio features [batch, other_len, hidden]
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
        """
        Args:
            hidden_size: Hidden dimension
            num_modalities: Number of modalities to fuse
        """
        super().__init__()

        self.num_modalities = num_modalities

        # Gate for each modality
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size * num_modalities, hidden_size),
                nn.Sigmoid(),
            )
            for _ in range(num_modalities)
        ])

        # Output projection
        self.proj = nn.Linear(hidden_size * num_modalities, hidden_size)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *features: Variable number of feature tensors [batch, seq, hidden]

        Returns:
            Fused features [batch, seq, hidden]
        """
        assert len(features) == self.num_modalities

        # Concatenate for gate computation
        concat = torch.cat(features, dim=-1)

        # Compute gates
        gated_features = []
        for i, (feat, gate) in enumerate(zip(features, self.gates)):
            g = gate(concat)
            gated_features.append(g * feat)

        # Combine and project
        combined = torch.cat(gated_features, dim=-1)
        return self.proj(combined)


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
        """
        Args:
            *features: Feature tensors to concatenate

        Returns:
            Projected features [batch, seq, hidden]
        """
        # Handle different sequence lengths by pooling
        pooled = []
        for feat in features:
            if feat.dim() == 3:
                pooled.append(feat.mean(dim=1))  # Pool over sequence
            else:
                pooled.append(feat)

        concat = torch.cat(pooled, dim=-1)
        return self.proj(concat)


class PerceiverResampler(nn.Module):
    """
    Perceiver-style resampler for multi-modal fusion.

    Uses learned latent queries to resample variable-length
    features to fixed-length representations.

    Reference: Perceiver IO (https://arxiv.org/abs/2107.14795)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_latents: int = 64,
        num_heads: int = 12,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_latents: Number of latent query vectors
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
        """
        super().__init__()

        self.num_latents = num_latents

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)

        config = FusionConfig(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_latents=num_latents,
        )

        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionBlock(config)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            features: Input features [batch, seq_len, hidden]
            attention_mask: Optional attention mask

        Returns:
            Resampled features [batch, num_latents, hidden]
        """
        batch_size = features.size(0)

        # Expand latents for batch
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        # Cross-attend to input features
        for layer in self.layers:
            latents = layer(latents, features, attention_mask)

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
    ):
        """
        Args:
            hidden_size: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of layers
            fusion_type: Type of fusion (cross_attention, gated, concat, perceiver)
            num_latents: Number of latents for perceiver
            dropout: Dropout rate
        """
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                hidden_size=hidden_size,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
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
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        text_features: torch.Tensor,
        other_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse text features with another modality.

        Args:
            text_features: Text features [batch, text_len, hidden]
            other_features: Other modality features [batch, other_len, hidden]
            attention_mask: Optional attention mask

        Returns:
            Fused features
        """
        if self.fusion_type == "cross_attention":
            return self.fusion(text_features, other_features, attention_mask)
        elif self.fusion_type == "gated":
            # Pool text features to match dimensions
            if text_features.dim() == 3:
                text_pooled = text_features.mean(dim=1, keepdim=True)
            else:
                text_pooled = text_features.unsqueeze(1)

            if other_features.dim() == 3:
                other_pooled = other_features.mean(dim=1, keepdim=True)
            else:
                other_pooled = other_features.unsqueeze(1)

            return self.fusion(text_pooled, other_pooled).squeeze(1)
        elif self.fusion_type == "concat":
            return self.fusion(text_features, other_features)
        elif self.fusion_type == "perceiver":
            # Resample other modality features
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
        """
        Args:
            vision_hidden_size: Vision encoder hidden size
            language_hidden_size: Language model hidden size
            num_tokens: Number of vision tokens
            connector_type: Type of connector (mlp, resampler)
        """
        super().__init__()

        self.num_tokens = num_tokens
        self.connector_type = connector_type

        if connector_type == "mlp":
            # Simple MLP projection (LLaVA 1.5 style)
            self.connector = nn.Sequential(
                nn.Linear(vision_hidden_size, language_hidden_size),
                nn.GELU(),
                nn.Linear(language_hidden_size, language_hidden_size),
            )
        elif connector_type == "resampler":
            # Perceiver resampler
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
