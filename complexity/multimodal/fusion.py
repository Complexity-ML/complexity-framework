"""Canonical TR-HASH modality blocks and task-specific connector components. Original position IDs are preserved for head/value routing and contextual expert assignment."""

from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.tr_hash.attention import make_domain_mlp, resolve_domain_heads


@dataclass
class FusionConfig:
    """Configuration for multi-modal fusion."""

    hidden_size: int = 768
    num_attention_heads: int = 4
    num_key_value_heads: int | None = field(default=None, kw_only=True)
    num_layers: int = 2
    dropout: float = 0.1
    num_latents: int = 64
    layer_norm_eps: float = 1e-06
    num_experts: int = 4
    shared_intermediate_size: Optional[int] = None

    def __post_init__(self):
        resolve_domain_heads(self)
        if self.num_experts != 4:
            raise ValueError("TR-HASH requires exactly four experts")
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


class FusionTokenRoutedMLP(nn.Module):
    """Shared SwiGLU plus two of four experts. Routes use original position IDs and their two preceding IDs; no learned gate."""

    def __init__(self, config: FusionConfig, layer_index: int = 0):
        super().__init__()
        self.engine = make_domain_mlp(
            hidden_size=config.hidden_size,
            vocab_size=config.num_latents,
            num_experts=4,
            top_k=2,
            shared_width=config.shared_intermediate_size,
            expert_width=config.hidden_size // 4,
            layer_index=layer_index,
        )

    def forward(self, x: torch.Tensor, expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.arange(x.size(1), device=x.device)
        route_ids = (
            expert_ids.unsqueeze(0).expand(x.size(0), -1) if expert_ids.ndim == 1 else expert_ids
        )
        return self.engine(x, route_ids)


class CrossAttention(nn.Module):
    """Cross-attention between two modalities."""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** (-0.5)
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
        q = (
            self.q_proj(query)
            .view(batch_size, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(key_value)
            .view(batch_size, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(key_value)
            .view(batch_size, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
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

    def __init__(self, config: FusionConfig, layer_index: int = 0):
        super().__init__()
        self.cross_attn = CrossAttention(
            config.hidden_size, config.num_attention_heads, config.dropout
        )
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp: nn.Module = FusionTokenRoutedMLP(config, layer_index)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        expert_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = query
        query = self.norm1(query)
        query = self.cross_attn(query, key_value, attention_mask)
        query = residual + query
        residual = query
        query = self.norm2(query)
        query = self.mlp(query, expert_ids)
        query = residual + query
        return query


class CrossAttentionFusion(nn.Module):
    """
    Fuse modalities using cross-attention.

    Text attends to image/audio features.
    Expert IDs computed on-the-fly from query length.
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_experts: int = 4,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        config = FusionConfig(
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_experts=num_experts,
        )
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(config, index) for index in range(num_layers)]
        )
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

    def __init__(self, hidden_size: int = 768, num_modalities: int = 2):
        super().__init__()
        self.num_modalities = num_modalities
        self.gates = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden_size * num_modalities, hidden_size), nn.Sigmoid())
                for _ in range(num_modalities)
            ]
        )
        self.proj = nn.Linear(hidden_size * num_modalities, hidden_size)

    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *features: Variable number of feature tensors [batch, seq, hidden]

        Returns:
            Fused features [batch, seq, hidden]
        """
        if len(features) != self.num_modalities:
            raise ValueError(
                f"GatedFusion configured for {self.num_modalities} modalities, got {len(features)}; a missing modality must be represented explicitly (e.g. a zero tensor) -- this module has no implicit missing-modality handling"
            )
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

    def __init__(self, hidden_size: int = 768, num_modalities: int = 2, dropout: float = 0.1):
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

    Original latent-position IDs feed the contextual expert MLP.

    Reference: Perceiver IO (https://arxiv.org/abs/2107.14795)
    """

    def __init__(
        self,
        hidden_size: int = 768,
        num_latents: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_experts: int = 4,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.latents = nn.Parameter(torch.randn(num_latents, hidden_size) * 0.02)
        config = FusionConfig(
            num_key_value_heads=num_key_value_heads,
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            num_latents=num_latents,
            num_experts=num_experts,
        )
        self.layers = nn.ModuleList(
            [CrossAttentionBlock(config, index) for index in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        expert_ids = torch.arange(num_latents)
        self.register_buffer("expert_ids", expert_ids)

    def forward(
        self, features: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
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
        num_heads: int = 4,
        num_layers: int = 2,
        fusion_type: str = "cross_attention",
        num_latents: int = 64,
        dropout: float = 0.1,
        num_experts: int = 4,
        *,
        num_key_value_heads: int | None = None,
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
                num_key_value_heads=num_key_value_heads,
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
                num_key_value_heads=num_key_value_heads,
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
            text_pooled = (
                text_features.mean(dim=1, keepdim=True)
                if text_features.dim() == 3
                else text_features.unsqueeze(1)
            )
            other_pooled = (
                other_features.mean(dim=1, keepdim=True)
                if other_features.dim() == 3
                else other_features.unsqueeze(1)
            )
            return self.fusion(text_pooled, other_pooled).squeeze(1)
        elif self.fusion_type == "concat":
            return self.fusion(text_features, other_features)
        elif self.fusion_type == "perceiver":
            return self.fusion(other_features, attention_mask)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")


class VisionLanguageConnector(nn.Module):
    """
    Connect vision encoder to language model.

    Used in vision-language models like LLaVA.
    """

    def __init__(
        self,
        vision_hidden_size: int = 1024,
        language_hidden_size: int = 4096,
        num_tokens: int = 576,
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
                hidden_size=vision_hidden_size, num_latents=64, num_heads=16, num_layers=2
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
