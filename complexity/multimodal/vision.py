"""Canonical TR-HASH modality blocks and task-specific connector components. Original position IDs are preserved for head/value routing and contextual expert assignment."""

import math
from dataclasses import asdict, dataclass, fields, field
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from complexity.tr_hash.attention import DomainSelfAttention, make_domain_mlp, resolve_domain_heads


@dataclass
class VisionConfig:
    """Configuration for vision encoder."""

    image_size: int = 224
    patch_size: int = 16
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
    use_class_token: bool = True
    use_2d_pos_embed: bool = True
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


class PatchEmbedding(nn.Module):
    """
    Convert image to patch embeddings.

    Splits image into non-overlapping patches and projects to hidden dimension.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.projection = nn.Conv2d(
            num_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, channels, height, width]

        Returns:
            Patch embeddings [batch, num_patches, hidden_size]
        """
        x = self.projection(pixel_values)
        x = x.flatten(2).transpose(1, 2)
        return x


class VisionAttention(nn.Module):
    """Multi-head self-attention for vision."""

    def __init__(self, config: VisionConfig, layer_index: int = 0):
        super().__init__()
        self.source_attention = DomainSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout_prob,
            num_key_value_heads=config.num_key_value_heads,
            layer_index=layer_index,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        route_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if route_ids is None:
            route_ids = torch.arange(hidden_states.size(1), device=hidden_states.device)
        if route_ids.ndim == 1:
            route_ids = route_ids.unsqueeze(0).expand(hidden_states.size(0), -1)
        return self.source_attention(hidden_states, route_ids, attention_mask)


class VisionTokenRoutedMLP(nn.Module):
    """Shared SwiGLU plus two of four experts. Routes use original position IDs and their two preceding IDs; no learned gate."""

    def __init__(self, config: VisionConfig, layer_index: int = 0):
        super().__init__()
        self.engine = make_domain_mlp(
            hidden_size=config.hidden_size,
            vocab_size=(config.image_size // config.patch_size) ** 2 + int(config.use_class_token),
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


class VisionTransformerBlock(nn.Module):
    """Single transformer block for vision (with contextual TR-HASH MLP)."""

    def __init__(self, config: VisionConfig, layer_index: int = 0):
        super().__init__()
        self.attention = VisionAttention(config, layer_index)
        self.mlp: nn.Module = VisionTokenRoutedMLP(config, layer_index)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask, expert_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, expert_ids)
        hidden_states = residual + hidden_states
        return hidden_states


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) with token-routed MLP.

    Original raster positions, with a leading CLS identity when present,
    feed attention and contextual expert routing without modulo reduction.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            hidden_size=config.hidden_size,
        )
        num_patches = self.patch_embed.num_patches
        if config.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        num_positions = num_patches + (1 if config.use_class_token else 0)
        self.position_embedding = nn.Parameter(torch.zeros(1, num_positions, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.blocks = nn.ModuleList(
            [VisionTransformerBlock(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        expert_ids = torch.arange(num_positions)
        self.register_buffer("expert_ids", expert_ids)
        self._init_weights()

    def _init_weights(self):
        if hasattr(self, "cls_token"):
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> dict:
        """
        Args:
            pixel_values: [batch, channels, height, width]
            attention_mask: Optional attention mask
            output_hidden_states: Return all hidden states

        Returns:
            Dictionary with last_hidden_state and optionally all hidden_states
        """
        batch_size = pixel_values.size(0)
        hidden_states = self.patch_embed(pixel_values)
        if self.config.use_class_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)
        hidden_states = hidden_states + self.position_embedding
        hidden_states = self.dropout(hidden_states)
        all_hidden_states = [] if output_hidden_states else None
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = block(hidden_states, self.expert_ids, attention_mask)
        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        if self.config.use_class_token:
            pooled_output = hidden_states[:, 0]
        else:
            pooled_output = hidden_states.mean(dim=1)
        result = {"last_hidden_state": hidden_states, "pooler_output": pooled_output}
        if output_hidden_states:
            result["hidden_states"] = all_hidden_states
        return result


class VisionEncoder(nn.Module):
    """
    Generic vision encoder wrapper.

    Can be used with different backbone configurations.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 4,
        num_experts: int = 4,
        output_dim: Optional[int] = None,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        config = VisionConfig(
            num_key_value_heads=num_key_value_heads,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_experts=num_experts,
        )
        self.encoder = VisionTransformer(config)
        if output_dim is not None and output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = None

    def forward(
        self, pixel_values: torch.Tensor, return_all_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, channels, height, width]
            return_all_features: Return all patch features instead of pooled

        Returns:
            Image features [batch, hidden_size] or [batch, num_patches, hidden_size]
        """
        outputs = self.encoder(pixel_values, output_hidden_states=False)
        if return_all_features:
            features = outputs["last_hidden_state"]
        else:
            features = outputs["pooler_output"]
        if self.proj is not None:
            features = self.proj(features)
        return features


class CLIPVisionEncoder(nn.Module):
    """
    CLIP-style vision encoder.

    Optimized for vision-language alignment.
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 14,
        hidden_size: int = 1024,
        num_layers: int = 24,
        num_heads: int = 4,
        output_dim: int = 768,
        num_experts: int = 4,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        config = VisionConfig(
            num_key_value_heads=num_key_value_heads,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size,
            num_experts=num_experts,
        )
        self.encoder = VisionTransformer(config)
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(self, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        outputs = self.encoder(pixel_values)
        features = outputs["pooler_output"]
        features = self.proj(features)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def get_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100)


class SigLIPEncoder(nn.Module):
    """
    SigLIP-style vision encoder.

    Uses sigmoid loss instead of softmax for better scaling.
    """

    def __init__(
        self,
        image_size: int = 384,
        patch_size: int = 14,
        hidden_size: int = 1152,
        num_layers: int = 27,
        num_heads: int = 4,
        output_dim: int = 1152,
        num_experts: int = 4,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        config = VisionConfig(
            num_key_value_heads=num_key_value_heads,
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size,
            use_class_token=False,
            num_experts=num_experts,
        )
        self.encoder = VisionTransformer(config)
        if output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = None
        self.bias = nn.Parameter(torch.zeros([]))
        self.scale = nn.Parameter(torch.ones([]) * 10)

    def forward(self, pixel_values: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        outputs = self.encoder(pixel_values)
        features = outputs["pooler_output"]
        if self.proj is not None:
            features = self.proj(features)
        if normalize:
            features = F.normalize(features, dim=-1)
        return features

    def compute_loss(
        self, image_features: torch.Tensor, text_features: torch.Tensor
    ) -> torch.Tensor:
        """Compute SigLIP sigmoid loss."""
        logits = image_features @ text_features.T * self.scale + self.bias
        batch_size = image_features.size(0)
        labels = 2 * torch.eye(batch_size, device=logits.device) - 1
        return -F.logsigmoid(labels * logits).mean()
