"""
Vision encoders for multi-modal models.

Implements:
- Vision Transformer (ViT)
- CLIP Vision Encoder
- SigLIP Encoder
- Patch embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class VisionConfig:
    """Configuration for vision encoder."""
    image_size: int = 224
    patch_size: int = 16
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    num_channels: int = 3
    layer_norm_eps: float = 1e-6
    use_class_token: bool = True
    use_2d_pos_embed: bool = True


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
        """
        Args:
            image_size: Input image size
            patch_size: Size of each patch
            num_channels: Number of input channels
            hidden_size: Output hidden dimension
        """
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Projection via convolution
        self.projection = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, channels, height, width]

        Returns:
            Patch embeddings [batch, num_patches, hidden_size]
        """
        # Project patches
        x = self.projection(pixel_values)  # [B, H, h/p, w/p]

        # Flatten spatial dimensions
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, H]

        return x


class VisionAttention(nn.Module):
    """Multi-head self-attention for vision."""

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # QKV projection
        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn = attn + attention_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output
        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        x = self.proj(x)

        return x


class VisionMLP(nn.Module):
    """MLP block for vision transformer."""

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.fc1(hidden_states)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class VisionTransformerBlock(nn.Module):
    """Single transformer block for vision."""

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.attention = VisionAttention(config)
        self.mlp = VisionMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT).

    Processes images as sequences of patches through transformer layers.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            hidden_size=config.hidden_size,
        )

        num_patches = self.patch_embed.num_patches

        # Class token
        if config.use_class_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        # Position embeddings
        num_positions = num_patches + (1 if config.use_class_token else 0)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_positions, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        if hasattr(self, 'cls_token'):
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

        # Patch embeddings
        hidden_states = self.patch_embed(pixel_values)

        # Add class token
        if self.config.use_class_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            hidden_states = torch.cat([cls_tokens, hidden_states], dim=1)

        # Add position embeddings
        hidden_states = hidden_states + self.position_embedding
        hidden_states = self.dropout(hidden_states)

        # Transformer blocks
        all_hidden_states = [] if output_hidden_states else None

        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        # Get class token output
        if self.config.use_class_token:
            pooled_output = hidden_states[:, 0]
        else:
            pooled_output = hidden_states.mean(dim=1)

        result = {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output,
        }

        if output_hidden_states:
            result['hidden_states'] = all_hidden_states

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
        num_heads: int = 12,
        output_dim: Optional[int] = None,
    ):
        """
        Args:
            image_size: Input image size
            patch_size: Patch size
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            output_dim: Output projection dimension
        """
        super().__init__()

        config = VisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
        )

        self.encoder = VisionTransformer(config)

        # Optional output projection
        if output_dim is not None and output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = None

    def forward(
        self,
        pixel_values: torch.Tensor,
        return_all_features: bool = False,
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
            features = outputs['last_hidden_state']
        else:
            features = outputs['pooler_output']

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
        num_heads: int = 16,
        output_dim: int = 768,
    ):
        super().__init__()

        config = VisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
        )

        self.encoder = VisionTransformer(config)
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)

        # Learnable temperature for contrastive learning
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, channels, height, width]
            normalize: L2 normalize output features

        Returns:
            Image features [batch, output_dim]
        """
        outputs = self.encoder(pixel_values)
        features = outputs['pooler_output']
        features = self.proj(features)

        if normalize:
            features = F.normalize(features, dim=-1)

        return features

    def get_logit_scale(self) -> torch.Tensor:
        """Get clamped logit scale for contrastive loss."""
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
        num_heads: int = 16,
        output_dim: int = 1152,
    ):
        super().__init__()

        config = VisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            use_class_token=False,  # SigLIP uses mean pooling
        )

        self.encoder = VisionTransformer(config)

        if output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = None

        # Bias and scale for sigmoid loss
        self.bias = nn.Parameter(torch.zeros([]))
        self.scale = nn.Parameter(torch.ones([]) * 10)

    def forward(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: [batch, channels, height, width]
            normalize: L2 normalize output features

        Returns:
            Image features [batch, output_dim]
        """
        outputs = self.encoder(pixel_values)
        features = outputs['pooler_output']

        if self.proj is not None:
            features = self.proj(features)

        if normalize:
            features = F.normalize(features, dim=-1)

        return features

    def compute_loss(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute SigLIP sigmoid loss."""
        # Cosine similarity
        logits = image_features @ text_features.T * self.scale + self.bias

        # Labels: diagonal is positive
        batch_size = image_features.size(0)
        labels = 2 * torch.eye(batch_size, device=logits.device) - 1

        # Sigmoid loss
        loss = -F.logsigmoid(labels * logits).mean()

        return loss
