"""
Vision encoders for multi-modal models.

Implements:
- Vision Transformer (ViT)
- CLIP Vision Encoder
- SigLIP Encoder
- Patch embeddings

v2: VisionTokenRoutedMLP — patch-position routing (patch_id % num_experts).
    Position-only, precomputed, fullgraph=True safe.
    Same fused BMM pattern as TokenRoutedMLPParallel.
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
    # Token-routed MLP: 1 = standard, >1 = position routing
    num_experts: int = 4


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
        x = self.projection(pixel_values)      # [B, H, h/p, w/p]
        x = x.flatten(2).transpose(1, 2)       # [B, num_patches, H]
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

        qkv = self.qkv(hidden_states)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)
        return self.proj(x)


# =============================================================================
# Token-Routed MLP (patch-position routing)
# =============================================================================

class VisionTokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP for vision patches.

    Routing key: patch position in raster order (0..num_patches-1)
    Expert assignment: patch_id % num_experts

    Deterministic, position-only — precomputed expert_ids are passed in.
    Fused BMM: gate+up → SwiGLU → down (fullgraph=True safe).
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.expert_intermediate_size = config.intermediate_size // config.num_experts

        self.gate_up_proj = nn.Parameter(
            torch.randn(config.num_experts, config.hidden_size, self.expert_intermediate_size * 2) * 0.02
        )
        self.down_proj = nn.Parameter(
            torch.randn(config.num_experts, self.expert_intermediate_size, config.hidden_size) * 0.02
        )

    def forward(self, x: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, H]
            expert_ids: [N]  precomputed per position

        Returns:
            [B, N, H]
        """
        B, N, H = x.shape
        flat = x.view(B * N, H)
        eids = expert_ids.unsqueeze(0).expand(B, -1).reshape(B * N)

        gu_w = self.gate_up_proj[eids]    # [B*N, H, 2*I_e]
        down_w = self.down_proj[eids]     # [B*N, I_e, H]

        gu = torch.bmm(flat.unsqueeze(1), gu_w).squeeze(1)
        gate, up = gu.split(self.expert_intermediate_size, dim=-1)
        inter = F.silu(gate) * up
        out = torch.bmm(inter.unsqueeze(1), down_w).squeeze(1)

        return out.view(B, N, H)


class VisionMLP(nn.Module):
    """Standard MLP block (fallback when num_experts == 1)."""

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, _expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.fc1(hidden_states)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class VisionTransformerBlock(nn.Module):
    """Single transformer block for vision (with optional token-routed MLP)."""

    def __init__(self, config: VisionConfig):
        super().__init__()

        self.attention = VisionAttention(config)

        if config.num_experts > 1:
            self.mlp: nn.Module = VisionTokenRoutedMLP(config)
        else:
            self.mlp = VisionMLP(config)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # Token-routed MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, expert_ids)
        hidden_states = residual + hidden_states

        return hidden_states


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) with token-routed MLP.

    Patches routed by spatial position: expert_id = patch_id % num_experts.
    Expert IDs are precomputed at construction and reused every forward pass.
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
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_positions, config.hidden_size)
        )

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.blocks = nn.ModuleList([
            VisionTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Precompute expert routing by patch position
        # CLS token (position 0) gets expert 0; patch i gets expert i % num_experts
        patch_ids = torch.arange(num_patches) % config.num_experts
        if config.use_class_token:
            cls_expert = torch.zeros(1, dtype=torch.long)
            expert_ids = torch.cat([cls_expert, patch_ids])
        else:
            expert_ids = patch_ids
        self.register_buffer("expert_ids", expert_ids)  # [num_positions], long

        self._init_weights()

    def _init_weights(self):
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
        num_experts: int = 4,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        config = VisionConfig(
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
        num_experts: int = 4,
    ):
        super().__init__()

        config = VisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            num_experts=num_experts,
        )

        self.encoder = VisionTransformer(config)
        self.proj = nn.Linear(hidden_size, output_dim, bias=False)

        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1 / 0.07))

    def forward(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        outputs = self.encoder(pixel_values)
        features = outputs['pooler_output']
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
        num_heads: int = 16,
        output_dim: int = 1152,
        num_experts: int = 4,
    ):
        super().__init__()

        config = VisionConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
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

    def forward(
        self,
        pixel_values: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
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
        logits = image_features @ text_features.T * self.scale + self.bias
        batch_size = image_features.size(0)
        labels = 2 * torch.eye(batch_size, device=logits.device) - 1
        return -F.logsigmoid(labels * logits).mean()
