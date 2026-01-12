"""
Robot Transformer models.

Implements:
- RT-1 (Robotics Transformer 1)
- RT-2 (Vision-Language-Action model)
- Octo (Generalist robot policy)

These are transformer-based policies for robot manipulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

from .action_space import ActionConfig, ActionTokenizer, ActionHead, DiffusionActionHead
from .state_encoder import StateConfig, StateEncoder, FiLMConditioner


@dataclass
class RobotConfig:
    """Configuration for robot transformer."""
    # Architecture
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1

    # State
    proprio_dim: int = 14              # Joint positions, velocities
    state_history: int = 6             # Number of past observations

    # Action
    action_dim: int = 7                # 6-DOF + gripper
    action_bins: int = 256             # For discrete actions
    action_chunk_size: int = 1         # Multi-step prediction
    use_discrete_actions: bool = True

    # Vision
    image_size: int = 224
    patch_size: int = 16
    num_cameras: int = 1
    vision_hidden_size: int = 768

    # Language
    use_language: bool = True
    language_hidden_size: int = 768
    max_language_length: int = 77

    # Training
    max_sequence_length: int = 2048


class EfficientNetEncoder(nn.Module):
    """
    Simplified EfficientNet-style vision encoder.

    For RT-1 style models that use CNN vision backbone.
    """

    def __init__(
        self,
        hidden_size: int = 512,
        num_channels: int = 3,
    ):
        super().__init__()

        # Simplified CNN backbone
        self.features = nn.Sequential(
            # Stem
            nn.Conv2d(num_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),

            # Stage 1
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),

            # Stage 2
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),

            # Stage 3
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),

            # Stage 4
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),

            # Global pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.proj = nn.Linear(512, hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch, channels, height, width]

        Returns:
            Features [batch, hidden_size]
        """
        features = self.features(images)
        return self.proj(features)


class TokenLearner(nn.Module):
    """
    TokenLearner module for adaptive tokenization.

    Learns to select/generate a fixed number of tokens from spatial features.
    Used in RT-1 for efficient token processing.

    Reference: TokenLearner (https://arxiv.org/abs/2106.11297)
    """

    def __init__(
        self,
        hidden_size: int,
        num_tokens: int = 8,
    ):
        """
        Args:
            hidden_size: Feature dimension
            num_tokens: Number of output tokens
        """
        super().__init__()

        self.num_tokens = num_tokens

        # Spatial attention for each token
        self.attention = nn.Sequential(
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.GELU(),
            nn.Conv2d(hidden_size, num_tokens, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Spatial features [batch, hidden, height, width]

        Returns:
            Tokens [batch, num_tokens, hidden]
        """
        batch_size, hidden, h, w = x.shape

        # Compute attention weights
        attn = self.attention(x)  # [B, num_tokens, H, W]
        attn = attn.view(batch_size, self.num_tokens, -1)  # [B, num_tokens, H*W]
        attn = F.softmax(attn, dim=-1)

        # Apply attention
        x_flat = x.view(batch_size, hidden, -1)  # [B, hidden, H*W]
        tokens = torch.bmm(attn, x_flat.transpose(1, 2))  # [B, num_tokens, hidden]

        return tokens


class RT1Model(nn.Module):
    """
    RT-1: Robotics Transformer 1.

    Uses EfficientNet vision + Transformer for action prediction.
    Actions are discretized into bins.

    Reference: RT-1 (https://arxiv.org/abs/2212.06817)
    """

    def __init__(self, config: RobotConfig):
        """
        Args:
            config: Robot configuration
        """
        super().__init__()

        self.config = config

        # Vision encoder (per-camera)
        self.vision_encoder = EfficientNetEncoder(
            hidden_size=config.hidden_size,
        )

        # TokenLearner for efficient processing
        self.token_learner = TokenLearner(
            hidden_size=config.hidden_size,
            num_tokens=8,
        )

        # Language encoder (use pre-trained embedding)
        if config.use_language:
            self.language_embed = nn.Embedding(
                32000,  # Vocab size
                config.language_hidden_size,
            )
            self.language_proj = nn.Linear(
                config.language_hidden_size,
                config.hidden_size,
            )

        # FiLM conditioning for language
        self.film_layers = nn.ModuleList([
            FiLMConditioner(config.hidden_size, config.hidden_size)
            for _ in range(config.num_layers)
        ])

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Action head (discrete)
        self.action_head = nn.Linear(
            config.hidden_size,
            config.action_dim * config.action_bins,
        )

    def forward(
        self,
        images: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [batch, num_cameras, C, H, W]
            language_tokens: Language instruction tokens [batch, seq]
            proprio: Proprioceptive state [batch, proprio_dim]

        Returns:
            Dictionary with action logits
        """
        batch_size = images.size(0)

        # Handle multiple cameras
        if images.dim() == 5:
            # [B, num_cam, C, H, W] -> [B * num_cam, C, H, W]
            num_cameras = images.size(1)
            images = images.view(-1, *images.shape[2:])
            vision_features = self.vision_encoder(images)
            vision_features = vision_features.view(batch_size, num_cameras, -1)
            vision_features = vision_features.mean(dim=1)  # Average cameras
        else:
            vision_features = self.vision_encoder(images)

        # Language encoding
        if self.config.use_language and language_tokens is not None:
            language_emb = self.language_embed(language_tokens)
            language_features = language_emb.mean(dim=1)  # Pool
            language_features = self.language_proj(language_features)
        else:
            language_features = torch.zeros(
                batch_size, self.config.hidden_size,
                device=images.device
            )

        # Prepare sequence
        tokens = vision_features.unsqueeze(1)  # [B, 1, H]

        # Transformer with FiLM conditioning
        x = tokens
        for i, layer in enumerate(self.transformer.layers):
            x = layer(x)
            x = self.film_layers[i](x, language_features)

        # Action prediction
        action_logits = self.action_head(x[:, -1, :])
        action_logits = action_logits.view(
            batch_size, self.config.action_dim, self.config.action_bins
        )

        return {
            'action_logits': action_logits,
            'action_probs': F.softmax(action_logits, dim=-1),
        }

    def get_action(
        self,
        images: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Get action for execution.

        Args:
            images: Input images
            language_tokens: Language instruction
            deterministic: Use argmax instead of sampling

        Returns:
            Action [batch, action_dim]
        """
        outputs = self.forward(images, language_tokens)
        action_logits = outputs['action_logits']

        if deterministic:
            action_bins = action_logits.argmax(dim=-1)
        else:
            probs = F.softmax(action_logits, dim=-1)
            action_bins = torch.multinomial(
                probs.view(-1, self.config.action_bins),
                num_samples=1
            ).view(-1, self.config.action_dim)

        # Convert bins to continuous actions
        actions = (action_bins.float() + 0.5) / self.config.action_bins
        actions = actions * 2 - 1  # Scale to [-1, 1]

        return actions


class RT2Model(nn.Module):
    """
    RT-2: Vision-Language-Action Model.

    Uses a VLM backbone with action tokens appended to vocabulary.
    Actions are predicted as text tokens.

    Reference: RT-2 (https://arxiv.org/abs/2307.15818)
    """

    def __init__(self, config: RobotConfig):
        """
        Args:
            config: Robot configuration
        """
        super().__init__()

        self.config = config

        # In practice, RT-2 uses PaLI-X or PaLM-E as backbone
        # Here we implement a simplified version

        # Vision encoder
        self.vision_encoder = EfficientNetEncoder(
            hidden_size=config.hidden_size,
        )

        # Shared embedding for text + action tokens
        vocab_size = 32000 + config.action_dim * config.action_bins
        self.embedding = nn.Embedding(vocab_size, config.hidden_size)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.intermediate_size,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=config.num_layers,
        )

        # Output head
        self.lm_head = nn.Linear(config.hidden_size, vocab_size, bias=False)

        # Action token offset
        self.action_token_start = 32000

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [batch, C, H, W]
            input_ids: Text + action tokens [batch, seq]
            attention_mask: Attention mask

        Returns:
            Dictionary with logits
        """
        batch_size = images.size(0)

        # Vision encoding
        vision_features = self.vision_encoder(images)
        vision_tokens = vision_features.unsqueeze(1)  # [B, 1, H]

        # Text embedding
        text_emb = self.embedding(input_ids)

        # Concatenate vision and text
        decoder_input = torch.cat([vision_tokens, text_emb], dim=1)

        # Create causal mask
        seq_len = decoder_input.size(1)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=images.device),
            diagonal=1
        ).bool()

        # Decode
        hidden_states = self.decoder(
            decoder_input,
            vision_tokens,  # Cross-attend to vision
            tgt_mask=causal_mask,
        )

        # LM head
        logits = self.lm_head(hidden_states)

        return {
            'logits': logits,
            'hidden_states': hidden_states,
        }

    def generate_action(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        max_action_tokens: int = 7,
    ) -> torch.Tensor:
        """
        Generate action tokens autoregressively.

        Args:
            images: Input images
            prompt_ids: Language prompt tokens
            max_action_tokens: Number of action tokens to generate

        Returns:
            Generated action tokens
        """
        batch_size = images.size(0)
        device = images.device

        # Start with prompt
        generated = prompt_ids.clone()

        for _ in range(max_action_tokens):
            outputs = self.forward(images, generated)
            logits = outputs['logits']

            # Get next token (only from action vocab)
            next_logits = logits[:, -1, self.action_token_start:]
            next_token = next_logits.argmax(dim=-1) + self.action_token_start

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        # Extract action tokens
        action_tokens = generated[:, -max_action_tokens:] - self.action_token_start

        # Convert to continuous actions
        actions = (action_tokens.float() + 0.5) / self.config.action_bins
        actions = actions * 2 - 1

        return actions


class OctoModel(nn.Module):
    """
    Octo: Generalist Robot Policy.

    A generalist policy trained on diverse robot data.
    Uses diffusion for action prediction.

    Reference: Octo (https://arxiv.org/abs/2405.12213)
    """

    def __init__(self, config: RobotConfig):
        """
        Args:
            config: Robot configuration
        """
        super().__init__()

        self.config = config

        # Vision encoder (ViT-style)
        from ..multimodal.vision import VisionEncoder
        self.vision_encoder = VisionEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_size=config.hidden_size,
            num_layers=6,
            num_heads=config.num_heads,
        )

        # Language encoder
        if config.use_language:
            self.language_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=config.num_heads,
                    dim_feedforward=config.intermediate_size,
                    dropout=config.dropout,
                    batch_first=True,
                ),
                num_layers=4,
            )
            self.language_embed = nn.Embedding(32000, config.hidden_size)

        # Proprioception encoder
        self.proprio_encoder = nn.Sequential(
            nn.Linear(config.proprio_dim, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.GELU(),
        )

        # Transformer backbone
        self.backbone = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                batch_first=True,
            ),
            num_layers=config.num_layers,
        )

        # Diffusion action head
        self.action_head = DiffusionActionHead(
            hidden_size=config.hidden_size,
            action_dim=config.action_dim,
            action_chunk_size=config.action_chunk_size,
            num_diffusion_steps=100,
        )

        # Readout token
        self.readout_token = nn.Parameter(torch.randn(1, 1, config.hidden_size) * 0.02)

    def forward(
        self,
        images: torch.Tensor,
        proprio: torch.Tensor,
        language_tokens: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: Input images [batch, C, H, W] or [batch, history, C, H, W]
            proprio: Proprioceptive state [batch, proprio_dim]
            language_tokens: Language instruction [batch, seq]
            actions: Ground truth actions (for training) [batch, chunk, action_dim]

        Returns:
            Dictionary with action predictions or loss
        """
        batch_size = images.size(0)

        # Vision encoding
        if images.dim() == 5:
            # History of images
            history_len = images.size(1)
            images_flat = images.view(-1, *images.shape[2:])
            vision_features = self.vision_encoder(images_flat, return_all_features=True)
            vision_features = vision_features.view(batch_size, history_len, -1, self.config.hidden_size)
            vision_features = vision_features.view(batch_size, -1, self.config.hidden_size)
        else:
            vision_features = self.vision_encoder(images, return_all_features=True)

        tokens = [vision_features]

        # Proprio encoding
        proprio_features = self.proprio_encoder(proprio).unsqueeze(1)
        tokens.append(proprio_features)

        # Language encoding
        if self.config.use_language and language_tokens is not None:
            language_emb = self.language_embed(language_tokens)
            language_features = self.language_encoder(language_emb)
            tokens.append(language_features)

        # Readout token
        readout = self.readout_token.expand(batch_size, -1, -1)
        tokens.append(readout)

        # Concatenate all tokens
        all_tokens = torch.cat(tokens, dim=1)

        # Backbone
        hidden_states = self.backbone(all_tokens)

        # Get readout hidden state
        readout_hidden = hidden_states[:, -1, :]

        # Action prediction
        if actions is not None:
            # Training: return loss
            loss, _ = self.action_head(readout_hidden, actions)
            return {'loss': loss, 'hidden_states': hidden_states}
        else:
            # Inference: sample actions
            actions = self.action_head(readout_hidden)
            return {'actions': actions, 'hidden_states': hidden_states}


class RobotTransformer(nn.Module):
    """
    Unified robot transformer interface.

    Factory for creating RT-1, RT-2, or Octo style models.
    """

    def __init__(
        self,
        model_type: str = "rt1",
        config: Optional[RobotConfig] = None,
        **kwargs,
    ):
        """
        Args:
            model_type: Type of model (rt1, rt2, octo)
            config: Robot configuration
            **kwargs: Override config values
        """
        super().__init__()

        if config is None:
            config = RobotConfig(**kwargs)

        self.config = config
        self.model_type = model_type

        if model_type == "rt1":
            self.model = RT1Model(config)
        elif model_type == "rt2":
            self.model = RT2Model(config)
        elif model_type == "octo":
            self.model = OctoModel(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def get_action(self, *args, **kwargs) -> torch.Tensor:
        """Get action for execution."""
        self.eval()

        if hasattr(self.model, 'get_action'):
            return self.model.get_action(*args, **kwargs)
        else:
            outputs = self.model(*args, **kwargs)
            return outputs.get('actions', outputs.get('action_logits'))
