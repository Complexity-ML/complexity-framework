"""
State encoders for robotics.

Provides:
- Proprioception encoding
- Multi-modal state encoding (vision + proprio)
- Temporal state encoding (history)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class StateConfig:
    """Configuration for state encoder."""
    proprio_dim: int = 14              # Proprioceptive state dimension
    hidden_size: int = 768
    num_layers: int = 2
    dropout: float = 0.1
    use_vision: bool = True
    vision_hidden_size: int = 768
    history_length: int = 1            # Number of past states to consider
    use_temporal: bool = False


class ProprioceptionEncoder(nn.Module):
    """
    Encode proprioceptive state (joint positions, velocities, etc.).

    Maps robot state to hidden representation.
    """

    def __init__(
        self,
        proprio_dim: int,
        hidden_size: int,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """
        Args:
            proprio_dim: Dimension of proprioceptive state
            hidden_size: Output hidden dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
        """
        super().__init__()

        self.proprio_dim = proprio_dim
        self.hidden_size = hidden_size

        layers = []
        input_dim = proprio_dim

        for i in range(num_layers):
            output_dim = hidden_size
            layers.extend([
                nn.Linear(input_dim, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            input_dim = output_dim

        self.encoder = nn.Sequential(*layers)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        Encode proprioceptive state.

        Args:
            proprio: Proprioceptive state [batch, proprio_dim]

        Returns:
            Encoded state [batch, hidden_size]
        """
        return self.encoder(proprio)


class MultimodalStateEncoder(nn.Module):
    """
    Encode multi-modal robot state.

    Combines:
    - Visual observations (images)
    - Proprioceptive state
    - Language instructions (optional)
    """

    def __init__(
        self,
        proprio_dim: int = 14,
        hidden_size: int = 768,
        vision_hidden_size: int = 768,
        language_hidden_size: int = 768,
        use_vision: bool = True,
        use_language: bool = False,
        num_cameras: int = 1,
        dropout: float = 0.1,
    ):
        """
        Args:
            proprio_dim: Proprioceptive dimension
            hidden_size: Output hidden dimension
            vision_hidden_size: Vision encoder hidden size
            language_hidden_size: Language encoder hidden size
            use_vision: Include vision modality
            use_language: Include language modality
            num_cameras: Number of camera views
            dropout: Dropout rate
        """
        super().__init__()

        self.use_vision = use_vision
        self.use_language = use_language
        self.num_cameras = num_cameras

        # Proprioception encoder
        self.proprio_encoder = ProprioceptionEncoder(
            proprio_dim=proprio_dim,
            hidden_size=hidden_size,
            dropout=dropout,
        )

        # Vision projection (assumes pre-encoded vision features)
        if use_vision:
            self.vision_proj = nn.Sequential(
                nn.Linear(vision_hidden_size * num_cameras, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )

        # Language projection
        if use_language:
            self.language_proj = nn.Sequential(
                nn.Linear(language_hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.GELU(),
            )

        # Fusion layer
        num_modalities = 1 + int(use_vision) + int(use_language)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * num_modalities, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        proprio: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        language_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode multi-modal state.

        Args:
            proprio: Proprioceptive state [batch, proprio_dim]
            vision_features: Pre-encoded vision [batch, num_cameras * vision_hidden]
            language_features: Pre-encoded language [batch, language_hidden]

        Returns:
            Fused state encoding [batch, hidden_size]
        """
        # Encode proprio
        proprio_encoded = self.proprio_encoder(proprio)
        modalities = [proprio_encoded]

        # Vision
        if self.use_vision and vision_features is not None:
            vision_encoded = self.vision_proj(vision_features)
            modalities.append(vision_encoded)

        # Language
        if self.use_language and language_features is not None:
            language_encoded = self.language_proj(language_features)
            modalities.append(language_encoded)

        # Fuse
        if len(modalities) > 1:
            fused = torch.cat(modalities, dim=-1)
            return self.fusion(fused)
        else:
            return proprio_encoded


class TemporalStateEncoder(nn.Module):
    """
    Encode temporal history of states.

    Uses transformer or RNN to encode state history.
    """

    def __init__(
        self,
        state_dim: int,
        hidden_size: int,
        history_length: int = 10,
        num_layers: int = 2,
        num_heads: int = 8,
        encoder_type: str = "transformer",  # transformer, lstm, gru
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: Input state dimension
            hidden_size: Hidden dimension
            history_length: Number of past states
            num_layers: Number of encoder layers
            num_heads: Attention heads (for transformer)
            encoder_type: Type of temporal encoder
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.hidden_size = hidden_size
        self.history_length = history_length
        self.encoder_type = encoder_type

        # State projection
        self.state_proj = nn.Linear(state_dim, hidden_size)

        # Positional encoding for history
        self.pos_embedding = nn.Parameter(
            torch.randn(1, history_length, hidden_size) * 0.02
        )

        if encoder_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        elif encoder_type == "lstm":
            self.encoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        elif encoder_type == "gru":
            self.encoder = nn.GRU(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
            )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        state_history: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode state history.

        Args:
            state_history: Past states [batch, history_length, state_dim]
            mask: Optional mask for padding

        Returns:
            Encoded history [batch, hidden_size]
        """
        batch_size, seq_len = state_history.shape[:2]

        # Project states
        x = self.state_proj(state_history)

        # Add positional encoding
        x = x + self.pos_embedding[:, :seq_len, :]

        if self.encoder_type == "transformer":
            # Transformer encoding
            x = self.encoder(x, src_key_padding_mask=mask)
            # Pool: take last position or mean
            output = x[:, -1, :]  # Last position

        else:
            # RNN encoding
            if mask is not None:
                # Pack sequence
                lengths = (~mask).sum(dim=1).cpu()
                x = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )

            output, _ = self.encoder(x)

            if mask is not None:
                output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

            if isinstance(output, tuple):
                output = output[0]

            # Take last valid position
            output = output[:, -1, :]

        return self.norm(output)


class StateEncoder(nn.Module):
    """
    Unified state encoder interface.

    Combines proprioception, vision, and temporal encoding.
    """

    def __init__(self, config: StateConfig):
        """
        Args:
            config: State encoder configuration
        """
        super().__init__()

        self.config = config

        # Base encoder
        if config.use_vision:
            self.base_encoder = MultimodalStateEncoder(
                proprio_dim=config.proprio_dim,
                hidden_size=config.hidden_size,
                vision_hidden_size=config.vision_hidden_size,
                use_vision=True,
                dropout=config.dropout,
            )
        else:
            self.base_encoder = ProprioceptionEncoder(
                proprio_dim=config.proprio_dim,
                hidden_size=config.hidden_size,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )

        # Temporal encoder
        if config.use_temporal and config.history_length > 1:
            self.temporal_encoder = TemporalStateEncoder(
                state_dim=config.hidden_size,
                hidden_size=config.hidden_size,
                history_length=config.history_length,
                num_layers=config.num_layers,
                dropout=config.dropout,
            )
        else:
            self.temporal_encoder = None

    def forward(
        self,
        proprio: torch.Tensor,
        vision_features: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode robot state.

        Args:
            proprio: Current proprioceptive state [batch, proprio_dim]
            vision_features: Vision features [batch, vision_hidden]
            history: State history [batch, history_length, state_dim]

        Returns:
            Encoded state [batch, hidden_size]
        """
        # Encode current state
        if self.config.use_vision and vision_features is not None:
            current = self.base_encoder(proprio, vision_features)
        else:
            current = self.base_encoder(proprio)

        # Apply temporal encoding
        if self.temporal_encoder is not None and history is not None:
            return self.temporal_encoder(history)

        return current


class FiLMConditioner(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioning.

    Modulates features based on conditioning input (e.g., language).
    Used in robot transformers for language conditioning.
    """

    def __init__(self, hidden_size: int, condition_size: int):
        """
        Args:
            hidden_size: Feature dimension to modulate
            condition_size: Conditioning input dimension
        """
        super().__init__()

        self.scale_proj = nn.Linear(condition_size, hidden_size)
        self.shift_proj = nn.Linear(condition_size, hidden_size)

    def forward(
        self,
        features: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM conditioning.

        Args:
            features: Features to modulate [batch, ..., hidden]
            condition: Conditioning input [batch, condition_size]

        Returns:
            Modulated features [batch, ..., hidden]
        """
        scale = self.scale_proj(condition)
        shift = self.shift_proj(condition)

        # Expand for broadcasting
        while scale.dim() < features.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)

        return features * (1 + scale) + shift
