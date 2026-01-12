"""
Audio encoders for multi-modal models.

Implements:
- Mel-spectrogram encoding
- Whisper-style audio encoder
- Convolutional audio feature extraction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from dataclasses import dataclass
import math


@dataclass
class AudioConfig:
    """Configuration for audio encoder."""
    n_mels: int = 80               # Number of mel filterbank channels
    n_fft: int = 400               # FFT size
    hop_length: int = 160          # Hop length for STFT
    sample_rate: int = 16000       # Audio sample rate
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.0
    max_length: int = 3000         # Maximum spectrogram length
    layer_norm_eps: float = 1e-5


class AudioConvStack(nn.Module):
    """
    Convolutional frontend for audio processing.

    Downsamples mel-spectrograms and projects to hidden dimension.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 768,
    ):
        """
        Args:
            n_mels: Number of mel channels
            hidden_size: Output hidden dimension
        """
        super().__init__()

        # Two conv layers with GELU, similar to Whisper
        self.conv1 = nn.Conv1d(
            n_mels,
            hidden_size,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.gelu = nn.GELU()

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]

        Returns:
            Features [batch, time/2, hidden_size]
        """
        x = self.conv1(mel_spectrogram)
        x = self.gelu(x)
        x = self.conv2(x)
        x = self.gelu(x)

        # [B, H, T] -> [B, T, H]
        x = x.transpose(1, 2)

        return x


class AudioAttention(nn.Module):
    """Multi-head self-attention for audio."""

    def __init__(self, config: AudioConfig):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape back
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)

        return self.out_proj(attn_output)


class AudioMLP(nn.Module):
    """MLP block for audio transformer."""

    def __init__(self, config: AudioConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class AudioTransformerBlock(nn.Module):
    """Single transformer block for audio."""

    def __init__(self, config: AudioConfig):
        super().__init__()

        self.self_attn = AudioAttention(config)
        self.mlp = AudioMLP(config)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # Pre-norm MLP
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MelSpectrogramEncoder(nn.Module):
    """
    Mel-spectrogram based audio encoder.

    Converts audio to mel-spectrograms and encodes with transformer.
    """

    def __init__(self, config: AudioConfig):
        super().__init__()

        self.config = config

        # Convolutional frontend
        self.conv_stack = AudioConvStack(config.n_mels, config.hidden_size)

        # Positional embedding
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            AudioTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]
            attention_mask: Optional attention mask

        Returns:
            Dictionary with encoder outputs
        """
        # Conv frontend
        hidden_states = self.conv_stack(mel_spectrogram)

        # Add positional embeddings
        seq_len = hidden_states.size(1)
        positions = torch.arange(seq_len, device=hidden_states.device)
        hidden_states = hidden_states + self.position_embedding(positions)

        # Transformer blocks
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        hidden_states = self.norm(hidden_states)

        return {
            'last_hidden_state': hidden_states,
            'pooler_output': hidden_states.mean(dim=1),
        }


class WhisperEncoder(nn.Module):
    """
    Whisper-style audio encoder.

    Based on the OpenAI Whisper architecture for speech processing.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        max_length: int = 1500,
    ):
        """
        Args:
            n_mels: Number of mel channels
            hidden_size: Hidden dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            max_length: Maximum sequence length
        """
        super().__init__()

        config = AudioConfig(
            n_mels=n_mels,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_length=max_length,
        )

        # Conv layers (like Whisper)
        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)

        # Sinusoidal positional embeddings
        self.register_buffer(
            "positional_embedding",
            self._create_sinusoidal_embeddings(max_length, hidden_size)
        )

        # Transformer layers
        self.blocks = nn.ModuleList([
            AudioTransformerBlock(config)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

    def _create_sinusoidal_embeddings(
        self,
        max_length: int,
        hidden_size: int,
    ) -> torch.Tensor:
        """Create sinusoidal positional embeddings."""
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size)
        )

        pe = torch.zeros(max_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return pe

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]
            attention_mask: Optional attention mask

        Returns:
            Dictionary with encoder outputs
        """
        # Conv layers
        x = F.gelu(self.conv1(mel_spectrogram))
        x = F.gelu(self.conv2(x))

        # [B, H, T] -> [B, T, H]
        x = x.permute(0, 2, 1)

        # Add positional embeddings
        seq_len = x.size(1)
        x = x + self.positional_embedding[:seq_len]

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attention_mask)

        x = self.norm(x)

        return {
            'last_hidden_state': x,
            'pooler_output': x.mean(dim=1),
        }


class AudioEncoder(nn.Module):
    """
    Generic audio encoder wrapper.

    Provides a simple interface for audio encoding.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        output_dim: Optional[int] = None,
        encoder_type: str = "whisper",
    ):
        """
        Args:
            n_mels: Number of mel channels
            hidden_size: Hidden dimension
            num_layers: Number of layers
            num_heads: Number of attention heads
            output_dim: Output projection dimension
            encoder_type: Type of encoder ("whisper" or "basic")
        """
        super().__init__()

        if encoder_type == "whisper":
            self.encoder = WhisperEncoder(
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        else:
            config = AudioConfig(
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
            )
            self.encoder = MelSpectrogramEncoder(config)

        # Output projection
        if output_dim is not None and output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = None

    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        return_all_features: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]
            return_all_features: Return all time-step features

        Returns:
            Audio features
        """
        outputs = self.encoder(mel_spectrogram)

        if return_all_features:
            features = outputs['last_hidden_state']
        else:
            features = outputs['pooler_output']

        if self.proj is not None:
            features = self.proj(features)

        return features


class AudioPreprocessor(nn.Module):
    """
    Audio preprocessing: waveform to mel-spectrogram.

    Note: Requires torchaudio for full functionality.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        normalize: bool = True,
    ):
        """
        Args:
            sample_rate: Audio sample rate
            n_mels: Number of mel channels
            n_fft: FFT size
            hop_length: Hop length
            normalize: Normalize mel spectrogram
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize

        # Create mel filterbank
        self.register_buffer(
            "mel_filters",
            self._create_mel_filters(sample_rate, n_fft, n_mels)
        )

    def _create_mel_filters(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int,
    ) -> torch.Tensor:
        """Create mel filterbank matrix."""
        # Simplified mel filterbank creation
        # In practice, use torchaudio.transforms.MelSpectrogram

        min_freq = 0
        max_freq = sample_rate / 2

        # Mel scale conversion
        def hz_to_mel(hz):
            return 2595 * math.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        min_mel = hz_to_mel(min_freq)
        max_mel = hz_to_mel(max_freq)

        mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m) for m in mel_points])

        # FFT bin frequencies
        fft_bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)

        # Create filterbank
        filters = torch.zeros(n_mels, n_fft // 2 + 1)

        for i in range(n_mels):
            left = hz_points[i]
            center = hz_points[i + 1]
            right = hz_points[i + 2]

            for j, freq in enumerate(fft_bins):
                if left <= freq <= center:
                    filters[i, j] = (freq - left) / (center - left)
                elif center <= freq <= right:
                    filters[i, j] = (right - freq) / (right - center)

        return filters

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert waveform to mel spectrogram.

        Args:
            waveform: [batch, samples] or [batch, 1, samples]

        Returns:
            Mel spectrogram [batch, n_mels, time]
        """
        if waveform.dim() == 3:
            waveform = waveform.squeeze(1)

        # STFT
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )

        # Power spectrogram
        power = stft.abs() ** 2

        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_filters.to(power.device), power)

        # Log mel spectrogram
        mel_spec = torch.log(mel_spec.clamp(min=1e-10))

        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec
