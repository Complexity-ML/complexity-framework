"""Canonical TR-HASH modality blocks and task-specific connector components. Original position IDs are preserved for head/value routing and contextual expert assignment."""

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from complexity.tr_hash.attention import DomainSelfAttention, make_domain_mlp, resolve_domain_heads


@dataclass
class AudioConfig:
    """Configuration for audio encoder."""

    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 4
    num_key_value_heads: int | None = field(default=None, kw_only=True)
    intermediate_size: int | None = None
    shared_intermediate_size: Optional[int] = None
    dropout: float = 0.0
    max_length: int = 3000
    layer_norm_eps: float = 1e-05
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


class AudioConvStack(nn.Module):
    """
    Convolutional frontend for audio processing.

    Downsamples mel-spectrograms and projects to hidden dimension.
    """

    def __init__(self, n_mels: int = 80, hidden_size: int = 768):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]

        Returns:
            Features [batch, time/2, hidden_size]
        """
        x = self.gelu(self.conv1(mel_spectrogram))
        x = self.gelu(self.conv2(x))
        return x.transpose(1, 2)


class AudioAttention(nn.Module):
    """Multi-head self-attention for audio."""

    def __init__(self, config: AudioConfig, layer_index: int = 0):
        super().__init__()
        self.source_attention = DomainSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.dropout,
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


class AudioTokenRoutedMLP(nn.Module):
    """Shared SwiGLU plus two of four experts. Routes use original position IDs and their two preceding IDs; no learned gate."""

    def __init__(self, config: AudioConfig, layer_index: int = 0):
        super().__init__()
        self.engine = make_domain_mlp(
            hidden_size=config.hidden_size,
            vocab_size=config.max_length,
            num_experts=4,
            top_k=2,
            shared_width=config.shared_intermediate_size,
            expert_width=config.intermediate_size // 4,
            layer_index=layer_index,
        )

    def forward(self, x: torch.Tensor, expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.arange(x.size(1), device=x.device)
        route_ids = (
            expert_ids.unsqueeze(0).expand(x.size(0), -1) if expert_ids.ndim == 1 else expert_ids
        )
        return self.engine(x, route_ids)


class AudioTransformerBlock(nn.Module):
    """Single transformer block for audio (with contextual TR-HASH MLP)."""

    def __init__(self, config: AudioConfig, layer_index: int = 0):
        super().__init__()
        self.self_attn = AudioAttention(config, layer_index)
        self.mlp: nn.Module = AudioTokenRoutedMLP(config, layer_index)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, expert_ids)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.mlp(hidden_states, expert_ids)
        hidden_states = residual + hidden_states
        return hidden_states


class MelSpectrogramEncoder(nn.Module):
    """
    Mel-spectrogram based audio encoder with token-routed MLP.

    Precomputes expert_ids up to max_length // 2 (after conv stride=2).
    """

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        self.conv_stack = AudioConvStack(config.n_mels, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_length, config.hidden_size)
        self.blocks = nn.ModuleList(
            [AudioTransformerBlock(config, index) for index in range(config.num_hidden_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        max_t = config.max_length // 2
        expert_ids = torch.arange(max_t)
        self.register_buffer("expert_ids_table", expert_ids)

    def forward(
        self, mel_spectrogram: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]
            attention_mask: Optional attention mask

        Returns:
            Dictionary with encoder outputs
        """
        hidden_states = self.conv_stack(mel_spectrogram)
        seq_len = hidden_states.size(1)
        max_frames = self.expert_ids_table.numel()
        if seq_len > max_frames:
            raise ValueError(
                f"mel spectrogram produced {seq_len} encoder frames after downsampling, exceeding max_length={self.config.max_length} ({max_frames} frames after stride-2 conv); truncate the input or construct this encoder with a larger max_length"
            )
        positions = torch.arange(seq_len, device=hidden_states.device)
        hidden_states = hidden_states + self.position_embedding(positions)
        expert_ids = self.expert_ids_table[:seq_len]
        for block in self.blocks:
            hidden_states = block(hidden_states, expert_ids, attention_mask)
        hidden_states = self.norm(hidden_states)
        return {"last_hidden_state": hidden_states, "pooler_output": hidden_states.mean(dim=1)}


class WhisperEncoder(nn.Module):
    """
    Whisper-style audio encoder with token-routed MLP.

    Based on the OpenAI Whisper architecture for speech processing.
    """

    def __init__(
        self,
        n_mels: int = 80,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 4,
        max_length: int = 1500,
        num_experts: int = 4,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        config = AudioConfig(
            num_key_value_heads=num_key_value_heads,
            n_mels=n_mels,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size,
            max_length=max_length,
            num_experts=num_experts,
        )
        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)
        self.register_buffer(
            "positional_embedding", self._create_sinusoidal_embeddings(max_length, hidden_size)
        )
        self.blocks = nn.ModuleList(
            [AudioTransformerBlock(config, index) for index in range(num_layers)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        expert_ids = torch.arange(max_length)
        self.register_buffer("expert_ids_table", expert_ids)

    def _create_sinusoidal_embeddings(self, max_length: int, hidden_size: int) -> torch.Tensor:
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
        pe = torch.zeros(max_length, hidden_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(
        self, mel_spectrogram: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]
            attention_mask: Optional attention mask

        Returns:
            Dictionary with encoder outputs
        """
        x = F.gelu(self.conv1(mel_spectrogram))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)
        seq_len = x.size(1)
        x = x + self.positional_embedding[:seq_len]
        expert_ids = self.expert_ids_table[:seq_len]
        for block in self.blocks:
            x = block(x, expert_ids, attention_mask)
        x = self.norm(x)
        return {"last_hidden_state": x, "pooler_output": x.mean(dim=1)}


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
        num_heads: int = 4,
        output_dim: Optional[int] = None,
        encoder_type: str = "whisper",
        num_experts: int = 4,
        *,
        num_key_value_heads: int | None = None,
    ):
        super().__init__()
        if encoder_type == "whisper":
            self.encoder = WhisperEncoder(
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                num_experts=num_experts,
                num_key_value_heads=num_key_value_heads,
            )
        else:
            config = AudioConfig(
                num_key_value_heads=num_key_value_heads,
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                num_experts=num_experts,
            )
            self.encoder = MelSpectrogramEncoder(config)
        if output_dim is not None and output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim)
        else:
            self.proj = None

    def forward(
        self, mel_spectrogram: torch.Tensor, return_all_features: bool = False
    ) -> torch.Tensor:
        """
        Args:
            mel_spectrogram: [batch, n_mels, time]
            return_all_features: Return all time-step features

        Returns:
            Audio features
        """
        outputs = self.encoder(mel_spectrogram)
        features = outputs["last_hidden_state"] if return_all_features else outputs["pooler_output"]
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
        super().__init__()
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.normalize = normalize
        self.register_buffer("mel_filters", self._create_mel_filters(sample_rate, n_fft, n_mels))

    def _create_mel_filters(self, sample_rate: int, n_fft: int, n_mels: int) -> torch.Tensor:
        """Create mel filterbank matrix."""

        def hz_to_mel(hz):
            return 2595 * math.log10(1 + hz / 700)

        def mel_to_hz(mel):
            return 700 * (10 ** (mel / 2595) - 1)

        min_mel = hz_to_mel(0)
        max_mel = hz_to_mel(sample_rate / 2)
        mel_points = torch.linspace(min_mel, max_mel, n_mels + 2)
        hz_points = torch.tensor([mel_to_hz(m.item()) for m in mel_points])
        fft_bins = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
        filters = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            left, center, right = (hz_points[i], hz_points[i + 1], hz_points[i + 2])
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
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )
        power = stft.abs() ** 2
        mel_spec = torch.matmul(self.mel_filters.to(power.device), power)
        mel_spec = torch.log(mel_spec.clamp(min=1e-10))
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-08)
        return mel_spec
