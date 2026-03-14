"""
Audio encoders for multi-modal models.

Implements:
- Mel-spectrogram encoding
- Whisper-style audio encoder
- Convolutional audio feature extraction

v2: AudioTokenRoutedMLP — time-step position routing (t % num_experts).
    Position-only, precomputed where length is fixed; computed on-the-fly
    otherwise. Same fused BMM pattern as TokenRoutedMLPParallel.
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
    # Token-routed MLP: 1 = standard, >1 = time-step position routing
    num_experts: int = 4


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
        return x.transpose(1, 2)   # [B, T/2, H]


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

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.out_proj(attn_output)


# =============================================================================
# Token-Routed MLP (time-step position routing)
# =============================================================================

class AudioTokenRoutedMLP(nn.Module):
    """
    Token-Routed MLP for audio tokens.

    Routing key: time-step position t in [0, seq_len)
    Expert assignment: t % num_experts

    Pass `expert_ids` precomputed ([seq_len]) for fixed-length encoders
    (e.g. Whisper after conv downsampling) or compute on-the-fly for
    variable-length inputs. Both paths are deterministic and
    fullgraph=True safe.

    Fused BMM: gate+up → SwiGLU → down.
    """

    def __init__(self, config: AudioConfig):
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

    def forward(
        self,
        x: torch.Tensor,                              # [B, T, H]
        expert_ids: Optional[torch.Tensor] = None,    # [T] or None
    ) -> torch.Tensor:
        B, T, H = x.shape

        if expert_ids is None:
            expert_ids = torch.arange(T, device=x.device) % self.num_experts

        flat = x.view(B * T, H)
        eids = expert_ids.unsqueeze(0).expand(B, -1).reshape(B * T)

        gu_w = self.gate_up_proj[eids]    # [B*T, H, 2*I_e]
        down_w = self.down_proj[eids]     # [B*T, I_e, H]

        gu = torch.bmm(flat.unsqueeze(1), gu_w).squeeze(1)
        gate, up = gu.split(self.expert_intermediate_size, dim=-1)
        inter = F.silu(gate) * up
        out = torch.bmm(inter.unsqueeze(1), down_w).squeeze(1)

        return out.view(B, T, H)


class AudioMLP(nn.Module):
    """Standard MLP block (fallback when num_experts == 1)."""

    def __init__(self, config: AudioConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.Tensor, _expert_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden_states = self.act(self.fc1(hidden_states))
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class AudioTransformerBlock(nn.Module):
    """Single transformer block for audio (with optional token-routed MLP)."""

    def __init__(self, config: AudioConfig):
        super().__init__()

        self.self_attn = AudioAttention(config)

        if config.num_experts > 1:
            self.mlp: nn.Module = AudioTokenRoutedMLP(config)
        else:
            self.mlp = AudioMLP(config)

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        expert_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = hidden_states
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)
        hidden_states = residual + hidden_states

        # Token-routed (or plain) MLP
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

        self.blocks = nn.ModuleList([
            AudioTransformerBlock(config)
            for _ in range(config.num_hidden_layers)
        ])

        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Precompute expert_ids for time steps after conv downsampling (stride=2)
        max_t = config.max_length // 2
        expert_ids = torch.arange(max_t) % config.num_experts
        self.register_buffer("expert_ids_table", expert_ids)  # [max_T]

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
        hidden_states = self.conv_stack(mel_spectrogram)  # [B, T, H]
        seq_len = hidden_states.size(1)

        positions = torch.arange(seq_len, device=hidden_states.device)
        hidden_states = hidden_states + self.position_embedding(positions)

        expert_ids = self.expert_ids_table[:seq_len]

        for block in self.blocks:
            hidden_states = block(hidden_states, expert_ids, attention_mask)

        hidden_states = self.norm(hidden_states)

        return {
            'last_hidden_state': hidden_states,
            'pooler_output': hidden_states.mean(dim=1),
        }


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
        num_heads: int = 12,
        max_length: int = 1500,
        num_experts: int = 4,
    ):
        super().__init__()

        config = AudioConfig(
            n_mels=n_mels,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            max_length=max_length,
            num_experts=num_experts,
        )

        self.conv1 = nn.Conv1d(n_mels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, stride=2, padding=1)

        self.register_buffer(
            "positional_embedding",
            self._create_sinusoidal_embeddings(max_length, hidden_size)
        )

        self.blocks = nn.ModuleList([
            AudioTransformerBlock(config)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_size)

        # Precompute expert routing (fixed Whisper output length after conv)
        expert_ids = torch.arange(max_length) % config.num_experts
        self.register_buffer("expert_ids_table", expert_ids)  # [max_length]

    def _create_sinusoidal_embeddings(self, max_length: int, hidden_size: int) -> torch.Tensor:
        position = torch.arange(max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2) * (-math.log(10000.0) / hidden_size))
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
        x = F.gelu(self.conv1(mel_spectrogram))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)                # [B, T, H]

        seq_len = x.size(1)
        x = x + self.positional_embedding[:seq_len]

        expert_ids = self.expert_ids_table[:seq_len]

        for block in self.blocks:
            x = block(x, expert_ids, attention_mask)

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
        num_experts: int = 4,
    ):
        super().__init__()

        if encoder_type == "whisper":
            self.encoder = WhisperEncoder(
                n_mels=n_mels,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                num_experts=num_experts,
            )
        else:
            config = AudioConfig(
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

        features = outputs['last_hidden_state'] if return_all_features else outputs['pooler_output']

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

        self.register_buffer(
            "mel_filters",
            self._create_mel_filters(sample_rate, n_fft, n_mels)
        )

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
            left, center, right = hz_points[i], hz_points[i + 1], hz_points[i + 2]
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
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        return mel_spec
