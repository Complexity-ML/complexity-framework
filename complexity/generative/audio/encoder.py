"""TR-Hash MoE audio encoder — frames route through deterministic expert IDs.

Same principle as ``complexity.generative.vision_language.vision_tower``:
each frame routes through real ``TRHashEngine`` experts keyed on its fixed
position in the sequence (deterministic, never learned or contextual).
Unlike image patches, the frame count is not fixed by the config alone (it
depends on input length), so the route-ID space is sized by ``max_frames``
and each forward pass only ever uses the first ``num_frames`` of it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from complexity.tr_hash import (
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPrecision,
    TRHashStrategy,
)

from .mel import LogMelSpectrogram


@dataclass
class AudioEncoderConfig:
    sample_rate: int = 16_000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 6
    num_experts: int = 4
    top_k: int = 2
    shared_width: int = 128
    expert_width: int = 48
    route_seed: int = 0x71D5A17
    max_frames: int = 1_500
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    conv_stride: int = 2

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")

    @property
    def route_vocab_size(self) -> int:
        return self.max_frames


class _AudioSelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (value.transpose(1, 2) for value in (q, k, v))
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
        )
        return self.out(output.transpose(1, 2).reshape(batch, length, width))


class AudioBlock(nn.Module):
    def __init__(self, config: AudioEncoderConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = _AudioSelfAttention(
            config.hidden_size, config.num_attention_heads, config.dropout
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = TRHashEngine(
            TRHashEngineConfig(
                hidden_size=config.hidden_size,
                vocab_size=config.route_vocab_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                shared_width=config.shared_width,
                expert_width=config.expert_width,
                routing_strategy=TRHashStrategy.BALANCED_HASH,
                layer_index=layer_index,
                route_seed=config.route_seed,
                precision=TRHashPrecision.BF16,
                backend=TRHashBackend.AUTO,
            )
        )

    def forward(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x), route_ids)
        return x


class AudioEncoder(nn.Module):
    """Waveform -> per-frame contextual embeddings ``[batch, frames, hidden_size]``.

    Frames route through real TR-Hash MoE experts by fixed position, the
    same principle as ``TRHashVisionTower`` for image patches.
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config
        self.mel = LogMelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mels,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )
        self.conv1 = nn.Conv1d(config.n_mels, config.hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=3,
            stride=config.conv_stride,
            padding=1,
        )
        self.gelu = nn.GELU()
        self.blocks = nn.ModuleList(
            AudioBlock(config, layer_index=index) for index in range(config.num_hidden_layers)
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.register_buffer(
            "route_ids", torch.arange(config.max_frames, dtype=torch.long), persistent=True
        )

    def output_frame_count(self, waveform_samples: int) -> int:
        mel_frames = self.mel.frame_count(waveform_samples)
        return (mel_frames + 2 * 1 - 3) // self.config.conv_stride + 1

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError("waveform must be [batch, samples]")
        features = self.mel(waveform)  # [batch, n_mels, mel_frames]
        x = self.gelu(self.conv1(features))
        x = self.gelu(self.conv2(x))  # [batch, hidden_size, frames]
        x = x.transpose(1, 2)  # [batch, frames, hidden_size]
        num_frames = x.size(1)
        if num_frames > self.config.max_frames:
            raise ValueError(
                f"waveform produced {num_frames} frames, exceeding max_frames="
                f"{self.config.max_frames}"
            )
        route_ids = self.route_ids[:num_frames].unsqueeze(0).expand(x.size(0), -1)
        for block in self.blocks:
            x = block(x, route_ids)
        return self.norm(x)
