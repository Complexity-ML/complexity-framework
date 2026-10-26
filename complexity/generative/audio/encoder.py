"""TR-Hash audio encoder with contextual TR-HASH routing by frame identity.

Raw post-convolution frame positions feed GQA and the two-lag expert MLP.
The route-ID space is sized by max_frames, independently of input length."""

from __future__ import annotations
from complexity.tr_hash.attention import resolve_domain_heads, resolve_domain_widths
from dataclasses import asdict, dataclass, fields, field
import torch
import torch.nn as nn
from complexity.tr_hash.attention import DomainSelfAttention, make_domain_mlp
from complexity.tr_hash import TRHashPrecision
from .mel import LogMelSpectrogram


@dataclass
class AudioEncoderConfig:
    sample_rate: int = 16000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    hidden_size: int = 384
    num_hidden_layers: int = 6
    num_attention_heads: int = 4
    num_key_value_heads: int | None = field(default=None, kw_only=True)
    num_experts: int = 4
    top_k: int = 2
    shared_width: int | None = None
    expert_width: int | None = None
    max_frames: int = 1500
    dropout: float = 0.0
    layer_norm_eps: float = 1e-06
    conv_stride: int = 2

    def __post_init__(self) -> None:
        resolve_domain_heads(self, heads="num_attention_heads", kv="num_key_value_heads")
        resolve_domain_widths(
            self, hidden="hidden_size", shared="shared_width", expert="expert_width"
        )
        if self.num_experts != 4 or self.top_k != 2:
            raise ValueError("TR-HASH requires four experts and top_k=2")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.max_frames <= 0:
            raise ValueError("max_frames must be positive")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown encoder config fields: {sorted(unknown)}")
        return cls(**values)

    @property
    def route_vocab_size(self) -> int:
        return self.max_frames


class _AudioSelfAttention(DomainSelfAttention):
    pass


class AudioBlock(nn.Module):

    def __init__(self, config: AudioEncoderConfig, layer_index: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = _AudioSelfAttention(
            config.hidden_size,
            config.num_attention_heads,
            config.dropout,
            num_key_value_heads=config.num_key_value_heads,
            layer_index=layer_index,
        )
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = make_domain_mlp(
            hidden_size=config.hidden_size,
            vocab_size=config.route_vocab_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
            shared_width=config.shared_width,
            expert_width=config.expert_width,
            layer_index=layer_index,
            precision=TRHashPrecision.BF16,
        )

    def forward(self, x: torch.Tensor, route_ids: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), route_ids)
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
            (AudioBlock(config, layer_index=index) for index in range(config.num_hidden_layers))
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
        features = self.mel(waveform)
        x = self.gelu(self.conv1(features))
        x = self.gelu(self.conv2(x))
        x = x.transpose(1, 2)
        num_frames = x.size(1)
        if num_frames > self.config.max_frames:
            raise ValueError(
                f"waveform produced {num_frames} frames, exceeding max_frames={self.config.max_frames}"
            )
        route_ids = self.route_ids[:num_frames].unsqueeze(0).expand(x.size(0), -1)
        for block in self.blocks:
            x = block(x, route_ids)
        return self.norm(x)
