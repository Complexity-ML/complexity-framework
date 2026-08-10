"""Configuration for TR-Hash speech-to-text and text-to-speech models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from complexity.config import ModelConfig

from .encoder import AudioEncoderConfig


@dataclass(frozen=True)
class TRHashSpeechToTextConfig:
    """A compact ASR model: audio encoder prefix + a canonical TR-Hash decoder."""

    sample_rate: int = 16_000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    audio_hidden_size: int = 384
    audio_layers: int = 6
    audio_heads: int = 6
    audio_num_experts: int = 4
    audio_top_k: int = 2
    audio_shared_width: int = 0
    audio_expert_width: int = 96
    audio_max_frames: int = 1_500
    num_audio_tokens: int = 64
    vocab_size: int = 32_000
    hidden_size: int = 768
    decoder_layers: int = 16
    attention_heads: int = 12
    key_value_heads: int = 4
    max_position_embeddings: int = 2_048
    num_experts: int = 4
    top_k: int = 2
    shared_width: int = 1_536
    routed_width: int = 2_048
    route_seed: int = 179_424_673
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.audio_hidden_size % self.audio_heads:
            raise ValueError("audio_hidden_size must be divisible by audio_heads")
        if self.audio_top_k > self.audio_num_experts:
            raise ValueError("audio_top_k cannot exceed audio_num_experts")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("attention_heads must be divisible by key_value_heads")
        if not 0 < self.num_audio_tokens < self.vocab_size:
            raise ValueError("num_audio_tokens must lie in (0, vocab_size)")
        if self.routed_width % self.num_experts:
            raise ValueError("routed_width must be divisible by num_experts")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")

    def audio_encoder_config(self) -> AudioEncoderConfig:
        """Build the TR-Hash MoE audio encoder configuration."""

        return AudioEncoderConfig(
            sample_rate=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            hidden_size=self.audio_hidden_size,
            num_hidden_layers=self.audio_layers,
            num_attention_heads=self.audio_heads,
            num_experts=self.audio_num_experts,
            top_k=self.audio_top_k,
            shared_width=self.audio_shared_width,
            expert_width=self.audio_expert_width,
            route_seed=self.route_seed,
            max_frames=self.audio_max_frames,
            dropout=self.dropout,
        )

    def decoder_config(self) -> ModelConfig:
        return ModelConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            num_hidden_layers=self.decoder_layers,
            num_attention_heads=self.attention_heads,
            num_key_value_heads=self.key_value_heads,
            intermediate_size=self.routed_width,
            max_position_embeddings=self.max_position_embeddings,
            attention_type="gqa",
            mlp_type="tr_hash_engine",
            num_experts=self.num_experts,
            top_k=self.top_k,
            top_k_primary_weight=0.5,
            routing_strategy="token_id_balanced_hash",
            shared_expert=True,
            shared_intermediate_size=self.shared_width,
            attention_dropout=self.dropout,
            norm_type="rmsnorm",
            tie_word_embeddings=True,
            use_sdpa=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashSpeechToTextConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown speech-to-text config fields: {unknown}")
        return cls(**values)


@dataclass(frozen=True)
class TRHashAudioConfig:
    """Architecture contract for a text-conditioned latent audio flow model (TTS).

    Audio is represented directly as normalized log-mel spectrogram frames —
    there is no separate learned audio codec/VAE, unlike
    ``TRHashImageConfig`` which assumes an external image VAE. A mel
    spectrogram is already a compact enough representation that a codec
    isn't needed to make the flow-matching sequence length tractable.
    """

    sample_rate: int = 16_000
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    frame_patch_size: int = 4
    max_audio_frames: int = 1_600
    vocab_size: int = 32_000
    max_text_length: int = 128
    text_hidden_size: int = 384
    text_layers: int = 4
    text_heads: int = 6
    hidden_size: int = 768
    num_layers: int = 14
    num_attention_heads: int = 12
    num_experts: int = 4
    top_k: int = 2
    shared_width: int = 1_536
    expert_width: int = 96
    time_buckets: int = 256
    route_seed: int = 0x71D5A17
    caption_dropout: float = 0.1
    attention_dropout: float = 0.0
    layer_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        positive = (
            "sample_rate",
            "n_mels",
            "n_fft",
            "hop_length",
            "frame_patch_size",
            "max_audio_frames",
            "vocab_size",
            "max_text_length",
            "text_hidden_size",
            "text_layers",
            "text_heads",
            "hidden_size",
            "num_layers",
            "num_attention_heads",
            "num_experts",
            "top_k",
            "shared_width",
            "expert_width",
            "time_buckets",
        )
        for name in positive:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_audio_frames % self.frame_patch_size:
            raise ValueError("max_audio_frames must be divisible by frame_patch_size")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.text_hidden_size % self.text_heads:
            raise ValueError("text_hidden_size must be divisible by text_heads")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if not 0.0 <= self.caption_dropout < 1.0:
            raise ValueError("caption_dropout must be in [0, 1)")

    @property
    def audio_token_count(self) -> int:
        return self.max_audio_frames // self.frame_patch_size

    @property
    def route_vocab_size(self) -> int:
        return self.audio_token_count * self.time_buckets

    @property
    def latent_patch_features(self) -> int:
        return self.n_mels * self.frame_patch_size

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashAudioConfig":
        return cls(**values)
