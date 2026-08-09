"""Configuration for direct image-and-text to text generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Dict

from complexity.config import ModelConfig


@dataclass(frozen=True)
class TRHashVisionLanguageConfig:
    """A compact VLM with a ViT prefix and a canonical TR-Hash decoder."""

    image_size: int = 224
    patch_size: int = 16
    vision_hidden_size: int = 384
    vision_layers: int = 6
    vision_heads: int = 6
    num_visual_tokens: int = 64
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
    route_seed: int = 73_856_093
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.image_size <= 0 or self.image_size % self.patch_size:
            raise ValueError("image_size must be positive and divisible by patch_size")
        if self.vision_hidden_size % self.vision_heads:
            raise ValueError("vision_hidden_size must be divisible by vision_heads")
        if self.hidden_size % self.attention_heads:
            raise ValueError("hidden_size must be divisible by attention_heads")
        if self.attention_heads % self.key_value_heads:
            raise ValueError("attention_heads must be divisible by key_value_heads")
        if not 0 < self.num_visual_tokens < self.vocab_size:
            raise ValueError("num_visual_tokens must lie in (0, vocab_size)")
        if self.routed_width % self.num_experts:
            raise ValueError("routed_width must be divisible by num_experts")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("top_k must lie in [1, num_experts]")

    def decoder_config(self) -> ModelConfig:
        """Build the native causal decoder configuration."""

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
    def from_dict(cls, values: Dict[str, Any]) -> "TRHashVisionLanguageConfig":
        allowed = {item.name for item in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown vision-language config fields: {unknown}")
        return cls(**values)
