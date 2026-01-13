"""
Unified Model Configuration for framework-complexity.

This is the single source of truth for model architecture configuration.
Users can define any architecture by setting these parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Literal
import json
import yaml


@dataclass
class ModelConfig:
    """
    Unified configuration for all model architectures.

    This config supports:
    - Llama-style models (GQA, SwiGLU, RMSNorm)
    - Mistral-style models (sliding window attention)
    - GPT-style models (MHA, GELU, LayerNorm)
    - Complexity custom models (Token-Routed MoE)

    Example:
        # Llama-style
        config = ModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            num_key_value_heads=8,  # GQA
            attention_type="gqa",
            mlp_type="swiglu",
            norm_type="rmsnorm",
        )

        # Complexity with MoE
        config = ModelConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            mlp_type="token_routed",
            num_experts=4,
        )
    """

    # === Model Architecture ===
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: Optional[int] = None  # Auto: hidden_size * 4 (or 8/3 for SwiGLU)
    vocab_size: int = 32000

    # === Attention ===
    num_attention_heads: int = 12
    num_key_value_heads: Optional[int] = None  # None = MHA, < num_heads = GQA, 1 = MQA
    attention_type: str = "gqa"  # gqa, mha, mqa
    attention_dropout: float = 0.0
    use_qk_norm: bool = True
    sliding_window: Optional[int] = None  # None = full attention

    # === Position Embeddings ===
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_type: str = "standard"  # standard, yarn, dynamic_ntk

    # === MLP / FFN ===
    mlp_type: str = "swiglu"  # swiglu, gelu, token_routed
    hidden_act: str = "silu"  # silu, gelu, relu

    # === MoE (Token-Routed) ===
    num_experts: int = 1  # 1 = standard MLP, >1 = MoE

    # === INL Dynamics (Complexity innovation) ===
    use_inl_dynamics: bool = False  # Enable velocity tracking for stability
    inl_beta_max: float = 2.0  # Max beta for damping
    inl_velocity_max: float = 10.0  # Max velocity clamp

    # === Normalization ===
    norm_type: str = "rmsnorm"  # rmsnorm, layernorm
    norm_eps: float = 1e-6

    # === Embeddings ===
    tie_word_embeddings: bool = True

    # === Training ===
    use_sdpa: bool = True  # Use Flash Attention via SDPA
    use_cache: bool = True  # KV cache for generation

    # === Initialization ===
    initializer_range: float = 0.02

    # === Extra (for custom extensions) ===
    extra_config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and set defaults."""
        # Auto-compute intermediate_size
        if self.intermediate_size is None:
            if self.mlp_type in ["swiglu", "silu", "geglu", "token_routed"]:
                # SwiGLU uses 8/3 ratio (rounded to multiple of 256)
                self.intermediate_size = int(self.hidden_size * 8 / 3)
                self.intermediate_size = ((self.intermediate_size + 255) // 256) * 256
            else:
                # Standard FFN uses 4x
                self.intermediate_size = self.hidden_size * 4

        # Default num_key_value_heads for GQA
        if self.num_key_value_heads is None:
            if self.attention_type == "mqa":
                self.num_key_value_heads = 1
            elif self.attention_type == "mha":
                self.num_key_value_heads = self.num_attention_heads
            else:
                # GQA default: 1/4 of heads (like Llama 2)
                self.num_key_value_heads = max(1, self.num_attention_heads // 4)

        # Validation
        self._validate()

    def _validate(self):
        """Validate configuration."""
        assert self.hidden_size > 0, "hidden_size must be positive"
        assert self.num_hidden_layers > 0, "num_hidden_layers must be positive"
        assert self.num_attention_heads > 0, "num_attention_heads must be positive"
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    def save(self, path: str):
        """Save config to file (JSON or YAML)."""
        data = self.to_dict()
        with open(path, "w") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                yaml.dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelConfig":
        """Create config from dictionary."""
        return cls(**data)

    @classmethod
    def load(cls, path: str) -> "ModelConfig":
        """Load config from file (JSON or YAML)."""
        with open(path, "r") as f:
            if path.endswith(".yaml") or path.endswith(".yml"):
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items() if v is not None)
        return f"ModelConfig({params})"


# Preset configurations
def llama_7b_config() -> ModelConfig:
    """Llama 2 7B configuration."""
    return ModelConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,  # Llama 2 7B uses MHA
        intermediate_size=11008,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="mha",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        rope_theta=10000.0,
    )


def llama_70b_config() -> ModelConfig:
    """Llama 2 70B configuration (GQA)."""
    return ModelConfig(
        hidden_size=8192,
        num_hidden_layers=80,
        num_attention_heads=64,
        num_key_value_heads=8,  # GQA with 8 KV heads
        intermediate_size=28672,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        rope_theta=10000.0,
    )


def mistral_7b_config() -> ModelConfig:
    """Mistral 7B configuration (sliding window)."""
    return ModelConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,  # GQA
        intermediate_size=14336,
        vocab_size=32000,
        max_position_embeddings=32768,
        sliding_window=4096,  # Sliding window attention
        attention_type="gqa",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        rope_theta=10000.0,
    )


def complexity_7b_config() -> ModelConfig:
    """Complexity 7B with Token-Routed MoE."""
    return ModelConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=11008,
        vocab_size=100000,
        max_position_embeddings=8192,
        attention_type="gqa",
        mlp_type="token_routed",  # Complexity innovation!
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def gpt2_config() -> ModelConfig:
    """GPT-2 Small configuration."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=12,  # MHA
        intermediate_size=3072,
        vocab_size=50257,
        max_position_embeddings=1024,
        attention_type="mha",
        mlp_type="gelu",
        norm_type="layernorm",
        hidden_act="gelu",
        use_qk_norm=False,
    )


# === Complexity Size Presets ===
def complexity_tiny_config() -> ModelConfig:
    """Complexity Tiny (~15M params) - for testing and debugging."""
    return ModelConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=704,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_small_config() -> ModelConfig:
    """Complexity Small (~50M params) - for rapid prototyping."""
    return ModelConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_key_value_heads=4,
        intermediate_size=1408,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_base_config() -> ModelConfig:
    """Complexity Base (~125M params) - balanced size for training."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_key_value_heads=4,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_large_config() -> ModelConfig:
    """Complexity Large (~350M params) - for serious experiments."""
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2816,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_xl_config() -> ModelConfig:
    """Complexity XL (~1B params) - large scale training."""
    return ModelConfig(
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=5632,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="swiglu",
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


# Registry of preset configs
PRESET_CONFIGS = {
    # Complexity size ladder
    "complexity-tiny": complexity_tiny_config,
    "complexity-small": complexity_small_config,
    "complexity-base": complexity_base_config,
    "complexity-large": complexity_large_config,
    "complexity-xl": complexity_xl_config,
    # Complexity with MoE
    "complexity-7b": complexity_7b_config,
    # Reference architectures
    "llama-7b": llama_7b_config,
    "llama-70b": llama_70b_config,
    "mistral-7b": mistral_7b_config,
    "gpt2": gpt2_config,
    # Aliases
    "tiny": complexity_tiny_config,
    "small": complexity_small_config,
    "base": complexity_base_config,
    "large": complexity_large_config,
    "xl": complexity_xl_config,
}


def get_preset(name: str) -> ModelConfig:
    """Get a preset configuration by name."""
    if name not in PRESET_CONFIGS:
        available = list(PRESET_CONFIGS.keys())
        raise ValueError(f"Unknown preset: {name}. Available: {available}")
    return PRESET_CONFIGS[name]()
