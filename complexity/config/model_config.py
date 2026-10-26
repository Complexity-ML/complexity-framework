"""
Unified Model Configuration for framework-complexity.

This is the single source of truth for model architecture configuration.
Users can define any architecture by setting these parameters.
"""

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml


@dataclass
class ModelConfig:
    """Architecture settings for the canonical TR-HASH model.

    Four query heads share one key/value head. Four routed SwiGLU experts
    use two fixed equal routes alongside the dense shared expert.
    """

    # === Model Architecture ===
    hidden_size: int = 768
    num_hidden_layers: int = 12
    intermediate_size: Optional[int] = None  # TR-HASH: hidden_size; generic: 4x width.
    vocab_size: int = 32000

    # === Attention ===
    num_attention_heads: Optional[int] = None  # TR-HASH: 4; four query heads.
    num_key_value_heads: Optional[int] = None  # One key/value head for TR-HASH.
    attention_type: str = "gqa"  # gqa, mha, mqa
    attention_dropout: float = 0.0
    use_qk_norm: bool = True
    sliding_window: Optional[int] = None  # None = full attention
    causal_conv_kernel_size: int = 4
    causal_conv_dilation_cycle: int = 8
    causal_state_rank: int = 16
    causal_context_gate_init: float = 1.0
    causal_contextual_mix_init: float = 0.0
    causal_context_fusion_size: int = 0
    causal_stable_delta: bool = False
    causal_delta_chunk_size: int = 512
    causal_delta_timescales: int = 1
    causal_delta_collision_normalized: bool = False
    causal_delta_lexical_values: bool = False
    causal_delta_lexical_forge: bool = False
    causal_delta_occurrence_address: bool = False
    tr_mha_num_experts: int = 4
    tr_mha_adapter_rank: int = 8
    tr_mha_top_k: int = 2
    tr_mha_adapter_gate_init: float = 0.1
    tr_mha_id_primary_logit: float = 2.0
    tr_mha_id_secondary_logit: float = 1.0
    tr_mha_id_other_logit: float = -2.0
    tr_mha_verifier_gate_init: float = 0.1
    tr_mha_verifier_temperature: float = 1.0
    tr_mha_targets: str = "qv"

    # === Position Embeddings ===
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rope_type: str = "standard"  # standard, partial_rope, yarn, dynamic_ntk
    rope_fraction: float = 1.0  # Fraction of head_dim to apply RoPE to (1.0 = full, 0.5 = partial)

    # === MLP / FFN ===
    mlp_type: str = "tr_hash_engine"  # tr_hash_engine (default) or tr_hash_moe; four routed experts
    hidden_act: str = "silu"  # silu, gelu, relu

    # === MoE (Token-Routed) ===
    num_experts: Optional[int] = None  # TR-HASH: four routed experts plus the shared expert.
    expert_initialization: str = "gpt_normal"  # TR-HASH requires gpt_normal; other initializers apply only to generic components.
    # Frequency tables are generic component settings and are rejected by TR-HASH.
    # TR-HASH routes use the original ID and its two causal predecessors.
    token_frequencies: Optional[torch.Tensor] = None
    routing_strategy: Optional[str] = None  # TR-HASH: token_id_composed_modulo.
    # Generic-only routing/gate options below cannot alter the canonical TR-HASH recipe.
    lsh_routing: bool = (
        False  # Route on a fixed random-hyperplane hash of h (semantic), not the token id
    )
    lsh_bits: int = 0  # Number of hyperplanes (0 = ceil(log2(num_experts)))
    lsh_from_layer: int = (
        0  # LSH routing only for layers >= this index; earlier layers stay lexical
    )
    lsh_threshold_mode: str = (
        "zero"  # zero is stable for inference; batch_median maximises training-batch balance.
    )
    shared_expert: bool = True  # Shared lexical expert: dense MLP + routed experts
    shared_intermediate_size: Optional[int] = (
        None  # TR-HASH shared width defaults to 2 * hidden_size
    )
    shared_expert_chunk_tokens: int = (
        0  # 0 = one dense pass; >0 chunks token dimension to reduce shared SwiGLU activation peak.
    )
    use_shared_routed_gates: bool = (
        False  # Generic component option; learned branch gates are rejected by TR-HASH.
    )
    shared_gate_init: float = 1.0  # Initial multiplier for shared expert output
    routed_gate_init: float = 1.0  # Initial multiplier for routed expert output
    shared_output_scale: float = 1.0  # Fixed, non-parameter shared branch multiplier
    routed_output_scale: Optional[float] = (
        None  # TR-HASH uses a fixed routed branch multiplier of 2.
    )
    shared_output_scale_first_layer: Optional[float] = None
    shared_output_scale_last_layer: Optional[float] = None
    routed_output_scale_first_layer: Optional[float] = None
    routed_output_scale_last_layer: Optional[float] = None
    top_k: Optional[int] = None  # TR-HASH: two deterministic routes.
    top_k_primary_weight: Optional[float] = None  # TR-HASH fixes the two weights at 0.5.
    learn_hash_pair_gates: bool = False  # Learn one mixture scalar per fixed unordered hash pair
    hash_pair_gate_init: float = (
        0.5  # Expert-a weight; 0.5 preserves the equal-route initialization
    )
    learn_hash_channel_modulation: bool = (
        False  # Fixed token-ID hash selects signs for learned expert/channel gains
    )
    hash_channel_scale_init: float = 0.0  # Zero preserves the baseline exactly at initialization
    static_expert_capacity: bool = (
        False  # Use fixed per-expert dispatch capacity for torch.export / pipeline tracing
    )
    use_custom_kernels: Any = (
        "auto"  # "auto", True, or False. ROCm defaults to PyTorch fallback in auto mode.
    )
    collect_moe_telemetry: bool = (
        False  # Per-layer expert/RMS diagnostics. Disabled by default for throughput.
    )
    use_cggr: Any = "auto"  # "auto", True, or False. CGGR grouped-GEMM Triton path for TR-Hash MoE when custom Triton is available.
    # Dynamic TR-Hash capacity (tr_hash_engine/tr_hash_moe only): start already
    # shrunk to a smaller deterministic ID/hash-routed sub-network of the
    # allocated (num_experts, routed width). None = full capacity. Adjustable
    # afterwards at runtime via the MLP's set_active_experts()/set_active_expert_width().
    active_num_experts: Optional[int] = None
    active_expert_width: Optional[int] = None
    # Native exports record non-uniform runtime controls without adding model
    # tensors or changing historical checkpoint layouts.
    layer_mlp_controls: Optional[list[Dict[str, Any]]] = None

    # === Lexical feature-modulated residual ===
    lexical_object_rank: int = 16
    lexical_object_gate_init: float = 0.1
    tie_lexical_object_embeddings: bool = False
    micro_num_experts: int = 4
    micro_expert_width: int = 16
    micro_expert_gate_init: float = 0.1
    lexical_gqa_rank: int = 16
    lexical_gqa_gate_init: float = 0.0
    lexical_gqa_use_token_code: bool = True
    lexical_key_gate_init: float = 0.05
    lexical_zipf_path: Optional[str] = None
    lexical_zipf_mode: str = "uniform"
    lexical_zipf_alpha: float = 0.25
    lexical_zipf_floor: float = 0.1
    lexical_zipf_permutation_seed: int = 1729

    # === Normalization ===
    norm_type: str = "rmsnorm"  # rmsnorm, layernorm
    norm_eps: float = 1e-6

    # === Embeddings ===
    tie_word_embeddings: bool = True

    # === Training ===
    use_sdpa: bool = True  # Use Flash Attention via SDPA
    use_cache: bool = True  # KV cache for generation
    is_causal: bool = True  # False = bidirectional (encoder-style) attention, for embedding models

    # === Initialization ===
    initializer_range: float = 0.02

    # === μP (Maximal Update Parametrization, Yang et al. 2022) ===
    # When use_mup_init=True, hidden→hidden Linears (FFN gate/up/down,
    # attention Q/K/V/O) are initialised with std = initializer_range /
    # √(hidden_size / mup_base_width). Embeddings keep their base std.
    # Combined with use_mup_attn_scale and use_mup_output_mult and the
    # adamw_mup optimiser (LR side), this enables hyper-parameter
    # transfer from a small proxy width (mup_base_width) to wider
    # variants without re-tuning.
    use_mup_init: bool = False
    use_mup_attn_scale: bool = False  # Attention logits / d_head (vs / √d_head)
    use_mup_output_mult: bool = False  # Divide lm_head output by width_mult
    mup_base_width: int = 256  # Reference width (proxy size)

    # === Extra (for custom extensions) ===
    extra_config: Dict[str, Any] = field(default_factory=dict)

    _removed_mlp_types = {
        "standard",
        "gelu",
        "swiglu",
        "silu",
        "llama",
        "geglu",
        "mixtral",
        "learned_router",
        "standard_moe",
        "i64_swiglu",
        "integer_swiglu",
        "i64_token_routed",
        "integer_moe",
        "token_routed",
        "sort_split",
        "sort_split_moe",
        "deterministic_moe",
        "complexity",
    }

    def __post_init__(self):
        """Validate and set defaults."""
        tr_hash = self.mlp_type in {"tr_hash_engine", "tr_hash_moe"}
        if self.num_experts is None:
            self.num_experts = 4 if tr_hash else 1
        if self.top_k is None:
            self.top_k = 2 if tr_hash else 1
        if self.routing_strategy is None:
            self.routing_strategy = "token_id_composed_modulo"
        if self.num_attention_heads is None:
            self.num_attention_heads = 4
        if self.routed_output_scale is None:
            self.routed_output_scale = 2.0 if tr_hash else 1.0
        if tr_hash:
            if (self.num_experts, self.top_k) != (4, 2):
                raise ValueError("TR-HASH requires four experts and top_k=2")
            if self.attention_type != "gqa":
                raise ValueError("TR-HASH requires GQA attention")
            if self.num_attention_heads != 4 or self.num_key_value_heads not in {None, 1}:
                raise ValueError("TR-HASH requires four query heads and one key/value head")
            if self.hidden_size % 4:
                raise ValueError("TR-HASH value binding requires hidden_size divisible by four")
            if self.top_k_primary_weight is None:
                self.top_k_primary_weight = 0.5
            if self.shared_intermediate_size is None:
                self.shared_intermediate_size = 2 * self.hidden_size
            if not self.shared_expert or not self.use_qk_norm or self.norm_type != "rmsnorm":
                raise ValueError("TR-HASH requires a shared expert and RMS normalization")
            if self.expert_initialization != "gpt_normal" or self.hidden_act != "silu":
                raise ValueError("TR-HASH requires normal initialization and SwiGLU")
            if (
                self.top_k_primary_weight != 0.5
                or self.shared_output_scale != 1.0
                or self.routed_output_scale != 2.0
            ):
                raise ValueError("TR-HASH requires fixed 50/50 routing and branch scales 1/2")
            if any(
                value is not None
                for value in (
                    self.shared_output_scale_first_layer,
                    self.shared_output_scale_last_layer,
                    self.routed_output_scale_first_layer,
                    self.routed_output_scale_last_layer,
                )
            ):
                raise ValueError("TR-HASH branch scales are fixed across layers")
            if self.active_num_experts not in {None, 4}:
                raise ValueError("TR-HASH requires four active experts")
            if (
                self.lsh_routing
                or self.learn_hash_pair_gates
                or self.learn_hash_channel_modulation
                or self.use_shared_routed_gates
                or self.token_frequencies is not None
            ):
                raise ValueError("TR-HASH uses fixed token-ID routing and fixed branch weights")
        if self.mlp_type in self._removed_mlp_types:
            raise ValueError(f"mlp_type={self.mlp_type!r} is unsupported; use tr_hash_engine")

        # Auto-compute intermediate_size
        if self.intermediate_size is None:
            if tr_hash:
                self.intermediate_size = self.hidden_size
            else:
                # Standard FFN uses 4x
                self.intermediate_size = self.hidden_size * 4

        # Default num_key_value_heads for GQA
        if self.num_key_value_heads is None:
            if tr_hash or self.attention_type == "mqa":
                self.num_key_value_heads = 1
            elif self.attention_type in {
                "mha",
                "tr_mha",
                "token_routed_mha",
                "tr_mha_v2",
                "token_routed_mha_v2",
            }:
                self.num_key_value_heads = self.num_attention_heads
            else:
                # GQA default: 1/4 of heads (like Llama 2)
                self.num_key_value_heads = max(1, self.num_attention_heads // 4)

        # Validation
        self._validate()

    def _validate(self):
        """Validate configuration."""
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.intermediate_size is None or self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.shared_output_scale < 0.0:
            raise ValueError("shared_output_scale must be non-negative")
        if self.routed_output_scale < 0.0:
            raise ValueError("routed_output_scale must be non-negative")
        for name in (
            "shared_output_scale_first_layer",
            "shared_output_scale_last_layer",
            "routed_output_scale_first_layer",
            "routed_output_scale_last_layer",
        ):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.num_key_value_heads is None or self.num_key_value_heads <= 0:
            raise ValueError("num_key_value_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by num_key_value_heads ({self.num_key_value_heads})"
            )
        if (
            self.attention_type
            in {
                "tr_mha",
                "token_routed_mha",
                "tr_mha_v2",
                "token_routed_mha_v2",
            }
            and self.num_key_value_heads != self.num_attention_heads
        ):
            raise ValueError(
                "TR-MHA requires one K/V head per query head "
                "(num_key_value_heads == num_attention_heads)"
            )
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if not 0.0 < self.rope_fraction <= 1.0:
            raise ValueError("rope_fraction must be in (0, 1]")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when set")
        if self.causal_conv_kernel_size <= 0:
            raise ValueError("causal_conv_kernel_size must be positive")
        if self.causal_conv_dilation_cycle <= 0:
            raise ValueError("causal_conv_dilation_cycle must be positive")
        if self.causal_state_rank <= 0:
            raise ValueError("causal_state_rank must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.expert_initialization not in {"gpt_normal", "legacy_kaiming"}:
            raise ValueError("expert_initialization must be 'gpt_normal' or 'legacy_kaiming'")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.top_k_primary_weight is not None and not 0.0 <= self.top_k_primary_weight <= 1.0:
            raise ValueError("top_k_primary_weight must be in [0, 1]")
        if self.layer_mlp_controls is not None:
            if self.mlp_type not in {"tr_hash_engine", "tr_hash_moe"}:
                raise ValueError("layer_mlp_controls requires a TR-Hash engine MLP")
            if (
                not isinstance(self.layer_mlp_controls, list)
                or len(self.layer_mlp_controls) != self.num_hidden_layers
            ):
                raise ValueError("layer_mlp_controls must contain one entry per layer")
            keys = {"top_k_primary_weight", "active_num_experts", "active_expert_width"}
            for control in self.layer_mlp_controls:
                if not isinstance(control, dict) or set(control) != keys:
                    raise ValueError(
                        "layer_mlp_controls requires exact routing and capacity fields"
                    )
                weight = control["top_k_primary_weight"]
                experts = control["active_num_experts"]
                width = control["active_expert_width"]
                if type(weight) not in {float, int} or not math.isfinite(weight) or weight != 0.5:
                    raise ValueError(
                        "layer_mlp_controls primary weight must be finite and in [0, 1]"
                    )
                if type(experts) is not int or experts != 4:
                    raise ValueError(
                        "layer_mlp_controls expert count exceeds allocated routing capacity"
                    )
                if (
                    type(width) is not int
                    or not 0 < width <= self.intermediate_size // self.num_experts
                ):
                    raise ValueError("layer_mlp_controls expert width exceeds allocated capacity")
        if not 0.0 < self.hash_pair_gate_init < 1.0:
            raise ValueError("hash_pair_gate_init must be strictly between 0 and 1")
        if abs(self.hash_channel_scale_init) > 1.0:
            raise ValueError("hash_channel_scale_init must be in [-1, 1]")
        if self.routing_strategy != "token_id_composed_modulo":
            raise ValueError("Only token_id_composed_modulo routing is supported")
        if self.lsh_threshold_mode not in {"batch_median", "zero"}:
            raise ValueError("lsh_threshold_mode must be 'batch_median' or 'zero'")
        if self.shared_intermediate_size is not None and self.shared_intermediate_size <= 0:
            raise ValueError("shared_intermediate_size must be positive when set")
        if self.shared_expert_chunk_tokens < 0:
            raise ValueError("shared_expert_chunk_tokens must be non-negative")
        if self.lexical_object_rank <= 0:
            raise ValueError("lexical_object_rank must be positive")
        if self.micro_num_experts <= 0:
            raise ValueError("micro_num_experts must be positive")
        if self.micro_expert_width <= 0:
            raise ValueError("micro_expert_width must be positive")
        if self.tr_mha_num_experts <= 0:
            raise ValueError("tr_mha_num_experts must be positive")
        if self.tr_mha_adapter_rank <= 0:
            raise ValueError("tr_mha_adapter_rank must be positive")
        if not 0 < self.tr_mha_top_k <= self.tr_mha_num_experts:
            raise ValueError("tr_mha_top_k must be between 1 and tr_mha_num_experts")
        if not 0.0 < self.tr_mha_verifier_gate_init < 1.0:
            raise ValueError("tr_mha_verifier_gate_init must be in (0, 1)")
        if self.tr_mha_verifier_temperature <= 0.0:
            raise ValueError("tr_mha_verifier_temperature must be positive")
        if self.tr_mha_targets not in {"q", "v", "qv"}:
            raise ValueError("tr_mha_targets must be one of: q, v, qv")
        if self.mup_base_width <= 0:
            raise ValueError("mup_base_width must be positive")
        if self.token_frequencies is not None:
            if not isinstance(self.token_frequencies, torch.Tensor):
                raise ValueError("token_frequencies must be a torch.Tensor")
            if self.token_frequencies.ndim != 1:
                raise ValueError("token_frequencies must be a 1D tensor")
            if self.token_frequencies.numel() != self.vocab_size:
                raise ValueError(
                    f"token_frequencies length ({self.token_frequencies.numel()}) "
                    f"must match vocab_size ({self.vocab_size})"
                )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_size // self.num_attention_heads

    @property
    def num_kv_groups(self) -> int:
        """Number of query heads per KV head (for GQA)."""
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mup_width_mult(self) -> float:
        """μP width multiplier: hidden_size / mup_base_width.

        Returns 1.0 when at base width or μP disabled — in that case the
        scaling factor 1/√width_mult equals 1 and is a no-op.
        """
        if not getattr(self, "use_mup_init", False):
            return 1.0
        return float(self.hidden_size) / float(self.mup_base_width)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (skips non-serializable fields like Tensors)."""
        result = {}
        for k, v in self.__dict__.items():
            if k.startswith("_"):
                continue
            if k == "layer_mlp_controls" and v is None:
                continue  # Keep unmodified historical model dictionaries identical.
            if isinstance(v, torch.Tensor):
                continue  # token_frequencies etc. — not JSON serializable
            result[k] = v
        return result

    def save(self, path: str | Path):
        """Save config to file (JSON or YAML)."""
        data = self.to_dict()
        path = Path(path)
        with path.open("w", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                yaml.safe_dump(data, f, default_flow_style=False)
            else:
                json.dump(data, f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], *, strict: bool = False) -> "ModelConfig":
        """Read architecture settings.

        Use ``strict=True`` when authoring a new model to reject misspelled
        fields and routing strategies. The default retains compatibility with
        historical exports containing extra Hugging Face metadata.
        """
        import dataclasses

        if not isinstance(data, Mapping):
            raise ValueError("Model config must be a mapping")
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - valid_keys
        if strict and unknown:
            raise ValueError("Unknown model config keys: " + ", ".join(sorted(map(str, unknown))))
        if strict:
            from complexity._config_utils import validate_field_types

            validate_field_types(cls, data)
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        obsolete = {
            "architecture_profile",
            "architecture_profile_version",
            "route_hash_count",
        } & set(data)
        if obsolete:
            raise ValueError(
                "Obsolete TR-HASH configuration fields: " + ", ".join(sorted(obsolete))
            )
        return cls(**filtered)

    @classmethod
    def load(cls, path: str | Path, *, strict: bool = False) -> "ModelConfig":
        """Load config from file (JSON or YAML)."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            if path.suffix.lower() in {".yaml", ".yml"}:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        return cls.from_dict(data, strict=strict)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.to_dict().items() if v is not None)
        return f"ModelConfig({params})"


# Preset configurations
#
# This framework is scoped to TR-Hash MoE (deterministic token-ID / hash-table
# routing): dense reference architectures (Llama/Mistral/GPT-2-shaped presets)
# and the I64 integer-precision presets were removed and will return later as
# explicit comparison baselines.
def complexity_7b_config() -> ModelConfig:
    """Canonical TR-HASH size preset."""
    return ModelConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=11008,
        vocab_size=100000,
        max_position_embeddings=8192,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


# === Complexity Size Presets (TR-Hash: GQA + Token-Routed MoE) ===
def complexity_tiny_config() -> ModelConfig:
    """Canonical TR-HASH size preset."""
    return ModelConfig(
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=704,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_small_config() -> ModelConfig:
    """Canonical TR-HASH size preset."""
    return ModelConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=1408,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_100m_config() -> ModelConfig:
    """Canonical TR-HASH size preset."""
    return ModelConfig(
        hidden_size=640,
        num_hidden_layers=10,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=1824,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_base_config() -> ModelConfig:
    """Canonical TR-HASH size preset."""
    return ModelConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=2048,
        vocab_size=32000,
        max_position_embeddings=2048,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_large_config() -> ModelConfig:
    """Canonical TR-HASH size preset."""
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=2816,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


def complexity_xl_config() -> ModelConfig:
    """Canonical TR-HASH size preset."""
    return ModelConfig(
        hidden_size=2048,
        num_hidden_layers=24,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=5632,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="tr_hash_engine",
        num_experts=4,
        norm_type="rmsnorm",
        use_qk_norm=True,
    )


# Registry of preset configs
PRESET_CONFIGS = {
    # Complexity size ladder
    "complexity-tiny": complexity_tiny_config,
    "complexity-small": complexity_small_config,
    "complexity-100m": complexity_100m_config,
    "complexity-base": complexity_base_config,
    "complexity-large": complexity_large_config,
    "complexity-xl": complexity_xl_config,
    # Complexity with MoE
    "complexity-7b": complexity_7b_config,
    # Aliases
    "tiny": complexity_tiny_config,
    "small": complexity_small_config,
    "100m": complexity_100m_config,
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
