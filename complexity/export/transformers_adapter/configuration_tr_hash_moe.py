"""Hugging Face configuration for TR-HASH deterministic MoE models.

This file is intentionally self-contained so it can be copied to the root of
a Hub model repository and loaded with ``trust_remote_code=True``.
"""

from __future__ import annotations

try:
    from transformers import PretrainedConfig
except ModuleNotFoundError as exc:
    if exc.name != "transformers":
        raise
    raise ImportError(
        'The TR-HASH Transformers adapter requires transformers. '
        'Install with: pip install "complexity-framework[hf]"'
    ) from exc


class TRHashConfig(PretrainedConfig):
    """Configuration for the public TR-HASH decoder-only checkpoints."""

    model_type = "tr_hash_moe"
    keys_to_ignore_at_inference = ["past_key_values"]

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if config_dict.get("routing_strategy") != "token_id_composed_modulo":
            raise ValueError(
                "TR-HASH checkpoints must explicitly declare routing_strategy='token_id_composed_modulo'"
            )
        return super().from_dict(config_dict, **kwargs)

    def __init__(
        self,
        hidden_size: int = 896,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 4,
        num_key_value_heads: int = 1,
        intermediate_size: int | None = None,
        shared_intermediate_size: int | None = None,
        vocab_size: int = 32000,
        max_position_embeddings: int = 2048,
        attention_type: str = "gqa",
        mlp_type: str = "tr_hash_engine",
        num_experts: int = 4,
        num_experts_per_tok: int = 2,
        top_k_primary_weight: float | None = 0.5,
        routing_strategy: str = "token_id_composed_modulo",
        shared_expert: bool = True,
        shared_output_scale: float = 1.0,
        routed_output_scale: float = 2.0,
        use_qk_norm: bool = True,
        norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        attention_dropout: float = 0.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        initializer_range: float = 0.02,
        **kwargs,
    ) -> None:
        # Accept native pre-adapter configs while keeping architectural top-k
        # separate from Transformers' generation ``top_k`` sampling option.
        native_top_k = kwargs.pop("top_k", None)
        if native_top_k is not None:
            num_experts_per_tok = int(native_top_k)
        intermediate_size = hidden_size if intermediate_size is None else intermediate_size
        shared_intermediate_size = 2 * hidden_size if shared_intermediate_size is None else shared_intermediate_size
        if (routing_strategy != "token_id_composed_modulo" or num_experts != 4 or num_experts_per_tok != 2
            or top_k_primary_weight != 0.5 or not shared_expert or shared_output_scale != 1.0
            or routed_output_scale != 2.0 or not use_qk_norm):
            raise ValueError("TR-HASH requires composed modulo, four experts, top-2 equal weights, shared branch, scales 1/2 and Q/K normalization")
        if hidden_size % 4:
            raise ValueError("hidden_size must be divisible by four for V channel binding")
        if attention_type != "gqa":
            raise ValueError("The public TR-HASH adapter currently supports GQA only")
        if mlp_type not in {"tr_hash_engine", "tr_hash_moe"}:
            raise ValueError("TRHashConfig requires mlp_type='tr_hash_engine'")
        if num_attention_heads != 4 or num_key_value_heads != 1:
            raise ValueError("TR-HASH requires four query heads and one key/value head")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if intermediate_size % num_experts:
            raise ValueError("intermediate_size must be divisible by num_experts")
        if not 1 <= num_experts_per_tok <= num_experts:
            raise ValueError("num_experts_per_tok must be between 1 and num_experts")

        self.hidden_size = int(hidden_size)
        self.num_hidden_layers = int(num_hidden_layers)
        self.num_attention_heads = int(num_attention_heads)
        self.num_key_value_heads = int(num_key_value_heads)
        self.intermediate_size = int(intermediate_size)
        self.shared_intermediate_size = int(shared_intermediate_size)
        self.vocab_size = int(vocab_size)
        self.max_position_embeddings = int(max_position_embeddings)
        self.attention_type = attention_type
        self.mlp_type = mlp_type
        self.num_experts = int(num_experts)
        self.num_experts_per_tok = int(num_experts_per_tok)
        self.top_k_primary_weight = top_k_primary_weight
        self.routing_strategy = routing_strategy
        self.shared_expert = bool(shared_expert)
        self.shared_output_scale = float(shared_output_scale)
        self.routed_output_scale = float(routed_output_scale)
        self.use_qk_norm = bool(use_qk_norm)
        self.norm_eps = float(norm_eps)
        self.rope_theta = float(rope_theta)
        self.attention_dropout = float(attention_dropout)
        self.initializer_range = float(initializer_range)
        self.use_cache = bool(use_cache)
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.expert_width = self.intermediate_size // self.num_experts

        kwargs.setdefault("architectures", ["TRHashForCausalLM"])
        # These are emitted by save_pretrained/export and therefore come back
        # through **kwargs on reload. Consume them before passing the canonical
        # values below so repeated save/load cycles stay valid.
        kwargs.pop("is_decoder", None)
        kwargs.pop("is_encoder_decoder", None)
        super().__init__(
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            is_decoder=True,
            is_encoder_decoder=False,
            **kwargs,
        )
        # Transformers 5 no longer materializes every model-specific keyword
        # passed to PretrainedConfig, while Transformers 4 did. Keep the
        # decoder cache contract explicit across both release lines.
        self.use_cache = bool(use_cache)


__all__ = ["TRHashConfig"]
