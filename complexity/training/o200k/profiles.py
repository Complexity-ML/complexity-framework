"""Model profiles for the o200k Token-Routed pretraining runner."""

from __future__ import annotations

from complexity.config import ModelConfig


PROFILES = {
    "50m": {
        "hidden_size": 224,
        "num_hidden_layers": 8,
        "num_attention_heads": 7,
        "num_key_value_heads": 1,
        "intermediate_size": 128,
        "shared_intermediate_size": 1024,
        "run_name": "50m-o200k-tr-local",
        "save_dir": "checkpoints/50m-o200k-tr-local",
        "description": "50M o200k TR",
    },
    "100m": {
        "hidden_size": 384,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "intermediate_size": 128,
        "shared_intermediate_size": 1536,
        "run_name": "100m-o200k-tr-local",
        "save_dir": "checkpoints/100m-o200k-tr-local",
        "description": "100M o200k TR",
    },
    "300m": {
        "hidden_size": 896,
        "num_hidden_layers": 10,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "intermediate_size": 256,
        "shared_intermediate_size": 3584,
        "run_name": "300m-o200k-tr-local",
        "save_dir": "checkpoints/300m-o200k-tr-local",
        "description": "300M o200k TR",
    },
    "1b": {
        "hidden_size": 1536,
        "num_hidden_layers": 20,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "intermediate_size": 512,
        "shared_intermediate_size": 6144,
        "run_name": "1b-o200k-tr-local",
        "save_dir": "checkpoints/1b-o200k-tr-local",
        "description": "1B o200k TR (20L hidden=1536, GQA 24/4, 4 routed + shared)",
    },
    "8b": {
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 3072,
        "shared_intermediate_size": 12288,
        "run_name": "8b-o200k-tr-local",
        "save_dir": "checkpoints/8b-o200k-tr-local",
        "description": "8B o200k TR",
    },
}


def make_config(args) -> ModelConfig:
    """Build a ModelConfig from parsed o200k runner args."""

    return ModelConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        intermediate_size=args.intermediate_size,
        vocab_size=args.vocab_size,
        max_position_embeddings=2048,
        attention_type=getattr(args, "attention_type", "gqa"),
        causal_conv_kernel_size=getattr(args, "causal_conv_kernel_size", 4),
        causal_conv_dilation_cycle=getattr(args, "causal_conv_dilation_cycle", 8),
        causal_state_rank=getattr(args, "causal_state_rank", 16),
        causal_context_gate_init=getattr(args, "causal_context_gate_init", 1.0),
        causal_contextual_mix_init=getattr(args, "causal_contextual_mix_init", 0.0),
        causal_context_fusion_size=getattr(args, "causal_context_fusion_size", 0),
        causal_stable_delta=bool(getattr(args, "causal_stable_delta", False)),
        causal_delta_chunk_size=getattr(args, "causal_delta_chunk_size", 512),
        causal_delta_timescales=getattr(args, "causal_delta_timescales", 1),
        causal_delta_collision_normalized=bool(
            getattr(args, "causal_delta_collision_normalized", False)
        ),
        causal_delta_lexical_values=bool(
            getattr(args, "causal_delta_lexical_values", False)
        ),
        causal_delta_lexical_forge=bool(
            getattr(args, "causal_delta_lexical_forge", False)
        ),
        causal_delta_occurrence_address=bool(
            getattr(args, "causal_delta_occurrence_address", False)
        ),

        lexical_attention_layer_indices=tuple(
            getattr(args, "lexical_attention_layer_indices", ())
        ),
        mlp_type=getattr(args, "mlp_type", None) or "token_routed",
        num_experts=4,
        shared_expert=bool(getattr(args, "shared_expert", True)),
        shared_intermediate_size=args.shared_intermediate_size,
        shared_expert_chunk_tokens=getattr(args, "shared_expert_chunk_tokens", 0),
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=args.use_mu_guidance,
        use_shared_routed_gates=args.learn_shared_routed_gates,
        shared_gate_init=args.shared_gate_init,
        routed_gate_init=args.routed_gate_init,
        top_k=args.top_k,
        top_k_primary_weight=args.top_k_primary_weight,
        use_custom_kernels=getattr(args, "use_custom_kernels", "auto"),
        use_cggr=getattr(args, "cggr", getattr(args, "use_cggr", "auto")),
        static_expert_capacity=bool(getattr(args, "static_expert_capacity", False)),
        collect_moe_telemetry=bool(getattr(args, "moe_telemetry", False)),
        routing_strategy=getattr(args, "routing_strategy", "zipf"),
        lsh_threshold_mode=getattr(args, "lsh_threshold_mode", "zero"),
        lexical_object_rank=getattr(args, "lexical_object_rank", 16),
        lexical_object_gate_init=getattr(args, "lexical_object_gate_init", 0.1),
        disable_lexical_wrv_residual=bool(
            getattr(args, "disable_lexical_wrv_residual", False)
        ),
        disable_lexical_wrv_norms=bool(
            getattr(args, "disable_lexical_wrv_norms", False)
        ),
        lexical_wrv_hybrid=bool(getattr(args, "lexical_wrv_hybrid", False)),
        lexical_wrv_gate_init=float(getattr(args, "lexical_wrv_gate_init", 0.0)),
        lexical_gqa_rank=int(getattr(args, "lexical_gqa_rank", 16)),
        lexical_gqa_gate_init=float(getattr(args, "lexical_gqa_gate_init", 0.0)),
        lexical_gqa_use_token_code=bool(
            getattr(args, "lexical_gqa_use_token_code", True)
        ),
        lexical_key_gate_init=float(getattr(args, "lexical_key_gate_init", 0.05)),
        lexical_zipf_path=getattr(args, "lexical_zipf_path", None),
        lexical_zipf_mode=str(getattr(args, "lexical_zipf_mode", "uniform")),
        lexical_zipf_alpha=float(getattr(args, "lexical_zipf_alpha", 0.25)),
        lexical_zipf_floor=float(getattr(args, "lexical_zipf_floor", 0.1)),
        lexical_zipf_permutation_seed=int(
            getattr(args, "lexical_zipf_permutation_seed", 1729)
        ),
        tie_lexical_object_embeddings=bool(
            getattr(args, "tie_lexical_object_embeddings", False)
        ),
        micro_num_experts=getattr(args, "micro_num_experts", 4),
        micro_expert_width=getattr(args, "micro_expert_width", 16),
        micro_expert_gate_init=getattr(args, "micro_expert_gate_init", 0.1),
        clamp_mu_contextual=args.mu_clamp,
        use_mu_norm=args.mu_norm,
        mu_alpha_init=args.mu_alpha_init,
        mu_init_value=args.mu_init_value,
        mu_context_min=args.mu_context_min,
        mu_context_max=args.mu_context_max,
    )
