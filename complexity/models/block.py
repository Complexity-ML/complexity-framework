"""
Transformer Block - the basic building unit.

A block consists of:
1. Attention (with pre-norm)
2. MLP/FFN (with pre-norm)
3. Residual connections
"""

import torch
import torch.nn as nn
from typing import Any, Optional, Tuple

from ..config import ModelConfig
from ..core.attention import AttentionConfig
from ..core.mlp import MLPConfig
from ..core.registry import ATTENTION_REGISTRY, MLP_REGISTRY, NORMALIZATION_REGISTRY


class TransformerBlock(nn.Module):
    """
    Single Transformer block with configurable components.

    Architecture (Pre-Norm):
        x = x + attention(norm1(x))
        x = x + mlp(norm2(x))
    """

    def __init__(self, config: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        # Pre-attention normalization
        self.input_layernorm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # Attention
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rope_theta=config.rope_theta,
            attention_dropout=config.attention_dropout,
            use_qk_norm=config.use_qk_norm,
            sliding_window=config.sliding_window,
            use_sdpa=config.use_sdpa,
            is_causal=getattr(config, "is_causal", True),
            rope_type=config.rope_type,
            use_mup_attn_scale=getattr(config, "use_mup_attn_scale", False),
            causal_conv_kernel_size=getattr(config, "causal_conv_kernel_size", 4),
            causal_conv_dilation=2 ** (
                layer_idx % int(getattr(config, "causal_conv_dilation_cycle", 8))
            ),
            causal_state_rank=getattr(config, "causal_state_rank", 16),
            causal_context_gate_init=getattr(
                config, "causal_context_gate_init", 1.0
            ),
            causal_contextual_mix_init=getattr(
                config, "causal_contextual_mix_init", 0.0
            ),
            causal_context_fusion_size=getattr(
                config, "causal_context_fusion_size", 0
            ),
            causal_stable_delta=getattr(config, "causal_stable_delta", False),
            causal_delta_chunk_size=getattr(
                config, "causal_delta_chunk_size", 512
            ),
            causal_delta_timescales=getattr(
                config, "causal_delta_timescales", 1
            ),
            causal_delta_collision_normalized=getattr(
                config, "causal_delta_collision_normalized", False
            ),
            causal_delta_lexical_values=getattr(
                config, "causal_delta_lexical_values", False
            ),
            causal_delta_lexical_forge=getattr(
                config, "causal_delta_lexical_forge", False
            ),
            causal_delta_occurrence_address=getattr(
                config, "causal_delta_occurrence_address", False
            ),

            lexical_object_rank=config.lexical_object_rank,
            lexical_gqa_rank=getattr(config, "lexical_gqa_rank", 16),
            lexical_gqa_gate_init=getattr(config, "lexical_gqa_gate_init", 0.0),
            lexical_gqa_use_token_code=getattr(
                config, "lexical_gqa_use_token_code", True
            ),
            lexical_key_gate_init=getattr(config, "lexical_key_gate_init", 0.05),
            vocab_size=config.vocab_size,
            layer_idx=layer_idx,
            tr_mha_num_experts=getattr(config, "tr_mha_num_experts", 4),
            tr_mha_adapter_rank=getattr(config, "tr_mha_adapter_rank", 8),
            tr_mha_top_k=getattr(config, "tr_mha_top_k", 2),
            tr_mha_adapter_gate_init=getattr(
                config, "tr_mha_adapter_gate_init", 0.1
            ),
            tr_mha_id_primary_logit=getattr(
                config, "tr_mha_id_primary_logit", 2.0
            ),
            tr_mha_id_secondary_logit=getattr(
                config, "tr_mha_id_secondary_logit", 1.0
            ),
            tr_mha_id_other_logit=getattr(
                config, "tr_mha_id_other_logit", -2.0
            ),
            tr_mha_verifier_gate_init=getattr(
                config, "tr_mha_verifier_gate_init", 0.1
            ),
            tr_mha_verifier_temperature=getattr(
                config, "tr_mha_verifier_temperature", 1.0
            ),
            tr_mha_targets=getattr(config, "tr_mha_targets", "qv"),
        )
        self.self_attn = ATTENTION_REGISTRY.build(config.attention_type, attn_config)

        # Post-attention normalization
        self.post_attention_layernorm = NORMALIZATION_REGISTRY.build(
            config.norm_type,
            config.hidden_size,
            eps=config.norm_eps,
        )

        # MLP
        def layer_scale(
            base: float,
            first: Optional[float],
            last: Optional[float],
        ) -> float:
            if first is None and last is None:
                return float(base)
            start = float(base if first is None else first)
            end = float(start if last is None else last)
            denominator = max(1, int(config.num_hidden_layers) - 1)
            ratio = min(1.0, max(0.0, layer_idx / denominator))
            return start + (end - start) * ratio

        shared_output_scale = layer_scale(
            getattr(config, 'shared_output_scale', 1.0),
            getattr(config, 'shared_output_scale_first_layer', None),
            getattr(config, 'shared_output_scale_last_layer', None),
        )
        routed_output_scale = layer_scale(
            getattr(config, 'routed_output_scale', 1.0),
            getattr(config, 'routed_output_scale_first_layer', None),
            getattr(config, 'routed_output_scale_last_layer', None),
        )
        mlp_config = MLPConfig(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            num_experts=config.num_experts,
            expert_initialization=getattr(
                config, "expert_initialization", "gpt_normal"
            ),
            initializer_range=getattr(config, "initializer_range", 0.02),
            vocab_size=config.vocab_size,
            routing_strategy=getattr(config, 'routing_strategy', 'modulo_cyclic'),
            route_hash_count=getattr(config, 'route_hash_count', 2),
            token_frequencies=config.token_frequencies,
            lsh_routing=getattr(config, 'lsh_routing', False),
            lsh_bits=getattr(config, 'lsh_bits', 0),
            lsh_from_layer=getattr(config, 'lsh_from_layer', 0),
            lsh_threshold_mode=getattr(config, 'lsh_threshold_mode', 'zero'),
            shared_expert=getattr(config, 'shared_expert', False),
            shared_intermediate_size=getattr(config, 'shared_intermediate_size', None),
            shared_expert_chunk_tokens=getattr(config, 'shared_expert_chunk_tokens', 0),
            use_shared_routed_gates=getattr(config, 'use_shared_routed_gates', False),
            shared_gate_init=getattr(config, 'shared_gate_init', 1.0),
            routed_gate_init=getattr(config, 'routed_gate_init', 1.0),
            shared_output_scale=shared_output_scale,
            routed_output_scale=routed_output_scale,
            top_k=getattr(config, 'top_k', 1),
            top_k_primary_weight=getattr(config, 'top_k_primary_weight', None),
            learn_hash_pair_gates=getattr(config, 'learn_hash_pair_gates', False),
            hash_pair_gate_init=getattr(config, 'hash_pair_gate_init', 0.5),
            learn_hash_channel_modulation=getattr(
                config, 'learn_hash_channel_modulation', False
            ),
            hash_channel_scale_init=getattr(
                config, 'hash_channel_scale_init', 0.0
            ),
            layer_idx=layer_idx,
            static_expert_capacity=getattr(config, 'static_expert_capacity', False),
            collect_moe_telemetry=getattr(config, 'collect_moe_telemetry', False),
            use_custom_kernels=getattr(config, 'use_custom_kernels', 'auto'),
            use_cggr=getattr(config, 'use_cggr', False),
            lexical_object_rank=getattr(config, 'lexical_object_rank', 16),
            lexical_object_gate_init=getattr(config, 'lexical_object_gate_init', 0.1),
            micro_num_experts=getattr(config, 'micro_num_experts', 4),
            micro_expert_width=getattr(config, 'micro_expert_width', 16),
            micro_expert_gate_init=getattr(config, 'micro_expert_gate_init', 0.1),
            active_num_experts=getattr(config, 'active_num_experts', None),
            active_expert_width=getattr(config, 'active_expert_width', None),
        )
        self.mlp = MLP_REGISTRY.build(config.mlp_type, mlp_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        velocity_state: Optional[torch.Tensor] = None,
        sort_idx: Optional[torch.Tensor] = None,
        lexical_token_scale_values: Optional[torch.Tensor] = None,
        lexical_zipf_values: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Any], Optional[torch.Tensor]]:
        """
        Forward pass through the transformer block.

        Args:
            hidden_states: [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional KV cache
            use_cache: Whether to return updated KV cache
            token_ids: Optional token IDs for MoE routing
            velocity_state: Unused (kept for backward compat)
            sort_idx: Unused (sort_idx computed internally by token_routed)

        Returns:
            hidden_states: [batch, seq_len, hidden_size]
            past_key_value: Optional updated KV cache
            velocity_state: None (kept for backward compat)
        """
        residual = hidden_states

        # Self Attention
        hidden_states = self.input_layernorm(hidden_states)

        attn_kwargs = dict(
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        if self.config.attention_type in {
            "causal_fast_weight_conv",
            "lexical_gqa",
            "lexical_bias_gqa",
            "lexical_key_gqa",
            "projected_lexical_key_gqa",
            "tr_mha",
            "token_routed_mha",
            "tr_mha_v2",
            "token_routed_mha_v2",
        }:
            attn_kwargs["token_ids"] = token_ids
        if self.config.attention_type == "causal_fast_weight_conv":
            attn_kwargs["lexical_token_scale_values"] = lexical_token_scale_values
        elif self.config.attention_type in {
            "lexical_gqa",
            "lexical_bias_gqa",
            "lexical_key_gqa",
            "projected_lexical_key_gqa",
        }:
            attn_kwargs["lexical_scale"] = lexical_token_scale_values
            if self.config.attention_type == "projected_lexical_key_gqa":
                attn_kwargs["lexical_weight"] = lexical_zipf_values
        hidden_states, new_kv = self.self_attn(hidden_states, **attn_kwargs)
        hidden_states = residual + hidden_states

        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            token_ids=token_ids,
            lexical_token_scale_values=lexical_token_scale_values,
        )
        hidden_states = residual + hidden_states

        return hidden_states, new_kv, None
