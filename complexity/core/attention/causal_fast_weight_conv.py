"""Causal convolution consuming a standalone shared context mechanism."""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from ...experiments.shared_associative_attention import (
    SharedAssociativeAttention,
    SharedContextFusion,
    StableDeltaAssociativeAttention,
    MultiTimescaleDeltaAttention,
    CollisionNormalizedDeltaAttention,
    LexicalValueDeltaAttention,
    LexicalForgeDeltaAttention,
    CollisionNormalizedLexicalForgeDeltaAttention,
    PositionAwareCollisionForgeDeltaAttention,

)
from ..registry import register_attention
from .base import AttentionConfig
from .causal_conv import CausalConvMixer


@register_attention("causal_fast_weight_conv")
class CausalFastWeightConvMixer(CausalConvMixer):
    """Local causal mixer with an externally defined shared context branch."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.context_enabled = True
        if (
            config.causal_stable_delta
            and config.causal_delta_lexical_forge
            and config.causal_delta_collision_normalized
            and config.causal_delta_occurrence_address
        ):
            self.shared_context = PositionAwareCollisionForgeDeltaAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
                lexical_object_rank=config.lexical_object_rank,
                context_signature_rank=8,
                delta_chunk_size=config.causal_delta_chunk_size,
            )
        elif (
            config.causal_stable_delta
            and config.causal_delta_lexical_forge
            and config.causal_delta_collision_normalized
        ):
            self.shared_context = CollisionNormalizedLexicalForgeDeltaAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
                lexical_object_rank=config.lexical_object_rank,
                delta_chunk_size=config.causal_delta_chunk_size,
            )
        elif config.causal_stable_delta and config.causal_delta_lexical_forge:
            self.shared_context = LexicalForgeDeltaAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
                lexical_object_rank=config.lexical_object_rank,
                delta_chunk_size=config.causal_delta_chunk_size,
            )
        elif config.causal_stable_delta and config.causal_delta_lexical_values:
            self.shared_context = LexicalValueDeltaAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
                delta_chunk_size=config.causal_delta_chunk_size,
            )
        elif config.causal_stable_delta and config.causal_delta_collision_normalized:
            self.shared_context = CollisionNormalizedDeltaAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
                delta_chunk_size=config.causal_delta_chunk_size,
            )
        elif config.causal_stable_delta and config.causal_delta_timescales > 1:
            self.shared_context = MultiTimescaleDeltaAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                num_timescales=config.causal_delta_timescales,
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
                delta_chunk_size=config.causal_delta_chunk_size,
            )
        elif config.causal_stable_delta:
            self.shared_context = StableDeltaAssociativeAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
                delta_chunk_size=config.causal_delta_chunk_size,
            )
        elif config.causal_context_fusion_size > 0:
            self.shared_context = SharedContextFusion(
                self.hidden_size,
                int(config.causal_state_rank),
                fusion_size=config.causal_context_fusion_size,
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                contextual_mix_init=config.causal_contextual_mix_init,
            )
        else:
            self.shared_context = SharedAssociativeAttention(
                self.hidden_size,
                int(config.causal_state_rank),
                vocab_size=config.vocab_size if config.layer_idx == 0 else None,
                output_gate_init=config.causal_context_gate_init,
                contextual_mix_init=config.causal_contextual_mix_init,
            )

    def _local_mix(
        self, hidden_states: torch.Tensor, conv_state: Optional[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conv_state is None:
            context = hidden_states
            channels_first = F.pad(
                context.transpose(1, 2), (self.receptive_span, 0)
            )
        else:
            if conv_state.shape[1] != self.receptive_span:
                raise ValueError(
                    "causal convolution state has invalid sequence length: "
                    f"expected {self.receptive_span}, got {conv_state.shape[1]}"
                )
            context = torch.cat((conv_state, hidden_states), dim=1)
            channels_first = context.transpose(1, 2)
        local = self.depthwise(channels_first).transpose(1, 2)
        history = context[:, -self.receptive_span :]
        missing = self.receptive_span - history.shape[1]
        if missing > 0:
            history = F.pad(history, (0, 0, missing, 0))
        return local, history

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Any = None,
        use_cache: bool = False,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Any:
        del attention_mask, kwargs
        if token_ids is None:
            raise ValueError("token_ids are required for shared associative context")
        conv_state = past_key_value[0] if past_key_value is not None else None
        local, next_conv_state = self._local_mix(hidden_states, conv_state)
        gated = F.silu(self.gate_proj(local)) * self.up_proj(local)
        local_output = self.o_proj(gated)
        if not self.context_enabled:
            new_state = None
            if use_cache:
                if past_key_value is not None and not torch.is_grad_enabled():
                    assert conv_state is not None
                    conv_state.copy_(next_conv_state)
                    new_state = (conv_state,)
                else:
                    new_state = (next_conv_state,)
            return local_output, new_state

        context_state = (
            tuple(past_key_value[1:])
            if past_key_value is not None
            else self.shared_context.initial_state(
                hidden_states.shape[0],
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        )
        context_residual, next_context_state = self.shared_context(
            local, token_ids, context_state
        )
        output = local_output + context_residual

        new_state = None
        if use_cache:
            if past_key_value is not None and not torch.is_grad_enabled():
                assert conv_state is not None
                conv_state.copy_(next_conv_state)
                if len(context_state) != len(next_context_state):
                    raise ValueError(
                        "context state arity changed during incremental decoding: "
                        f"{len(context_state)} -> {len(next_context_state)}"
                    )
                for current, updated in zip(context_state, next_context_state):
                    current.copy_(updated)
                new_state = (conv_state, *context_state)
            else:
                new_state = (next_conv_state, *next_context_state)
        return output, new_state
