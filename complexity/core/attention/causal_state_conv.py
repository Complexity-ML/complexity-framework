"""Causal convolution plus a persistent diagonal recurrent state."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_attention
from .base import AttentionConfig
from .causal_conv import CausalConvMixer


@register_attention("causal_state_conv")
@register_attention("lexical_object_state")
class CausalStateConvMixer(CausalConvMixer):
    """Combine finite local context with a fixed-size persistent state."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        rank = int(config.causal_state_rank)
        self.state_decay_down = nn.Linear(self.hidden_size, rank, bias=False)
        self.state_decay_up = nn.Linear(rank, self.hidden_size, bias=False)
        self.state_decay_bias = nn.Parameter(torch.full((self.hidden_size,), 2.944439))
        self.state_output_gate = nn.Parameter(torch.tensor(0.1))

    def _local_mix(
        self,
        hidden_states: torch.Tensor,
        conv_state: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def _parallel_state(
        self, local: torch.Tensor, initial_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the recurrence in stable, statically-sized parallel chunks."""
        device_type = local.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            local_f = local.float()
            write = torch.sigmoid(
                self.state_decay_up(F.silu(self.state_decay_down(local_f)))
            )
            update = write * local_f
            decay = torch.sigmoid(self.state_decay_bias.float()).clamp(0.95, 0.999)
            carry = initial_state.float()
            state_chunks = []
            for update_chunk in update.split(128, dim=1):
                steps = torch.arange(
                    1,
                    update_chunk.shape[1] + 1,
                    device=local.device,
                    dtype=torch.float32,
                )
                powers = decay[None, None, :] ** steps[None, :, None]
                weighted = (1.0 - decay)[None, None, :] * update_chunk / powers
                chunk_states = powers * (
                    carry[:, None, :] + torch.cumsum(weighted, dim=1)
                )
                carry = chunk_states[:, -1]
                state_chunks.append(chunk_states)
            states = torch.cat(state_chunks, dim=1)
        return states[:, -1], states.to(local.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        del attention_mask, kwargs
        conv_state = past_key_value[0] if past_key_value is not None else None
        initial_state = (
            past_key_value[1]
            if past_key_value is not None
            else hidden_states.new_zeros(hidden_states.shape[0], self.hidden_size)
        )
        local, next_conv_state = self._local_mix(hidden_states, conv_state)
        if hidden_states.shape[1] == 1:
            with torch.autocast(device_type=hidden_states.device.type, enabled=False):
                local_f = local[:, 0].float()
                write = torch.sigmoid(
                    self.state_decay_up(F.silu(self.state_decay_down(local_f)))
                )
                decay = torch.sigmoid(self.state_decay_bias.float()).clamp(0.95, 0.999)
                next_state = (
                    decay * initial_state.float() + (1.0 - decay) * write * local_f
                )
            states = next_state[:, None, :].to(local.dtype)
        else:
            next_state, states = self._parallel_state(local, initial_state)

        combined = local + self.state_output_gate * states
        gated = F.silu(self.gate_proj(combined)) * self.up_proj(combined)
        new_state = None
        if use_cache:
            if past_key_value is not None and not torch.is_grad_enabled():
                conv_state.copy_(next_conv_state)
                initial_state.copy_(next_state)
                new_state = (conv_state, initial_state)
            else:
                new_state = (next_conv_state, next_state)
        return self.o_proj(gated), new_state