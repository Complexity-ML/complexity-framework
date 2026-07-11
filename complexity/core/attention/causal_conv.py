"""Attention-free causal convolution sequence mixer."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import register_attention
from .base import AttentionBase, AttentionConfig


@register_attention("causal_conv")
@register_attention("lexical_object_conv")
class CausalConvMixer(AttentionBase):
    """Mix sequence context with a dilated causal depthwise convolution."""

    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.kernel_size = int(config.causal_conv_kernel_size)
        self.dilation = int(config.causal_conv_dilation)
        self.receptive_span = (self.kernel_size - 1) * self.dilation
        self.depthwise = nn.Conv1d(
            self.hidden_size,
            self.hidden_size,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            groups=self.hidden_size,
            bias=False,
        )
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        del attention_mask, kwargs
        if past_key_value is None:
            context = hidden_states
            channels_first = F.pad(
                context.transpose(1, 2), (self.receptive_span, 0)
            )
        else:
            if past_key_value.shape[1] != self.receptive_span:
                raise ValueError(
                    "causal convolution state has invalid sequence length: "
                    f"expected {self.receptive_span}, got {past_key_value.shape[1]}"
                )
            context = torch.cat((past_key_value, hidden_states), dim=1)
            channels_first = context.transpose(1, 2)

        mixed = self.depthwise(channels_first)
        mixed = mixed.transpose(1, 2)
        gated = F.silu(self.gate_proj(mixed)) * self.up_proj(mixed)
        new_state = None
        if use_cache:
            history = context[:, -self.receptive_span :]
            missing = self.receptive_span - history.shape[1]
            if past_key_value is not None and not torch.is_grad_enabled():
                past_key_value.copy_(history)
                new_state = past_key_value
            else:
                new_state = (
                    F.pad(history, (0, 0, missing, 0)) if missing > 0 else history
                )
        return self.o_proj(gated), new_state