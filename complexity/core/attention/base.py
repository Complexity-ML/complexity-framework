"""
Base attention class for framework-complexity.

All attention implementations must inherit from this class.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for attention modules."""
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int  # For GQA/MQA
    head_dim: Optional[int] = None  # Auto-computed if None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    attention_dropout: float = 0.0
    use_qk_norm: bool = True
    sliding_window: Optional[int] = None
    use_sdpa: bool = True
    rope_type: str = "standard"  # standard, yarn, dynamic
    use_mup_attn_scale: bool = False  # μP: 1/d_head attention logit scale (vs 1/√d_head)
    use_mu_guidance: bool = False  # Add mu-to-K/Q/V projections for guided attention.
    scale: Optional[float] = None
    causal_conv_kernel_size: int = 4
    causal_conv_dilation: int = 1
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

    lexical_object_rank: int = 16
    disable_lexical_wrv_residual: bool = False
    disable_lexical_wrv_norms: bool = False
    lexical_wrv_hybrid: bool = False
    lexical_wrv_gate_init: float = 0.0
    lexical_gqa_rank: int = 16
    lexical_gqa_gate_init: float = 0.0
    lexical_gqa_use_token_code: bool = True
    vocab_size: Optional[int] = None
    layer_idx: int = 0

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.num_key_value_heads <= 0:
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
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError("hidden_size must equal num_attention_heads * head_dim")
        if self.scale is None:
            self.scale = self.head_dim ** -0.5
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.sliding_window is not None and self.sliding_window <= 0:
            raise ValueError("sliding_window must be positive when set")
        if self.causal_conv_kernel_size <= 0:
            raise ValueError("causal_conv_kernel_size must be positive")
        if self.causal_conv_dilation <= 0:
            raise ValueError("causal_conv_dilation must be positive")
        if self.causal_state_rank <= 0:
            raise ValueError("causal_state_rank must be positive")

        if self.causal_context_gate_init < 0.0:
            raise ValueError("causal_context_gate_init must be non-negative")


class AttentionBase(nn.Module, ABC):
    """
    Abstract base class for attention mechanisms.

    All attention implementations in the framework must inherit from this class
    and implement the forward method with the specified signature.
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for attention.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Optional attention mask
            past_key_value: Optional cached KV for generation
            use_cache: Whether to return updated KV cache

        Returns:
            output: Tensor of shape [batch, seq_len, hidden_size]
            past_key_value: Optional updated KV cache tuple (k, v)
        """
        pass

    def _init_projections(self, bias: bool = False):
        """Initialize Q, K, V, O projections."""
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=bias)
