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

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        # Validation
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_heads ({self.num_attention_heads})"
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_heads ({self.num_attention_heads}) must be divisible by num_kv_heads ({self.num_key_value_heads})"


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
