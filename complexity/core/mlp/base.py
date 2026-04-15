"""
Base MLP class for framework-complexity.

All MLP implementations must inherit from this class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MLPConfig:
    """Configuration for MLP modules."""
    hidden_size: int
    intermediate_size: int
    hidden_act: str = "silu"  # silu, gelu, relu
    bias: bool = False

    # MoE specific
    num_experts: int = 1  # 1 = standard MLP, >1 = MoE
    vocab_size: int = 100000  # For token-routed MoE
    hash_routing: str = ""  # "" = modulo (token_id % E), "learned" = learned projection router
    token_frequencies: Optional[torch.Tensor] = None  # [vocab_size] token counts for frequency-balanced routing
    shared_expert: bool = True  # Shared lexical expert: dense MLP + routed experts
    shared_intermediate_size: Optional[int] = None  # Shared expert size (default: intermediate_size)
    routed_gate: bool = False  # Learnable α on routed path: out = shared + α·routed
    routed_gate_init: float = 0.0  # Initial value of α (0 = pure dense start, >0 = experts contribute early)
    gpt2_residual_init: bool = False  # Scale down_proj init by 1/sqrt(2·num_hidden_layers) — GPT-2 style
    num_hidden_layers: int = 1  # Required for gpt2_residual_init scaling


class MLPBase(nn.Module, ABC):
    """
    Abstract base class for MLP/FFN layers.

    All MLP implementations must inherit from this class and implement forward().
    """

    def __init__(self, config: MLPConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

    @abstractmethod
    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass for MLP.

        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            token_ids: Optional token IDs for routing (MoE only)
            **kwargs: Extra kwargs (e.g. sort_idx, mu) absorbed by subclasses

        Returns:
            output: [batch, seq_len, hidden_size]
        """
        pass

    @staticmethod
    def get_activation(name: str):
        """Get activation function by name."""
        activations = {
            "silu": F.silu,
            "swish": F.silu,
            "gelu": F.gelu,
            "relu": F.relu,
            "gelu_new": lambda x: F.gelu(x, approximate="tanh"),
        }
        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")
        return activations[name]
