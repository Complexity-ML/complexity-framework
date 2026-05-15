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
    routing_strategy: str = "zipf"  # "zipf" or "zipf_token_class"
    token_frequencies: Optional[torch.Tensor] = None  # [vocab_size] token counts for frequency-balanced routing
    token_classes: Optional[torch.Tensor] = None  # [vocab_size] coarse token classes for class-balanced routing
    shared_expert: bool = True  # Shared lexical expert: dense MLP + routed experts
    shared_intermediate_size: Optional[int] = None  # Shared expert size (default: intermediate_size)
    use_shared_routed_gates: bool = False  # Learn scalar gates for shared vs routed expert outputs.
    shared_gate_init: float = 1.0  # Initial shared expert output multiplier.
    routed_gate_init: float = 1.0  # Initial routed expert output multiplier.
    top_k: int = 1  # Token-Routed top-K deterministic: each token activates K experts via cyclic shift of the Zipf primary. Primary keeps weight 0.95, secondary experts share 0.05 (tuned for specialization preservation).
    top_k_primary_weight: Optional[float] = None  # K>1 primary expert blend weight; None keeps the default 0.95.
    layer_idx: int = 0  # Layer index propagated from the block; used for the built-in per-layer routing permutation.
    static_expert_capacity: bool = False  # Fixed capacity for torch.export / pipeline tracing.

    def __post_init__(self):
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_k > self.num_experts:
            raise ValueError("top_k cannot exceed num_experts")
        if self.top_k_primary_weight is not None and not 0.0 <= self.top_k_primary_weight <= 1.0:
            raise ValueError("top_k_primary_weight must be in [0, 1]")
        if self.routing_strategy not in {"zipf", "zipf_token_class"}:
            raise ValueError("routing_strategy must be 'zipf' or 'zipf_token_class'")
        if self.shared_intermediate_size is not None and self.shared_intermediate_size <= 0:
            raise ValueError("shared_intermediate_size must be positive when set")
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
        if self.token_classes is not None:
            if not isinstance(self.token_classes, torch.Tensor):
                raise ValueError("token_classes must be a torch.Tensor")
            if self.token_classes.ndim != 1:
                raise ValueError("token_classes must be a 1D tensor")
            if self.token_classes.numel() != self.vocab_size:
                raise ValueError(
                    f"token_classes length ({self.token_classes.numel()}) "
                    f"must match vocab_size ({self.vocab_size})"
                )


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
