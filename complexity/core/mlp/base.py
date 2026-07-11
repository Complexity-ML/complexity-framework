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
    routing_strategy: str = "zipf"  # zipf, modulo, round_robin, random, lsh_hidden
    token_frequencies: Optional[torch.Tensor] = None  # [vocab_size] token counts for frequency-balanced routing
    # Semantic LSH routing: route on a fixed random-hyperplane hash of the
    # hidden state (post-attention) instead of the token id. Deterministic, no
    # learned gate; the expert choice now depends on the contextual/semantic
    # representation. lsh_bits=0 falls back to ceil(log2(num_experts)).
    lsh_routing: bool = False
    lsh_bits: int = 0
    lsh_from_layer: int = 0  # Use LSH routing only for layers >= this index; earlier layers stay lexical (h not yet semantic).
    lsh_threshold_mode: str = "zero"  # "zero" or "batch_median"
    shared_expert: bool = True  # Shared lexical expert: dense MLP + routed experts
    shared_intermediate_size: Optional[int] = None  # Shared expert size (default: intermediate_size)
    shared_expert_chunk_tokens: int = 0  # 0 = dense shared expert in one pass; >0 chunks token dimension to reduce activation peak.
    use_shared_routed_gates: bool = False  # Learn scalar gates for shared vs routed expert outputs.
    shared_gate_init: float = 1.0  # Initial shared expert output multiplier.
    routed_gate_init: float = 1.0  # Initial routed expert output multiplier.
    top_k: int = 1  # Token-Routed top-K deterministic: each token activates K Zipf-balanced expert routes.
    top_k_primary_weight: Optional[float] = None  # K>1 primary expert blend weight; None keeps the default 0.95.
    layer_idx: int = 0  # Layer index propagated from the block; used for the built-in per-layer routing permutation.
    static_expert_capacity: bool = False  # Fixed capacity for torch.export / pipeline tracing.
    collect_moe_telemetry: bool = False  # Per-layer expert/RMS diagnostics. Off by default for throughput.
    use_custom_kernels: object = "auto"  # "auto", True, or False. Controls optional Triton paths.
    use_cggr: object = "auto"  # "auto", True, or False. Use CGGR grouped-GEMM when the backend policy allows custom Triton.

    # Lexical feature-modulated residual: dense shared SwiGLU plus a low-rank
    # token-conditioned residual with no expert dispatch.
    lexical_object_rank: int = 16
    lexical_object_gate_init: float = 0.1
    micro_num_experts: int = 4
    micro_expert_width: int = 16
    micro_expert_gate_init: float = 0.1

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
        if self.routing_strategy not in {"zipf", "modulo", "round_robin", "random", "lsh_hidden"}:
            raise ValueError("routing_strategy must be one of zipf, modulo, round_robin, random, lsh_hidden")
        if self.lsh_threshold_mode not in {"batch_median", "zero"}:
            raise ValueError("lsh_threshold_mode must be 'batch_median' or 'zero'")
        if isinstance(self.use_cggr, str) and self.use_cggr.strip().lower() not in {"auto", "true", "false"}:
            raise ValueError("use_cggr must be one of 'auto', 'true', 'false', True, or False")
        if self.shared_intermediate_size is not None and self.shared_intermediate_size <= 0:
            raise ValueError("shared_intermediate_size must be positive when set")
        if self.shared_expert_chunk_tokens < 0:
            raise ValueError("shared_expert_chunk_tokens must be non-negative")
        if self.lexical_object_rank <= 0:
            raise ValueError("lexical_object_rank must be positive")
        if self.micro_num_experts <= 0:
            raise ValueError("micro_num_experts must be positive")
        if self.micro_expert_width <= 0:
            raise ValueError("micro_expert_width must be positive")
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
