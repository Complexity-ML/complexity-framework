"""
Mu Projection - Lightweight mu guidance without PID dynamics.

Produces contextual mu from hidden states via a simple learned projection.
No velocity tracking, no controller MLP, no PID overhead.

This gives the same mu-guided attention/routing benefits as INLDynamics
but without the second-order dynamical system that slows loss descent.

Usage:
    mu_proj = MuProjection(hidden_size=2048)
    mu_contextual = mu_proj(hidden_states)
    # Pass mu_contextual as mu_prev to next layer's attention/MLP
"""

import torch
import torch.nn as nn
from typing import Optional


class MuProjection(nn.Module):
    """
    Lightweight mu generator: mu_contextual = clamp(mu + mu_proj(h)).

    Same mu signal as INLDynamics but without velocity/PID.
    ~10x fewer parameters, no sequential dependency on velocity state.

    Args:
        hidden_size: Model hidden dimension
        mu_init: Initial equilibrium value (default: 1.0, center of [0, 2])
        mu_min: Minimum mu clamp (default: -2.0)
        mu_max: Maximum mu clamp (default: 2.0)
    """

    def __init__(
        self,
        hidden_size: int,
        mu_init: float = 1.0,
        mu_min: float = -2.0,
        mu_max: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mu_min = mu_min
        self.mu_max = mu_max

        # Learnable base equilibrium
        self.mu = nn.Parameter(torch.full((hidden_size,), mu_init))

        # Context-dependent projection
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)  # Start neutral

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Compute contextual mu from hidden states.

        Args:
            h: Hidden states [batch, seq_len, hidden_size]

        Returns:
            mu_contextual: [batch, seq_len, hidden_size]
        """
        return self.mu + self.mu_proj(h)
