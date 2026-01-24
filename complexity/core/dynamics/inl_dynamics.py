"""
INL Dynamics - Second-order dynamical system for LLM hidden states.

Like a robot controller:
- Smooth trajectories (no jerky movements)
- Stable convergence (PID-like control)
- Learnable dynamics per dimension
- Real-time capable

v0.3.0: Contextual mu via mu_proj (INL 2025)
v0.3.0: Returns mu_contextual for next layer guidance
v0.3.1: Safety clamp integration (Representation Engineering)

CRITICAL: beta in [0, 2], NOT [0, inf)!
Discovered after 400k step explosion during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class DynamicsConfig:
    """Configuration for INL Dynamics."""
    hidden_size: int = 768
    init_alpha: float = 0.9       # Inertia (momentum)
    init_beta: float = 0.1        # Correction strength
    init_gate: float = 0.5        # Amplitude control
    dt: float = 0.1               # Integration timestep
    controller_hidden: int = 64   # Controller MLP hidden size

    # CRITICAL STABILITY BOUNDS
    beta_max: float = 2.0         # Max correction (prevents explosion!)
    velocity_max: float = 10.0    # Max velocity (prevents runaway)

    # mu bounds (equilibrium point)
    mu_min: float = 0.0
    mu_max: float = 2.0


class INLDynamics(nn.Module):
    """
    Full INL Dynamics - Robotics-grade control with velocity tracking.

    Equations (like a physical system):
        error = h - mu                      # deviation from equilibrium
        v_next = alpha * v - beta * error   # velocity update (momentum + correction)
        h_next = h + dt * gate * v_next     # position update (integration)

    This creates smooth, stable trajectories like a robot controller:
        - alpha: inertia (momentum, smooth movements) - [0, 1]
        - beta: correction strength (feedback) - [0, 2] CLAMPED!
        - gate: amplitude control (safety) - [0, 1]
        - mu: target equilibrium - [0, 2] CLAMPED!
        - dt: timestep (integration speed)

    CRITICAL STABILITY:
        - beta is clamped to [0, 2] - softplus without clamp goes to infinity!
        - velocity is clamped to [-10, 10] - prevents runaway accumulation
        - These constraints were discovered after 400k training steps explosion
    """

    def __init__(
        self,
        hidden_size: int,
        init_alpha: float = 0.9,
        init_beta: float = 0.1,
        init_gate: float = 0.5,
        dt: float = 0.1,
        controller_hidden: int = 64,
        beta_max: float = 2.0,
        velocity_max: float = 10.0,
        mu_min: float = 0.0,
        mu_max: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.beta_max = beta_max
        self.velocity_max = velocity_max
        self.mu_min = mu_min
        self.mu_max = mu_max

        # Learnable equilibrium (target position)
        # Initialize in middle of valid range
        self.mu = nn.Parameter(torch.full((hidden_size,), (mu_min + mu_max) / 2))

        # v0.3.0: Contextual mu projection (INL 2025)
        # mu_contextual = mu + mu_proj(h) - allows mu to adapt based on input
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)  # Start neutral (just mu)

        # Controller MLP - computes alpha, beta, gate from context
        # Input: [h, v] concatenated
        self.controller = nn.Sequential(
            nn.Linear(hidden_size * 2, controller_hidden),
            nn.SiLU(),
            nn.Linear(controller_hidden, hidden_size * 3),  # outputs: alpha, beta, gate
        )

        # Initialize controller biases for desired initial values
        self._init_controller(init_alpha, init_beta, init_gate)

        # Safety clamp (Representation Engineering)
        # Installed via install_safety() method
        self.safety_clamp = None

    def _init_controller(self, init_alpha: float, init_beta: float, init_gate: float):
        """Initialize controller for stable starting values."""
        with torch.no_grad():
            bias = self.controller[-1].bias

            # alpha in [0,1] via sigmoid, init to ~init_alpha
            # sigmoid(x) = init_alpha -> x = log(init_alpha / (1 - init_alpha))
            alpha_bias = torch.log(torch.tensor(init_alpha / (1 - init_alpha + 1e-8)))
            bias[:self.hidden_size].fill_(alpha_bias.item())

            # beta in [0, beta_max] via clamped softplus, init to ~init_beta
            # softplus(x) = init_beta -> x = log(exp(init_beta) - 1)
            beta_bias = torch.log(torch.tensor(max(torch.exp(torch.tensor(init_beta)) - 1, 1e-8)))
            bias[self.hidden_size:self.hidden_size*2].fill_(beta_bias.item())

            # gate in [0,1] via sigmoid, init to ~init_gate
            gate_bias = torch.log(torch.tensor(init_gate / (1 - init_gate + 1e-8)))
            bias[self.hidden_size*2:].fill_(gate_bias.item())

            # Small weights for stable start
            self.controller[-1].weight.normal_(0, 0.01)

    @property
    def mu_clamped(self) -> torch.Tensor:
        """Get mu clamped to valid range [mu_min, mu_max]."""
        return torch.clamp(self.mu, self.mu_min, self.mu_max)

    def forward(
        self,
        h: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        return_mu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply dynamics update.

        Args:
            h: Hidden states [batch, seq_len, hidden_size]
            v: Velocity states [batch, seq_len, hidden_size] (None = init to zero)
            return_mu: Whether to return contextual mu for next layer guidance

        Returns:
            h_next: Updated hidden states
            v_next: Updated velocity states
            mu_contextual: (if return_mu=True) Contextual mu for next layer
        """
        # Initialize velocity if not provided
        if v is None:
            v = torch.zeros_like(h)

        # Controller computes adaptive parameters from [h, v]
        controller_input = torch.cat([h, v], dim=-1)
        controller_out = self.controller(controller_input)

        # Split and apply activations with STABILITY CONSTRAINTS
        alpha_raw, beta_raw, gate_raw = torch.split(
            controller_out, self.hidden_size, dim=-1
        )

        alpha = torch.sigmoid(alpha_raw)  # [0, 1] - inertia

        # CRITICAL FIX: Clamp beta to prevent explosion!
        # softplus can go to infinity, causing NaN after long training
        # Max beta=2.0 keeps dynamics stable (like a real PID controller)
        beta = torch.clamp(F.softplus(beta_raw), max=self.beta_max)  # [0, beta_max]

        gate = torch.sigmoid(gate_raw)  # [0, 1] - amplitude

        # Use clamped mu for equilibrium
        mu = self.mu_clamped

        # Dynamics equations
        error = h - mu                            # deviation from equilibrium
        v_next = alpha * v - beta * error         # velocity update

        # STABILITY: Clamp velocity to prevent runaway accumulation
        # Like velocity limits in real robotics systems
        v_next = torch.clamp(v_next, min=-self.velocity_max, max=self.velocity_max)

        h_next = h + self.dt * gate * v_next      # position update

        # v0.3.0: Contextual mu for next layer guidance (INL 2025)
        mu_contextual = None
        if return_mu:
            mu_contextual = mu + self.mu_proj(h)
            # v0.3.1: Safety clamp on mu_contextual
            if self.safety_clamp is not None:
                mu_contextual = self.safety_clamp(mu_contextual)

        # v0.3.1: Safety clamp on hidden states
        if self.safety_clamp is not None:
            h_next = self.safety_clamp(h_next)

        return h_next, v_next, mu_contextual

    def init_velocity(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize velocity to zero."""
        return torch.zeros(
            batch_size, seq_len, self.hidden_size,
            device=device, dtype=dtype
        )

    def get_dynamics_stats(self) -> dict:
        """Get statistics about current dynamics parameters."""
        with torch.no_grad():
            return {
                "mu_mean": self.mu_clamped.mean().item(),
                "mu_std": self.mu_clamped.std().item(),
                "mu_min": self.mu_clamped.min().item(),
                "mu_max": self.mu_clamped.max().item(),
            }

    # ========== Safety Integration (v0.3.1) ==========

    def install_safety(self, safety_clamp: nn.Module) -> None:
        """
        Install safety clamp for Representation Engineering.

        Args:
            safety_clamp: SafetyClamp module that projects out harm directions
        """
        self.safety_clamp = safety_clamp

    def remove_safety(self) -> None:
        """Remove safety clamp."""
        self.safety_clamp = None

    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety clamping statistics."""
        if self.safety_clamp is not None and hasattr(self.safety_clamp, 'get_stats'):
            return self.safety_clamp.get_stats()
        return {'enabled': False}


class INLDynamicsLite(nn.Module):
    """
    Simplified INL Dynamics with fixed (non-learned) control parameters.

    Use this for:
    - Faster training (fewer parameters)
    - When dynamics should be consistent across all positions
    - Testing/debugging

    Still maintains stability constraints:
    - beta clamped to [0, 2]
    - velocity clamped to [-10, 10]
    - mu clamped to [0, 2]
    """

    def __init__(
        self,
        hidden_size: int,
        alpha: float = 0.9,
        beta: float = 0.1,
        gate: float = 0.5,
        dt: float = 0.1,
        mu_min: float = 0.0,
        mu_max: float = 2.0,
        velocity_max: float = 10.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dt = dt
        self.mu_min = mu_min
        self.mu_max = mu_max
        self.velocity_max = velocity_max

        # Fixed control parameters (clamped for safety)
        self.register_buffer("alpha", torch.tensor(min(max(alpha, 0.0), 1.0)))
        self.register_buffer("beta", torch.tensor(min(max(beta, 0.0), 2.0)))  # CLAMPED!
        self.register_buffer("gate", torch.tensor(min(max(gate, 0.0), 1.0)))

        # Learnable equilibrium only
        self.mu = nn.Parameter(torch.full((hidden_size,), (mu_min + mu_max) / 2))

        # v0.3.0: Contextual mu projection (INL 2025)
        self.mu_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        nn.init.zeros_(self.mu_proj.weight)

        # v0.3.1: Safety clamp (Representation Engineering)
        self.safety_clamp = None

    @property
    def mu_clamped(self) -> torch.Tensor:
        """Get mu clamped to valid range."""
        return torch.clamp(self.mu, self.mu_min, self.mu_max)

    def forward(
        self,
        h: torch.Tensor,
        v: Optional[torch.Tensor] = None,
        return_mu: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Apply fixed-parameter dynamics."""
        if v is None:
            v = torch.zeros_like(h)

        mu = self.mu_clamped

        # Dynamics with fixed parameters
        error = h - mu
        v_next = self.alpha * v - self.beta * error

        # Velocity clamping for stability
        v_next = torch.clamp(v_next, min=-self.velocity_max, max=self.velocity_max)

        h_next = h + self.dt * self.gate * v_next

        # v0.3.0: Contextual mu for next layer guidance
        mu_contextual = None
        if return_mu:
            mu_contextual = mu + self.mu_proj(h)
            # v0.3.1: Safety clamp on mu_contextual
            if self.safety_clamp is not None:
                mu_contextual = self.safety_clamp(mu_contextual)

        # v0.3.1: Safety clamp on hidden states
        if self.safety_clamp is not None:
            h_next = self.safety_clamp(h_next)

        return h_next, v_next, mu_contextual

    def init_velocity(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Initialize velocity to zero."""
        return torch.zeros(
            batch_size, seq_len, self.hidden_size,
            device=device, dtype=dtype
        )

    # ========== Safety Integration (v0.3.1) ==========

    def install_safety(self, safety_clamp: nn.Module) -> None:
        """Install safety clamp for Representation Engineering."""
        self.safety_clamp = safety_clamp

    def remove_safety(self) -> None:
        """Remove safety clamp."""
        self.safety_clamp = None

    def get_safety_stats(self) -> Dict[str, Any]:
        """Get safety clamping statistics."""
        if self.safety_clamp is not None and hasattr(self.safety_clamp, 'get_stats'):
            return self.safety_clamp.get_stats()
        return {'enabled': False}


# Factory function
def create_dynamics(
    hidden_size: int,
    lite: bool = False,
    **kwargs,
) -> nn.Module:
    """
    Create INL Dynamics module.

    Args:
        hidden_size: Hidden dimension
        lite: Use simplified version with fixed parameters
        **kwargs: Additional arguments for the dynamics module

    Returns:
        INLDynamics or INLDynamicsLite module
    """
    if lite:
        return INLDynamicsLite(hidden_size, **kwargs)
    return INLDynamics(hidden_size, **kwargs)
