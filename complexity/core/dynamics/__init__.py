"""
INL Dynamics - Robotics-grade control with velocity tracking.
==============================================================

Second-order dynamical system for stable, smooth representation evolution.

Key insight: LLM hidden states are like robot positions - they need smooth,
controlled trajectories to avoid instability.

Components:
- INLDynamics: Full controller with learnable parameters
- INLDynamicsLite: Simplified version with fixed parameters
- DynamicsConfig: Configuration dataclass

CRITICAL STABILITY NOTES:
- beta MUST be clamped to [0, 2] - softplus goes to infinity!
- velocity MUST be clamped to prevent runaway accumulation
- These constraints discovered after 400k training steps explosion

Usage:
    from complexity.core.dynamics import INLDynamics, DynamicsConfig

    dynamics = INLDynamics(hidden_size=768)
    h_next, v_next = dynamics(hidden_states, velocity_states)
"""

from .inl_dynamics import (
    INLDynamics,
    INLDynamicsLite,
    DynamicsConfig,
)

__all__ = [
    "INLDynamics",
    "INLDynamicsLite",
    "DynamicsConfig",
]
