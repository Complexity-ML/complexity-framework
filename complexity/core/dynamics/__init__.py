"""
Mu-Projection — cross-layer top-down guidance signal.
======================================================

Lightweight projection that provides top-down mu-guidance
to steer expert routing and representation evolution.

Components:
- MuProjection: Cross-layer guidance projection

Usage:
    from complexity.core.dynamics import MuProjection

    mu = MuProjection(hidden_size=768)
"""

from .mu_projection import MuProjection

__all__ = [
    "MuProjection",
]
