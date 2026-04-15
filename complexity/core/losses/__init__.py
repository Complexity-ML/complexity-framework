"""Loss primitives for framework-complexity."""

from .causal_lm import causal_lm_loss, CausalLMLossMetrics

__all__ = ["causal_lm_loss", "CausalLMLossMetrics"]
