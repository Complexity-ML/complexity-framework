"""
Safety Module for Complexity Framework

Representation Engineering approach for inference safety.
Clamps activations along learned harm directions.

This module provides:
- SafetyClamp: Core clamping mechanism
- SafetyCallback: Training callback for safety monitoring
- ContrastiveSafetyLoss: Loss for learning harm directions during SFT
- Integration with INL Dynamics

Usage:
    from complexity.utils.safety import (
        SafetyClamp,
        install_safety,
        SafetyCallback,
    )

    # Install on model
    install_safety(model, harm_direction, threshold=2.0)

    # Or use callback during training
    trainer = Trainer(callbacks=[SafetyCallback(model)])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Any, Union, Callable
from dataclasses import dataclass
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Safety Configuration
# =============================================================================

@dataclass
class SafetyConfig:
    """Configuration for safety clamping."""
    enabled: bool = False
    threshold: float = 2.0
    soft_clamp: bool = True
    temperature: float = 1.0
    layers: List[int] = None  # Which layers to clamp (default: last 3)
    clamp_mu: bool = True
    clamp_hidden: bool = True
    clamp_velocity: bool = False
    direction_path: Optional[str] = None


# =============================================================================
# Safety Clamp
# =============================================================================

class SafetyClamp(nn.Module):
    """
    Clamps activations along harm direction.

    Representation Engineering approach:
        projection = activation @ harm_direction
        if projection > threshold:
            activation -= (projection - threshold) * harm_direction

    This prevents the model from generating outputs that have
    high projection onto the learned "harm" direction in activation space.
    """

    def __init__(
        self,
        hidden_size: int,
        threshold: float = 2.0,
        soft_clamp: bool = True,
        temperature: float = 1.0,
    ):
        """
        Args:
            hidden_size: Dimension of hidden states
            threshold: Maximum allowed projection onto harm direction
            soft_clamp: Use differentiable soft clamping
            temperature: Temperature for soft clamp (lower = sharper)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.threshold = threshold
        self.soft_clamp = soft_clamp
        self.temperature = max(temperature, 1e-8)

        # Harm direction (initialized to zero = no effect)
        self.register_buffer('harm_direction', torch.zeros(hidden_size))

        # Statistics
        self.register_buffer('num_clamped', torch.tensor(0))
        self.register_buffer('total_processed', torch.tensor(0))
        self.register_buffer('max_projection', torch.tensor(0.0))

        self.enabled = False

    def set_harm_direction(self, direction: torch.Tensor):
        """Set and normalize harm direction."""
        direction = F.normalize(direction.float(), dim=0)
        self.harm_direction.copy_(direction)
        logger.info(f"Safety: harm direction set (norm={direction.norm():.4f})")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp activations.

        Args:
            x: Activations [..., hidden_size]

        Returns:
            Clamped activations, same shape
        """
        if not self.enabled or self.harm_direction.norm() < 1e-6:
            return x

        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        # Project onto harm direction
        projection = x_flat @ self.harm_direction

        # Update statistics
        self.total_processed += projection.numel()
        self.num_clamped += (projection > self.threshold).sum()
        self.max_projection = max(self.max_projection, projection.max())

        # Compute correction
        if self.soft_clamp:
            excess = projection - self.threshold
            clamp_factor = torch.sigmoid(excess / self.temperature)
            correction = clamp_factor * F.relu(excess)
        else:
            correction = F.relu(projection - self.threshold)

        # Subtract excess projection
        x_clamped = x_flat - correction.unsqueeze(-1) * self.harm_direction

        return x_clamped.view(original_shape)

    def get_stats(self) -> Dict[str, float]:
        """Get clamping statistics."""
        total = self.total_processed.item()
        if total == 0:
            return {'clamp_rate': 0.0, 'total': 0, 'enabled': self.enabled}
        return {
            'clamp_rate': self.num_clamped.item() / total,
            'num_clamped': self.num_clamped.item(),
            'total': total,
            'max_projection': self.max_projection.item(),
            'enabled': self.enabled,
        }

    def reset_stats(self):
        """Reset statistics."""
        self.num_clamped.zero_()
        self.total_processed.zero_()
        self.max_projection.zero_()


class MultiDirectionSafetyClamp(nn.Module):
    """
    Clamps multiple harm directions (one per category).

    Useful for different types of harmful content:
    - violence
    - drugs
    - weapons
    - etc.
    """

    def __init__(
        self,
        hidden_size: int,
        num_directions: int = 8,
        threshold: float = 2.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_directions = num_directions
        self.threshold = threshold

        # [num_directions, hidden_size]
        self.register_buffer('harm_directions', torch.zeros(num_directions, hidden_size))
        self.register_buffer('thresholds', torch.full((num_directions,), threshold))
        self.register_buffer('active_mask', torch.zeros(num_directions, dtype=torch.bool))

        self.enabled = False

    def set_direction(self, index: int, direction: torch.Tensor, threshold: float = None):
        """Set a specific harm direction."""
        direction = F.normalize(direction.float(), dim=0)
        self.harm_directions[index] = direction
        self.active_mask[index] = True
        if threshold is not None:
            self.thresholds[index] = threshold

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Clamp all active directions."""
        if not self.enabled or not self.active_mask.any():
            return x

        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])

        active_dirs = self.harm_directions[self.active_mask]
        active_thresholds = self.thresholds[self.active_mask]

        for i in range(active_dirs.shape[0]):
            projection = x_flat @ active_dirs[i]
            correction = F.relu(projection - active_thresholds[i])
            x_flat = x_flat - correction.unsqueeze(-1) * active_dirs[i]

        return x_flat.view(original_shape)


# =============================================================================
# Contrastive Safety Loss
# =============================================================================

class ContrastiveSafetyLoss(nn.Module):
    """
    Contrastive loss for learning harm direction during SFT.

    Given pairs of (safe, harmful) examples, learns a direction such that:
    - harmful_activations @ direction > threshold
    - safe_activations @ direction < threshold
    """

    def __init__(
        self,
        hidden_size: int,
        margin: float = 1.0,
        temperature: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.margin = margin
        self.temperature = max(temperature, 1e-8)

        # Learnable harm direction
        self.harm_direction = nn.Parameter(
            F.normalize(torch.randn(hidden_size), dim=0)
        )

    def forward(
        self,
        safe_activations: torch.Tensor,
        harmful_activations: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive safety loss.

        Args:
            safe_activations: [batch, seq, hidden] or [batch, hidden]
            harmful_activations: Same shape

        Returns:
            Dict with loss and metrics
        """
        # Mean pool if sequence
        if safe_activations.dim() == 3:
            safe_activations = safe_activations.mean(dim=1)
            harmful_activations = harmful_activations.mean(dim=1)

        direction = F.normalize(self.harm_direction, dim=0)

        safe_proj = safe_activations @ direction
        harmful_proj = harmful_activations @ direction

        # Margin loss
        margin_loss = F.relu(self.margin - (harmful_proj - safe_proj)).mean()

        # Contrastive loss
        logits = torch.stack([harmful_proj, safe_proj], dim=1) / self.temperature
        labels = torch.zeros(logits.shape[0], device=logits.device, dtype=torch.long)
        contrast_loss = F.cross_entropy(logits, labels)

        return {
            'loss': margin_loss + contrast_loss,
            'margin_loss': margin_loss,
            'contrast_loss': contrast_loss,
            'separation': (harmful_proj - safe_proj).mean(),
        }

    def get_direction(self) -> torch.Tensor:
        """Get normalized harm direction."""
        return F.normalize(self.harm_direction, dim=0)


# =============================================================================
# Model Integration
# =============================================================================

def install_safety(
    model: nn.Module,
    harm_direction: torch.Tensor,
    threshold: float = 2.0,
    layers: List[int] = None,
    soft_clamp: bool = True,
) -> None:
    """
    Install safety clamping on a model.

    Args:
        model: Model with layers that have INL dynamics
        harm_direction: [hidden_size] harm direction vector
        threshold: Clamping threshold
        layers: Which layers to install on (default: last 3)
        soft_clamp: Use soft clamping
    """
    # Find layers
    model_layers = _find_layers(model)
    if not model_layers:
        raise ValueError("Cannot find layers in model")

    if layers is None:
        layers = [-3, -2, -1]

    hidden_size = harm_direction.shape[0]

    for idx in layers:
        layer = model_layers[idx]

        # Create safety clamp
        safety = SafetyClamp(hidden_size, threshold=threshold, soft_clamp=soft_clamp)
        safety.set_harm_direction(harm_direction)
        safety.enabled = True

        # Install on dynamics or layer
        if hasattr(layer, 'dynamics'):
            layer.dynamics.safety_clamp = safety
        elif hasattr(layer, 'inl_dynamics'):
            layer.inl_dynamics.safety_clamp = safety
        else:
            # Fallback: store on layer itself
            layer.safety_clamp = safety

    logger.info(f"Safety installed on layers {layers} with threshold={threshold}")


def remove_safety(model: nn.Module) -> None:
    """Remove safety clamping from model."""
    model_layers = _find_layers(model)
    if not model_layers:
        return

    for layer in model_layers:
        if hasattr(layer, 'dynamics') and hasattr(layer.dynamics, 'safety_clamp'):
            del layer.dynamics.safety_clamp
        elif hasattr(layer, 'inl_dynamics') and hasattr(layer.inl_dynamics, 'safety_clamp'):
            del layer.inl_dynamics.safety_clamp
        elif hasattr(layer, 'safety_clamp'):
            del layer.safety_clamp

    logger.info("Safety removed from model")


def get_safety_stats(model: nn.Module) -> Dict[str, Any]:
    """Get safety statistics from model."""
    model_layers = _find_layers(model)
    if not model_layers:
        return {'error': 'No layers found'}

    stats = {}
    for i, layer in enumerate(model_layers):
        clamp = None
        if hasattr(layer, 'dynamics') and hasattr(layer.dynamics, 'safety_clamp'):
            clamp = layer.dynamics.safety_clamp
        elif hasattr(layer, 'inl_dynamics') and hasattr(layer.inl_dynamics, 'safety_clamp'):
            clamp = layer.inl_dynamics.safety_clamp
        elif hasattr(layer, 'safety_clamp'):
            clamp = layer.safety_clamp

        if clamp is not None:
            stats[f'layer_{i}'] = clamp.get_stats()

    return stats


def _find_layers(model: nn.Module) -> List[nn.Module]:
    """Find transformer layers in model."""
    paths = [
        'layers',
        'model.layers',
        'transformer.layers',
        'blocks',
        'h',
    ]

    for attr in paths:
        parts = attr.split('.')
        module = model
        try:
            for part in parts:
                module = getattr(module, part)
            if isinstance(module, (nn.ModuleList, list)):
                return list(module)
        except AttributeError:
            continue

    return []


# =============================================================================
# Training Callback
# =============================================================================

class SafetyCallback:
    """
    Training callback for safety monitoring.

    Logs safety statistics during training.
    """

    def __init__(
        self,
        model: nn.Module,
        log_every: int = 100,
        reset_stats_every: int = 1000,
    ):
        self.model = model
        self.log_every = log_every
        self.reset_stats_every = reset_stats_every
        self.step = 0

    def on_step_end(self, metrics: Dict[str, Any] = None):
        """Called at end of training step."""
        self.step += 1

        if self.step % self.log_every == 0:
            stats = get_safety_stats(self.model)
            if stats and 'error' not in stats:
                logger.info(f"Safety stats at step {self.step}: {stats}")

        if self.step % self.reset_stats_every == 0:
            self._reset_all_stats()

    def _reset_all_stats(self):
        """Reset statistics on all safety clamps."""
        model_layers = _find_layers(self.model)
        for layer in model_layers:
            clamp = None
            if hasattr(layer, 'dynamics') and hasattr(layer.dynamics, 'safety_clamp'):
                clamp = layer.dynamics.safety_clamp
            elif hasattr(layer, 'safety_clamp'):
                clamp = layer.safety_clamp

            if clamp is not None and hasattr(clamp, 'reset_stats'):
                clamp.reset_stats()


# =============================================================================
# Load/Save Utilities
# =============================================================================

def load_harm_direction(path: Union[str, Path], device: torch.device = None) -> torch.Tensor:
    """Load harm direction from file."""
    path = Path(path)
    data = torch.load(path, map_location=device or 'cpu')

    if isinstance(data, torch.Tensor):
        direction = data
    elif isinstance(data, dict):
        direction = data.get('harm_direction', data.get('direction', data.get('default')))
        if direction is None:
            raise ValueError(f"Could not find harm direction in {path}")
    else:
        raise ValueError(f"Unknown format in {path}")

    return F.normalize(direction.float(), dim=0)


def save_harm_direction(
    direction: torch.Tensor,
    path: Union[str, Path],
    metadata: Dict[str, Any] = None,
):
    """Save harm direction to file."""
    path = Path(path)
    data = {
        'harm_direction': F.normalize(direction.float(), dim=0),
        'hidden_size': direction.shape[0],
    }
    if metadata:
        data.update(metadata)
    torch.save(data, path)
    logger.info(f"Saved harm direction to {path}")
