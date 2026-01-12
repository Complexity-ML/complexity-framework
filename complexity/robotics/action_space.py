"""
Action space representations for robotics.

Provides:
- Action tokenization (discrete bins)
- Continuous action spaces
- Action heads (MLP, diffusion)
- Multi-dimensional action handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass
import math


@dataclass
class ActionConfig:
    """Configuration for action space."""
    action_dim: int = 7                    # e.g., 6-DOF + gripper
    num_bins: int = 256                    # For discretization
    action_min: float = -1.0               # Min action value
    action_max: float = 1.0                # Max action value
    use_discrete: bool = False             # Discrete vs continuous
    action_chunk_size: int = 1             # Predict multiple steps
    hidden_size: int = 768
    num_action_layers: int = 2
    dropout: float = 0.1


class ContinuousActionSpace:
    """
    Continuous action space with normalization.

    Handles scaling and clipping of continuous robot actions.
    """

    def __init__(
        self,
        action_dim: int,
        action_min: Union[float, torch.Tensor] = -1.0,
        action_max: Union[float, torch.Tensor] = 1.0,
    ):
        """
        Args:
            action_dim: Dimension of action space
            action_min: Minimum values (scalar or per-dimension)
            action_max: Maximum values (scalar or per-dimension)
        """
        self.action_dim = action_dim

        if isinstance(action_min, (int, float)):
            self.action_min = torch.full((action_dim,), action_min)
        else:
            self.action_min = action_min

        if isinstance(action_max, (int, float)):
            self.action_max = torch.full((action_dim,), action_max)
        else:
            self.action_max = action_max

        self.action_range = self.action_max - self.action_min

    def normalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Normalize actions to [-1, 1]."""
        device = actions.device
        action_min = self.action_min.to(device)
        action_range = self.action_range.to(device)
        return 2 * (actions - action_min) / action_range - 1

    def denormalize(self, actions: torch.Tensor) -> torch.Tensor:
        """Denormalize actions from [-1, 1] to original range."""
        device = actions.device
        action_min = self.action_min.to(device)
        action_range = self.action_range.to(device)
        return (actions + 1) / 2 * action_range + action_min

    def clip(self, actions: torch.Tensor) -> torch.Tensor:
        """Clip actions to valid range."""
        device = actions.device
        return torch.clamp(
            actions,
            self.action_min.to(device),
            self.action_max.to(device)
        )


class DiscreteActionSpace:
    """
    Discrete action space via binning.

    Converts continuous actions to discrete bins for token prediction.
    Used in RT-1 style models.
    """

    def __init__(
        self,
        action_dim: int,
        num_bins: int = 256,
        action_min: float = -1.0,
        action_max: float = 1.0,
    ):
        """
        Args:
            action_dim: Dimension of action space
            num_bins: Number of discrete bins per dimension
            action_min: Minimum action value
            action_max: Maximum action value
        """
        self.action_dim = action_dim
        self.num_bins = num_bins
        self.action_min = action_min
        self.action_max = action_max
        self.bin_size = (action_max - action_min) / num_bins

        # Total vocabulary size for action tokens
        self.vocab_size = num_bins * action_dim

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode continuous actions to discrete bins.

        Args:
            actions: Continuous actions [batch, action_dim]

        Returns:
            Bin indices [batch, action_dim]
        """
        # Normalize to [0, 1]
        normalized = (actions - self.action_min) / (self.action_max - self.action_min)
        normalized = torch.clamp(normalized, 0, 1 - 1e-6)

        # Convert to bins
        bins = (normalized * self.num_bins).long()

        return bins

    def decode(self, bins: torch.Tensor) -> torch.Tensor:
        """
        Decode discrete bins to continuous actions.

        Args:
            bins: Bin indices [batch, action_dim]

        Returns:
            Continuous actions [batch, action_dim]
        """
        # Get bin centers
        normalized = (bins.float() + 0.5) / self.num_bins

        # Denormalize
        actions = normalized * (self.action_max - self.action_min) + self.action_min

        return actions

    def to_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert actions to unique token IDs.

        Each action dimension gets a separate range of tokens.

        Args:
            actions: Continuous actions [batch, action_dim]

        Returns:
            Token IDs [batch, action_dim]
        """
        bins = self.encode(actions)

        # Add offset for each dimension
        offsets = torch.arange(self.action_dim, device=bins.device) * self.num_bins
        tokens = bins + offsets

        return tokens

    def from_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs back to continuous actions.

        Args:
            tokens: Token IDs [batch, action_dim]

        Returns:
            Continuous actions [batch, action_dim]
        """
        # Remove offset for each dimension
        offsets = torch.arange(self.action_dim, device=tokens.device) * self.num_bins
        bins = tokens - offsets

        return self.decode(bins)


class ActionTokenizer(nn.Module):
    """
    Tokenize actions for transformer models.

    Can operate in discrete (classification) or continuous (regression) mode.
    """

    def __init__(self, config: ActionConfig):
        """
        Args:
            config: Action configuration
        """
        super().__init__()

        self.config = config

        if config.use_discrete:
            self.action_space = DiscreteActionSpace(
                action_dim=config.action_dim,
                num_bins=config.num_bins,
                action_min=config.action_min,
                action_max=config.action_max,
            )
            # Embedding for action tokens
            self.action_embed = nn.Embedding(
                self.action_space.vocab_size,
                config.hidden_size
            )
        else:
            self.action_space = ContinuousActionSpace(
                action_dim=config.action_dim,
                action_min=config.action_min,
                action_max=config.action_max,
            )
            # Linear projection for continuous actions
            self.action_proj = nn.Linear(config.action_dim, config.hidden_size)

    def encode(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Encode actions to embeddings.

        Args:
            actions: Actions [batch, seq, action_dim] or [batch, action_dim]

        Returns:
            Action embeddings [batch, seq, hidden] or [batch, hidden]
        """
        if self.config.use_discrete:
            tokens = self.action_space.to_tokens(actions)
            # Sum embeddings for all action dimensions
            embeddings = self.action_embed(tokens).sum(dim=-2)
        else:
            normalized = self.action_space.normalize(actions)
            embeddings = self.action_proj(normalized)

        return embeddings

    def decode(
        self,
        hidden_states: torch.Tensor,
        action_head: Optional['ActionHead'] = None,
    ) -> torch.Tensor:
        """
        Decode hidden states to actions.

        Args:
            hidden_states: Hidden states [batch, hidden]
            action_head: Action prediction head

        Returns:
            Predicted actions [batch, action_dim]
        """
        if action_head is not None:
            return action_head(hidden_states)

        # Simple linear decode
        if not hasattr(self, 'action_decode'):
            device = hidden_states.device
            self.action_decode = nn.Linear(
                self.config.hidden_size,
                self.config.action_dim
            ).to(device)

        actions = self.action_decode(hidden_states)

        if not self.config.use_discrete:
            actions = self.action_space.denormalize(actions)

        return actions


class ActionHead(nn.Module):
    """
    Action prediction head.

    MLP that predicts actions from hidden states.
    """

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        action_chunk_size: int = 1,
    ):
        """
        Args:
            hidden_size: Input hidden dimension
            action_dim: Output action dimension
            num_layers: Number of MLP layers
            dropout: Dropout rate
            action_chunk_size: Number of action steps to predict
        """
        super().__init__()

        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        output_dim = action_dim * action_chunk_size

        layers = []
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(hidden_size, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Predict actions.

        Args:
            hidden_states: [batch, hidden_size]

        Returns:
            Actions [batch, chunk_size, action_dim] or [batch, action_dim]
        """
        output = self.mlp(hidden_states)

        if self.action_chunk_size > 1:
            batch_size = output.size(0)
            output = output.view(batch_size, self.action_chunk_size, self.action_dim)

        return torch.tanh(output)  # Bound to [-1, 1]


class DiffusionActionHead(nn.Module):
    """
    Diffusion-based action prediction head.

    Uses denoising diffusion for action generation.
    Provides better multi-modal action distributions.

    Reference: Diffusion Policy (https://arxiv.org/abs/2303.04137)
    """

    def __init__(
        self,
        hidden_size: int,
        action_dim: int,
        action_chunk_size: int = 8,
        num_diffusion_steps: int = 100,
        num_layers: int = 4,
    ):
        """
        Args:
            hidden_size: Conditioning hidden dimension
            action_dim: Action dimension
            action_chunk_size: Number of actions to predict
            num_diffusion_steps: Diffusion timesteps
            num_layers: Denoiser MLP layers
        """
        super().__init__()

        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.num_diffusion_steps = num_diffusion_steps

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Denoising network
        input_dim = action_dim * action_chunk_size + hidden_size + hidden_size

        layers = [nn.Linear(input_dim, hidden_size * 2), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.SiLU(),
            ])
        layers.append(nn.Linear(hidden_size * 2, action_dim * action_chunk_size))

        self.denoiser = nn.Sequential(*layers)

        # Noise schedule (linear)
        self.register_buffer(
            'betas',
            torch.linspace(1e-4, 0.02, num_diffusion_steps)
        )
        alphas = 1 - self.betas
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))

    def forward(
        self,
        hidden_states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass.

        Training: Returns noise prediction loss
        Inference: Returns sampled actions

        Args:
            hidden_states: Conditioning [batch, hidden]
            actions: Ground truth actions for training [batch, chunk, action_dim]
        """
        if actions is not None:
            return self._training_step(hidden_states, actions)
        else:
            return self._sample(hidden_states)

    def _training_step(
        self,
        hidden_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training: predict noise."""
        batch_size = actions.size(0)
        device = actions.device

        # Flatten actions
        actions_flat = actions.view(batch_size, -1)

        # Sample random timesteps
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)

        # Sample noise
        noise = torch.randn_like(actions_flat)

        # Add noise (forward diffusion)
        alpha_t = self.alphas_cumprod[t].view(-1, 1)
        noisy_actions = torch.sqrt(alpha_t) * actions_flat + torch.sqrt(1 - alpha_t) * noise

        # Predict noise
        time_emb = self.time_embed(t.float().unsqueeze(-1) / self.num_diffusion_steps)
        denoiser_input = torch.cat([noisy_actions, hidden_states, time_emb], dim=-1)
        noise_pred = self.denoiser(denoiser_input)

        # Loss
        loss = F.mse_loss(noise_pred, noise)

        return loss, noise_pred

    @torch.no_grad()
    def _sample(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Inference: sample actions via reverse diffusion."""
        batch_size = hidden_states.size(0)
        device = hidden_states.device

        # Start from pure noise
        x = torch.randn(batch_size, self.action_dim * self.action_chunk_size, device=device)

        # Reverse diffusion
        for t in reversed(range(self.num_diffusion_steps)):
            t_tensor = torch.full((batch_size,), t, device=device)
            time_emb = self.time_embed(t_tensor.float().unsqueeze(-1) / self.num_diffusion_steps)

            # Predict noise
            denoiser_input = torch.cat([x, hidden_states, time_emb], dim=-1)
            noise_pred = self.denoiser(denoiser_input)

            # Denoise step
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]

            if t > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t - 1]
                beta_t = self.betas[t]

                # Compute x_{t-1}
                pred_x0 = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)

                # Add noise for next step
                noise = torch.randn_like(x)
                x = (
                    torch.sqrt(alpha_cumprod_prev) * pred_x0 +
                    torch.sqrt(1 - alpha_cumprod_prev) * noise
                )
            else:
                # Final step: just predict x0
                x = (x - torch.sqrt(1 - alpha_cumprod_t) * noise_pred) / torch.sqrt(alpha_cumprod_t)

        # Reshape to action chunks
        actions = x.view(batch_size, self.action_chunk_size, self.action_dim)
        return torch.tanh(actions)
