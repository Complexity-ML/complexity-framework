"""
World models for model-based robotics.

Provides:
- Latent dynamics models
- Reward prediction
- DreamerV3-style world model
- Planning with world models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class WorldModelConfig:
    """Configuration for world model."""
    state_dim: int = 32              # Observation dimension
    action_dim: int = 7
    latent_dim: int = 256            # Latent state dimension
    hidden_size: int = 512
    num_layers: int = 2

    # RSSM (Recurrent State Space Model)
    rssm_hidden_size: int = 256
    stochastic_size: int = 32        # Stochastic latent dimension
    discrete_latent: bool = True     # Discrete or continuous latent
    num_categories: int = 32         # For discrete latent

    # Prediction
    predict_reward: bool = True
    predict_terminal: bool = True

    # Training
    kl_weight: float = 1.0
    reward_weight: float = 1.0
    terminal_weight: float = 1.0


class LatentDynamics(nn.Module):
    """
    Latent dynamics model.

    Learns dynamics in latent space: z_{t+1} = f(z_t, a_t)
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        residual: bool = True,
    ):
        """
        Args:
            latent_dim: Latent state dimension
            action_dim: Action dimension
            hidden_size: Hidden layer size
            num_layers: Number of layers
            residual: Use residual connection
        """
        super().__init__()

        self.residual = residual

        layers = [
            nn.Linear(latent_dim + action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ELU(),
        ]

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ELU(),
            ])

        layers.append(nn.Linear(hidden_size, latent_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next latent state.

        Args:
            latent: Current latent [batch, latent_dim]
            action: Action [batch, action_dim]

        Returns:
            Next latent [batch, latent_dim]
        """
        x = torch.cat([latent, action], dim=-1)
        delta = self.net(x)

        if self.residual:
            return latent + delta
        else:
            return delta


class RewardPredictor(nn.Module):
    """
    Reward prediction head.

    Predicts reward from latent state.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict reward."""
        return self.net(latent).squeeze(-1)


class TerminalPredictor(nn.Module):
    """
    Terminal/done prediction head.

    Predicts probability of episode termination.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        """Predict termination probability."""
        return torch.sigmoid(self.net(latent)).squeeze(-1)


class Encoder(nn.Module):
    """
    Observation encoder.

    Maps observations to latent space.
    """

    def __init__(
        self,
        state_dim: int,
        latent_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Decoder(nn.Module):
    """
    Observation decoder.

    Reconstructs observations from latent.
    """

    def __init__(
        self,
        latent_dim: int,
        state_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ELU(),
            nn.Linear(hidden_size, state_dim),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class RSSM(nn.Module):
    """
    Recurrent State Space Model.

    Core component of DreamerV3.
    Combines deterministic recurrent state with stochastic latent.

    h_t = f(h_{t-1}, z_{t-1}, a_{t-1})  # Deterministic
    z_t ~ p(z_t | h_t)                   # Prior
    z_t ~ q(z_t | h_t, o_t)              # Posterior
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        rssm_hidden_size: int = 256,
        stochastic_size: int = 32,
        discrete: bool = True,
        num_categories: int = 32,
    ):
        """
        Args:
            state_dim: Observation dimension
            action_dim: Action dimension
            rssm_hidden_size: Deterministic hidden size
            stochastic_size: Stochastic latent size
            discrete: Use discrete latent
            num_categories: Categories per dimension (if discrete)
        """
        super().__init__()

        self.rssm_hidden_size = rssm_hidden_size
        self.stochastic_size = stochastic_size
        self.discrete = discrete
        self.num_categories = num_categories

        if discrete:
            self.latent_dim = stochastic_size * num_categories
        else:
            self.latent_dim = stochastic_size * 2  # mean + logvar

        # Recurrent model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.rnn = nn.GRUCell(
            self.latent_dim + action_dim,
            rssm_hidden_size,
        )

        # Prior: p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(rssm_hidden_size, rssm_hidden_size),
            nn.ELU(),
            nn.Linear(rssm_hidden_size, self.latent_dim),
        )

        # Posterior: q(z_t | h_t, o_t)
        self.posterior_net = nn.Sequential(
            nn.Linear(rssm_hidden_size + state_dim, rssm_hidden_size),
            nn.ELU(),
            nn.Linear(rssm_hidden_size, self.latent_dim),
        )

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(state_dim, rssm_hidden_size),
            nn.ELU(),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial RSSM state."""
        h = torch.zeros(batch_size, self.rssm_hidden_size, device=device)
        z = torch.zeros(batch_size, self.stochastic_size, device=device)
        return h, z

    def _sample_latent(self, logits: torch.Tensor) -> torch.Tensor:
        """Sample latent from logits."""
        if self.discrete:
            # Reshape to [batch, stochastic_size, num_categories]
            logits = logits.view(-1, self.stochastic_size, self.num_categories)
            dist = torch.distributions.OneHotCategorical(logits=logits)
            sample = dist.sample()
            # Straight-through gradient
            sample = sample + dist.probs - dist.probs.detach()
            return sample.view(-1, self.stochastic_size * self.num_categories)
        else:
            mean, logvar = logits.chunk(2, dim=-1)
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mean + std * eps

    def _kl_loss(
        self,
        prior_logits: torch.Tensor,
        posterior_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute KL divergence between prior and posterior."""
        if self.discrete:
            prior = prior_logits.view(-1, self.stochastic_size, self.num_categories)
            posterior = posterior_logits.view(-1, self.stochastic_size, self.num_categories)

            prior_dist = torch.distributions.OneHotCategorical(logits=prior)
            posterior_dist = torch.distributions.OneHotCategorical(logits=posterior)

            kl = torch.distributions.kl_divergence(posterior_dist, prior_dist)
            return kl.sum(dim=-1)
        else:
            prior_mean, prior_logvar = prior_logits.chunk(2, dim=-1)
            post_mean, post_logvar = posterior_logits.chunk(2, dim=-1)

            kl = 0.5 * (
                prior_logvar - post_logvar +
                (post_logvar.exp() + (post_mean - prior_mean) ** 2) / prior_logvar.exp() - 1
            )
            return kl.sum(dim=-1)

    def observe(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Process observation and update state.

        Args:
            obs: Observation [batch, state_dim]
            action: Previous action [batch, action_dim]
            state: (h, z) tuple

        Returns:
            Tuple of (new_state, prior_logits, posterior_logits)
        """
        h, z = state

        # Update deterministic state
        rnn_input = torch.cat([z, action], dim=-1)
        h_new = self.rnn(rnn_input, h)

        # Compute prior
        prior_logits = self.prior_net(h_new)

        # Compute posterior (with observation)
        obs_enc = self.obs_encoder(obs)
        posterior_logits = self.posterior_net(torch.cat([h_new, obs], dim=-1))

        # Sample from posterior
        z_new = self._sample_latent(posterior_logits)

        return (h_new, z_new), prior_logits, posterior_logits

    def imagine(
        self,
        action: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Imagine forward without observation (for planning).

        Args:
            action: Action to take [batch, action_dim]
            state: (h, z) tuple

        Returns:
            Tuple of (new_state, prior_logits)
        """
        h, z = state

        # Update deterministic state
        rnn_input = torch.cat([z, action], dim=-1)
        h_new = self.rnn(rnn_input, h)

        # Sample from prior
        prior_logits = self.prior_net(h_new)
        z_new = self._sample_latent(prior_logits)

        return (h_new, z_new), prior_logits

    def get_latent(self, state: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Get full latent representation."""
        h, z = state
        return torch.cat([h, z], dim=-1)


class DreamerV3(nn.Module):
    """
    DreamerV3 World Model.

    State-of-the-art model-based RL algorithm.
    Learns world model and uses it for planning.

    Reference: https://arxiv.org/abs/2301.04104
    """

    def __init__(self, config: WorldModelConfig):
        """
        Args:
            config: World model configuration
        """
        super().__init__()

        self.config = config

        # RSSM
        self.rssm = RSSM(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            rssm_hidden_size=config.rssm_hidden_size,
            stochastic_size=config.stochastic_size,
            discrete=config.discrete_latent,
            num_categories=config.num_categories,
        )

        # Full latent dimension
        if config.discrete_latent:
            stoch_dim = config.stochastic_size * config.num_categories
        else:
            stoch_dim = config.stochastic_size
        full_latent_dim = config.rssm_hidden_size + stoch_dim

        # Decoder (reconstruct observations)
        self.decoder = Decoder(
            latent_dim=full_latent_dim,
            state_dim=config.state_dim,
            hidden_size=config.hidden_size,
        )

        # Reward predictor
        if config.predict_reward:
            self.reward_predictor = RewardPredictor(
                latent_dim=full_latent_dim,
                hidden_size=config.hidden_size,
            )

        # Terminal predictor
        if config.predict_terminal:
            self.terminal_predictor = TerminalPredictor(
                latent_dim=full_latent_dim,
                hidden_size=config.hidden_size,
            )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
        terminals: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.

        Args:
            observations: [batch, seq, state_dim]
            actions: [batch, seq, action_dim]
            rewards: [batch, seq]
            terminals: [batch, seq]

        Returns:
            Dictionary with losses and predictions
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device

        # Initialize state
        state = self.rssm.initial_state(batch_size, device)

        # Process sequence
        prior_logits_list = []
        posterior_logits_list = []
        latents = []

        for t in range(seq_len):
            obs_t = observations[:, t]
            action_t = actions[:, t] if t < seq_len - 1 else torch.zeros_like(actions[:, 0])

            state, prior_logits, posterior_logits = self.rssm.observe(
                obs_t, action_t, state
            )

            prior_logits_list.append(prior_logits)
            posterior_logits_list.append(posterior_logits)
            latents.append(self.rssm.get_latent(state))

        # Stack
        prior_logits = torch.stack(prior_logits_list, dim=1)
        posterior_logits = torch.stack(posterior_logits_list, dim=1)
        latents = torch.stack(latents, dim=1)

        # Compute losses
        losses = {}

        # KL loss
        kl = self.rssm._kl_loss(prior_logits.view(-1, prior_logits.size(-1)),
                                 posterior_logits.view(-1, posterior_logits.size(-1)))
        losses['kl'] = kl.view(batch_size, seq_len).mean()

        # Reconstruction loss
        recon = self.decoder(latents.view(-1, latents.size(-1)))
        recon = recon.view(batch_size, seq_len, -1)
        losses['recon'] = F.mse_loss(recon, observations)

        # Reward loss
        if self.config.predict_reward and rewards is not None:
            reward_pred = self.reward_predictor(latents.view(-1, latents.size(-1)))
            reward_pred = reward_pred.view(batch_size, seq_len)
            losses['reward'] = F.mse_loss(reward_pred, rewards)

        # Terminal loss
        if self.config.predict_terminal and terminals is not None:
            terminal_pred = self.terminal_predictor(latents.view(-1, latents.size(-1)))
            terminal_pred = terminal_pred.view(batch_size, seq_len)
            losses['terminal'] = F.binary_cross_entropy(terminal_pred, terminals.float())

        # Total loss
        total_loss = (
            losses['recon'] +
            self.config.kl_weight * losses['kl']
        )
        if 'reward' in losses:
            total_loss += self.config.reward_weight * losses['reward']
        if 'terminal' in losses:
            total_loss += self.config.terminal_weight * losses['terminal']

        losses['total'] = total_loss

        return losses

    @torch.no_grad()
    def imagine_trajectory(
        self,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
        policy: nn.Module,
        horizon: int = 15,
    ) -> Dict[str, torch.Tensor]:
        """
        Imagine trajectory using learned dynamics.

        Args:
            initial_state: Starting RSSM state
            policy: Policy network that maps latent to action
            horizon: Imagination horizon

        Returns:
            Dictionary with imagined trajectory
        """
        state = initial_state
        latents = []
        actions = []
        rewards = []

        for _ in range(horizon):
            latent = self.rssm.get_latent(state)
            latents.append(latent)

            # Get action from policy
            action = policy(latent)
            actions.append(action)

            # Predict reward
            reward = self.reward_predictor(latent)
            rewards.append(reward)

            # Imagine forward
            state, _ = self.rssm.imagine(action, state)

        return {
            'latents': torch.stack(latents, dim=1),
            'actions': torch.stack(actions, dim=1),
            'rewards': torch.stack(rewards, dim=1),
        }


class WorldModel(nn.Module):
    """
    Simplified world model interface.

    For simpler use cases than full DreamerV3.
    """

    def __init__(self, config: WorldModelConfig):
        """
        Args:
            config: World model configuration
        """
        super().__init__()

        self.config = config

        # Encoder
        self.encoder = Encoder(
            state_dim=config.state_dim,
            latent_dim=config.latent_dim,
            hidden_size=config.hidden_size,
        )

        # Dynamics
        self.dynamics = LatentDynamics(
            latent_dim=config.latent_dim,
            action_dim=config.action_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
        )

        # Decoder
        self.decoder = Decoder(
            latent_dim=config.latent_dim,
            state_dim=config.state_dim,
            hidden_size=config.hidden_size,
        )

        # Reward predictor
        if config.predict_reward:
            self.reward_predictor = RewardPredictor(
                latent_dim=config.latent_dim,
                hidden_size=config.hidden_size,
            )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            states: Current states [batch, state_dim]
            actions: Actions [batch, action_dim]
            next_states: Next states [batch, state_dim]
            rewards: Rewards [batch]

        Returns:
            Dictionary with losses
        """
        # Encode
        z = self.encoder(states)
        z_next_true = self.encoder(next_states)

        # Predict next latent
        z_next_pred = self.dynamics(z, actions)

        # Decode
        recon = self.decoder(z)
        next_recon = self.decoder(z_next_pred)

        # Losses
        losses = {
            'recon': F.mse_loss(recon, states),
            'dynamics': F.mse_loss(z_next_pred, z_next_true.detach()),
            'next_recon': F.mse_loss(next_recon, next_states),
        }

        if self.config.predict_reward and rewards is not None:
            reward_pred = self.reward_predictor(z)
            losses['reward'] = F.mse_loss(reward_pred, rewards)

        losses['total'] = sum(losses.values())

        return losses

    @torch.no_grad()
    def predict(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next state and reward.

        Returns:
            Tuple of (next_state, reward)
        """
        z = self.encoder(state)
        z_next = self.dynamics(z, action)
        next_state = self.decoder(z_next)

        reward = None
        if self.config.predict_reward:
            reward = self.reward_predictor(z)

        return next_state, reward
