"""
Imitation learning algorithms.

Provides:
- Behavior Cloning (BC)
- DAgger (Dataset Aggregation)
- GAIL (Generative Adversarial Imitation Learning)
- Inverse Dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple, Callable
from dataclasses import dataclass
import math


@dataclass
class BCConfig:
    """Configuration for Behavior Cloning."""
    state_dim: int = 32
    action_dim: int = 7
    hidden_size: int = 256
    num_layers: int = 3
    dropout: float = 0.1
    action_type: str = "continuous"  # continuous or discrete
    num_bins: int = 256              # For discrete actions
    use_history: bool = False
    history_length: int = 10
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5


class BehaviorCloning(nn.Module):
    """
    Behavior Cloning (BC).

    Supervised learning from expert demonstrations.
    Maps observations to actions via regression or classification.
    """

    def __init__(self, config: BCConfig):
        """
        Args:
            config: BC configuration
        """
        super().__init__()

        self.config = config

        # State encoder
        if config.use_history:
            self.state_encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=config.hidden_size,
                    nhead=8,
                    dim_feedforward=config.hidden_size * 4,
                    dropout=config.dropout,
                    batch_first=True,
                ),
                num_layers=2,
            )
            self.state_proj = nn.Linear(config.state_dim, config.hidden_size)
        else:
            self.state_encoder = None
            layers = [nn.Linear(config.state_dim, config.hidden_size), nn.ReLU()]
            for _ in range(config.num_layers - 2):
                layers.extend([
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.LayerNorm(config.hidden_size),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ])
            self.feature_extractor = nn.Sequential(*layers)

        # Action head
        if config.action_type == "continuous":
            self.action_head = nn.Sequential(
                nn.Linear(config.hidden_size, config.hidden_size),
                nn.ReLU(),
                nn.Linear(config.hidden_size, config.action_dim),
                nn.Tanh(),
            )
            # Optional: learnable action std for Gaussian policy
            self.log_std = nn.Parameter(torch.zeros(config.action_dim))
        else:
            self.action_head = nn.Linear(
                config.hidden_size,
                config.action_dim * config.num_bins
            )

    def forward(
        self,
        states: torch.Tensor,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict actions.

        Args:
            states: States [batch, state_dim] or [batch, seq, state_dim]
            deterministic: Use mean action (no sampling)

        Returns:
            Dictionary with actions and optionally log_prob
        """
        if self.config.use_history and states.dim() == 3:
            # Encode history
            x = self.state_proj(states)
            x = self.state_encoder(x)
            features = x[:, -1, :]  # Last position
        else:
            if states.dim() == 3:
                states = states[:, -1, :]  # Take last state
            features = self.feature_extractor(states)

        if self.config.action_type == "continuous":
            action_mean = self.action_head(features)

            if deterministic:
                return {'actions': action_mean}

            # Sample from Gaussian
            action_std = self.log_std.exp()
            noise = torch.randn_like(action_mean)
            actions = action_mean + action_std * noise

            # Compute log probability
            log_prob = -0.5 * (
                ((actions - action_mean) / action_std) ** 2 +
                2 * self.log_std +
                math.log(2 * math.pi)
            ).sum(dim=-1)

            return {
                'actions': torch.tanh(actions),
                'action_mean': action_mean,
                'action_std': action_std,
                'log_prob': log_prob,
            }
        else:
            logits = self.action_head(features)
            logits = logits.view(-1, self.config.action_dim, self.config.num_bins)

            if deterministic:
                action_bins = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits, dim=-1)
                action_bins = torch.multinomial(
                    probs.view(-1, self.config.num_bins),
                    num_samples=1
                ).view(-1, self.config.action_dim)

            # Convert to continuous
            actions = (action_bins.float() + 0.5) / self.config.num_bins * 2 - 1

            return {
                'actions': actions,
                'action_logits': logits,
            }

    def compute_loss(
        self,
        states: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute BC loss.

        Args:
            states: States [batch, state_dim]
            expert_actions: Expert actions [batch, action_dim]

        Returns:
            Loss scalar
        """
        outputs = self.forward(states, deterministic=True)

        if self.config.action_type == "continuous":
            # MSE loss
            loss = F.mse_loss(outputs['actions'], expert_actions)
        else:
            # Cross-entropy loss
            # Convert expert actions to bins
            expert_bins = ((expert_actions + 1) / 2 * self.config.num_bins).long()
            expert_bins = torch.clamp(expert_bins, 0, self.config.num_bins - 1)

            logits = outputs['action_logits']
            loss = F.cross_entropy(
                logits.view(-1, self.config.num_bins),
                expert_bins.view(-1),
            )

        return loss


class DAgger(nn.Module):
    """
    DAgger: Dataset Aggregation.

    Iteratively queries expert during policy rollouts
    to aggregate training data.

    Reference: https://arxiv.org/abs/1011.0686
    """

    def __init__(
        self,
        policy: nn.Module,
        expert_policy: Callable,
        beta_schedule: str = "linear",  # linear, exponential
        initial_beta: float = 1.0,
        final_beta: float = 0.0,
        decay_steps: int = 100,
    ):
        """
        Args:
            policy: Learnable policy (e.g., BehaviorCloning)
            expert_policy: Expert policy function
            beta_schedule: How to decay expert mixing
            initial_beta: Initial probability of using expert
            final_beta: Final probability of using expert
            decay_steps: Steps to decay beta
        """
        super().__init__()

        self.policy = policy
        self.expert_policy = expert_policy
        self.beta_schedule = beta_schedule
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.decay_steps = decay_steps

        self.step = 0
        self.dataset = []  # Aggregated dataset

    def get_beta(self) -> float:
        """Get current expert mixing probability."""
        if self.beta_schedule == "linear":
            progress = min(self.step / self.decay_steps, 1.0)
            return self.initial_beta + progress * (self.final_beta - self.initial_beta)
        elif self.beta_schedule == "exponential":
            decay_rate = (self.final_beta / self.initial_beta) ** (1 / self.decay_steps)
            return max(self.initial_beta * (decay_rate ** self.step), self.final_beta)
        else:
            return self.initial_beta

    def get_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action using DAgger strategy.

        With probability beta, use expert; otherwise use learned policy.
        Always query expert for labeling.

        Args:
            state: Current state

        Returns:
            Tuple of (executed_action, expert_action)
        """
        beta = self.get_beta()

        # Get expert action (for labeling)
        with torch.no_grad():
            expert_action = self.expert_policy(state)

        # Get policy action
        with torch.no_grad():
            policy_output = self.policy(state, deterministic=True)
            policy_action = policy_output['actions']

        # Choose which action to execute
        if torch.rand(1).item() < beta:
            executed_action = expert_action
        else:
            executed_action = policy_action

        return executed_action, expert_action

    def add_to_dataset(
        self,
        states: torch.Tensor,
        expert_actions: torch.Tensor,
    ):
        """Add transitions to aggregated dataset."""
        self.dataset.append({
            'states': states.cpu(),
            'actions': expert_actions.cpu(),
        })

    def train_step(self, batch_size: int = 64) -> float:
        """
        Train policy on aggregated dataset.

        Args:
            batch_size: Training batch size

        Returns:
            Loss value
        """
        if len(self.dataset) == 0:
            return 0.0

        # Sample from dataset
        all_states = torch.cat([d['states'] for d in self.dataset], dim=0)
        all_actions = torch.cat([d['actions'] for d in self.dataset], dim=0)

        indices = torch.randperm(len(all_states))[:batch_size]
        states = all_states[indices].to(next(self.policy.parameters()).device)
        actions = all_actions[indices].to(next(self.policy.parameters()).device)

        loss = self.policy.compute_loss(states, actions)

        self.step += 1
        return loss.item()


class Discriminator(nn.Module):
    """Discriminator for GAIL."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminate state-action pairs.

        Returns probability of being from expert.
        """
        x = torch.cat([states, actions], dim=-1)
        return torch.sigmoid(self.net(x))


class GAIL(nn.Module):
    """
    GAIL: Generative Adversarial Imitation Learning.

    Uses adversarial training to match expert distribution.

    Reference: https://arxiv.org/abs/1606.03476
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        discriminator_lr: float = 3e-4,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_size: Hidden layer size
            discriminator_lr: Discriminator learning rate
        """
        super().__init__()

        self.discriminator = Discriminator(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
        )

        self.optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
        )

    def compute_reward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GAIL reward.

        r = -log(1 - D(s, a))

        Args:
            states: States
            actions: Actions

        Returns:
            Reward values
        """
        with torch.no_grad():
            d = self.discriminator(states, actions)
            reward = -torch.log(1 - d + 1e-8)
        return reward.squeeze(-1)

    def update_discriminator(
        self,
        expert_states: torch.Tensor,
        expert_actions: torch.Tensor,
        policy_states: torch.Tensor,
        policy_actions: torch.Tensor,
    ) -> float:
        """
        Update discriminator.

        Args:
            expert_states: Expert states
            expert_actions: Expert actions
            policy_states: Policy states
            policy_actions: Policy actions

        Returns:
            Discriminator loss
        """
        # Expert should be classified as 1
        expert_d = self.discriminator(expert_states, expert_actions)
        expert_loss = -torch.log(expert_d + 1e-8).mean()

        # Policy should be classified as 0
        policy_d = self.discriminator(policy_states, policy_actions)
        policy_loss = -torch.log(1 - policy_d + 1e-8).mean()

        loss = expert_loss + policy_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


class InverseDynamics(nn.Module):
    """
    Inverse Dynamics Model.

    Predicts action given (s_t, s_{t+1}).
    Useful for learning from observation-only data.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_size: Hidden layer size
            num_layers: Number of layers
        """
        super().__init__()

        layers = [
            nn.Linear(state_dim * 2, hidden_size),
            nn.ReLU(),
        ]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(hidden_size, action_dim))
        layers.append(nn.Tanh())

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        next_state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict action.

        Args:
            state: Current state [batch, state_dim]
            next_state: Next state [batch, state_dim]

        Returns:
            Predicted action [batch, action_dim]
        """
        x = torch.cat([state, next_state], dim=-1)
        return self.net(x)

    def compute_loss(
        self,
        states: torch.Tensor,
        next_states: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute inverse dynamics loss.

        Args:
            states: Current states
            next_states: Next states
            actions: True actions

        Returns:
            MSE loss
        """
        predicted_actions = self.forward(states, next_states)
        return F.mse_loss(predicted_actions, actions)


class ForwardDynamics(nn.Module):
    """
    Forward Dynamics Model.

    Predicts s_{t+1} given (s_t, a_t).
    Used for model-based planning and curiosity.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        predict_residual: bool = True,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_size: Hidden layer size
            num_layers: Number of layers
            predict_residual: Predict state delta instead of absolute
        """
        super().__init__()

        self.predict_residual = predict_residual

        layers = [
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
        ]

        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            ])

        layers.append(nn.Linear(hidden_size, state_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict next state.

        Args:
            state: Current state [batch, state_dim]
            action: Action [batch, action_dim]

        Returns:
            Predicted next state [batch, state_dim]
        """
        x = torch.cat([state, action], dim=-1)
        output = self.net(x)

        if self.predict_residual:
            return state + output
        else:
            return output

    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
    ) -> torch.Tensor:
        """Compute forward dynamics loss."""
        predicted_next = self.forward(states, actions)
        return F.mse_loss(predicted_next, next_states)
