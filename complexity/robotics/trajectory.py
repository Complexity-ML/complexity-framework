"""
Trajectory modeling for robotics.

Provides:
- Trajectory transformer (sequence modeling)
- Goal-conditioned policies
- Trajectory encoding/decoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import math


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory modeling."""
    state_dim: int = 32                # State dimension
    action_dim: int = 7                # Action dimension
    hidden_size: int = 768
    num_layers: int = 6
    num_heads: int = 12
    max_trajectory_length: int = 1000
    dropout: float = 0.1
    use_returns: bool = True           # Condition on returns (like Decision Transformer)
    use_goals: bool = False            # Goal-conditioned


class TrajectoryEncoder(nn.Module):
    """
    Encode trajectories (state-action sequences).

    Projects (s, a, r, g) tuples into hidden space.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        use_returns: bool = True,
        use_goals: bool = False,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_size: Hidden dimension
            use_returns: Include return-to-go
            use_goals: Include goal conditioning
        """
        super().__init__()

        self.use_returns = use_returns
        self.use_goals = use_goals

        # Embeddings for each component
        self.state_embed = nn.Linear(state_dim, hidden_size)
        self.action_embed = nn.Linear(action_dim, hidden_size)

        if use_returns:
            self.return_embed = nn.Linear(1, hidden_size)

        if use_goals:
            self.goal_embed = nn.Linear(state_dim, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        returns_to_go: Optional[torch.Tensor] = None,
        goals: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Encode trajectory components.

        Args:
            states: States [batch, seq, state_dim]
            actions: Actions [batch, seq, action_dim]
            returns_to_go: Returns [batch, seq, 1]
            goals: Goals [batch, goal_dim] or [batch, seq, goal_dim]

        Returns:
            Tuple of (embeddings [batch, seq * num_tokens, hidden], tokens_per_step)
        """
        batch_size, seq_len = states.shape[:2]
        tokens = []

        # State embeddings
        state_emb = self.state_embed(states)
        tokens.append(state_emb)

        # Return embeddings
        if self.use_returns and returns_to_go is not None:
            return_emb = self.return_embed(returns_to_go)
            tokens.append(return_emb)

        # Action embeddings
        if actions is not None:
            action_emb = self.action_embed(actions)
            tokens.append(action_emb)

        # Goal embeddings
        if self.use_goals and goals is not None:
            if goals.dim() == 2:
                goals = goals.unsqueeze(1).expand(-1, seq_len, -1)
            goal_emb = self.goal_embed(goals)
            tokens.append(goal_emb)

        tokens_per_step = len(tokens)

        # Interleave: [R1, S1, A1, R2, S2, A2, ...]
        # Stack and reshape
        stacked = torch.stack(tokens, dim=2)  # [batch, seq, num_tokens, hidden]
        interleaved = stacked.view(batch_size, seq_len * tokens_per_step, -1)

        return self.norm(interleaved), tokens_per_step


class TrajectoryDecoder(nn.Module):
    """
    Decode trajectory from hidden states.

    Predicts next actions (and optionally states/returns).
    """

    def __init__(
        self,
        hidden_size: int,
        state_dim: int,
        action_dim: int,
        predict_state: bool = False,
        predict_return: bool = False,
    ):
        """
        Args:
            hidden_size: Hidden dimension
            state_dim: State dimension
            action_dim: Action dimension
            predict_state: Also predict next state
            predict_return: Also predict returns
        """
        super().__init__()

        self.predict_state = predict_state
        self.predict_return = predict_return

        self.action_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

        if predict_state:
            self.state_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, state_dim),
            )

        if predict_return:
            self.return_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, 1),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        tokens_per_step: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode actions (and optionally states/returns).

        Args:
            hidden_states: Hidden states [batch, seq, hidden]
            tokens_per_step: Number of tokens per timestep

        Returns:
            Dictionary with predictions
        """
        # Get action positions (every tokens_per_step)
        batch_size, seq_len, _ = hidden_states.shape

        # For Decision Transformer style: predict action from state position
        # Assuming order is [R, S, A] per step, action comes from S position
        action_positions = hidden_states

        outputs = {
            'actions': self.action_head(action_positions),
        }

        if self.predict_state:
            outputs['states'] = self.state_head(action_positions)

        if self.predict_return:
            outputs['returns'] = self.return_head(action_positions)

        return outputs


class TrajectoryTransformer(nn.Module):
    """
    Transformer for trajectory modeling.

    Based on Decision Transformer architecture.
    Models trajectories as sequences: (R, s, a, R, s, a, ...)

    Reference: Decision Transformer (https://arxiv.org/abs/2106.01345)
    """

    def __init__(self, config: TrajectoryConfig):
        """
        Args:
            config: Trajectory configuration
        """
        super().__init__()

        self.config = config

        # Encoder
        self.encoder = TrajectoryEncoder(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_size=config.hidden_size,
            use_returns=config.use_returns,
            use_goals=config.use_goals,
        )

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, config.max_trajectory_length * 3, config.hidden_size) * 0.02
        )

        # Transformer
        self.dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
        )

        # Decoder
        self.decoder = TrajectoryDecoder(
            hidden_size=config.hidden_size,
            state_dim=config.state_dim,
            action_dim=config.action_dim,
        )

        # Causal mask
        self.register_buffer(
            'causal_mask',
            self._create_causal_mask(config.max_trajectory_length * 3)
        )

    def _create_causal_mask(self, size: int) -> torch.Tensor:
        """Create causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()

    def forward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        returns_to_go: Optional[torch.Tensor] = None,
        goals: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            states: States [batch, seq, state_dim]
            actions: Actions [batch, seq, action_dim]
            returns_to_go: Returns [batch, seq, 1]
            goals: Goals [batch, goal_dim]
            attention_mask: Attention mask

        Returns:
            Dictionary with action predictions
        """
        batch_size, seq_len = states.shape[:2]

        # Encode trajectory
        embeddings, tokens_per_step = self.encoder(
            states, actions, returns_to_go, goals
        )

        total_len = embeddings.size(1)

        # Add positional encoding
        embeddings = embeddings + self.pos_embedding[:, :total_len, :]
        embeddings = self.dropout(embeddings)

        # Get causal mask
        causal_mask = self.causal_mask[:total_len, :total_len]

        # Transformer
        hidden_states = self.transformer(
            embeddings,
            mask=causal_mask,
            src_key_padding_mask=attention_mask,
        )

        # Decode
        outputs = self.decoder(hidden_states, tokens_per_step)

        return outputs

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        goals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Get action for current state (inference).

        Args:
            states: State history [batch, seq, state_dim]
            actions: Action history [batch, seq-1, action_dim]
            returns_to_go: Return history [batch, seq, 1]
            goals: Goal state [batch, goal_dim]

        Returns:
            Next action [batch, action_dim]
        """
        # Pad actions to match states
        if actions.size(1) < states.size(1):
            pad = torch.zeros(
                actions.size(0), 1, actions.size(2),
                device=actions.device
            )
            actions = torch.cat([actions, pad], dim=1)

        outputs = self.forward(states, actions, returns_to_go, goals)

        # Return last action prediction
        return outputs['actions'][:, -1, :]


class GoalConditionedPolicy(nn.Module):
    """
    Goal-conditioned policy network.

    Predicts actions given current state and goal state.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        goal_dim: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            goal_dim: Goal dimension
            hidden_size: Hidden layer size
            num_layers: Number of layers
            dropout: Dropout rate
        """
        super().__init__()

        self.state_dim = state_dim
        self.goal_dim = goal_dim

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Goal encoder
        self.goal_encoder = nn.Sequential(
            nn.Linear(goal_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )

        # Policy network
        layers = []
        input_dim = hidden_size * 2  # state + goal

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_size

        layers.append(nn.Linear(hidden_size, action_dim))
        layers.append(nn.Tanh())

        self.policy = nn.Sequential(*layers)

    def forward(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict action.

        Args:
            state: Current state [batch, state_dim]
            goal: Goal state [batch, goal_dim]

        Returns:
            Action [batch, action_dim]
        """
        state_enc = self.state_encoder(state)
        goal_enc = self.goal_encoder(goal)

        combined = torch.cat([state_enc, goal_enc], dim=-1)

        return self.policy(combined)

    def compute_goal_distance(
        self,
        state: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distance to goal.

        Args:
            state: Current state
            goal: Goal state

        Returns:
            Distance scalar
        """
        return torch.norm(state - goal, dim=-1)


class HierarchicalPolicy(nn.Module):
    """
    Hierarchical policy with high-level and low-level controllers.

    High-level: Sets subgoals
    Low-level: Achieves subgoals
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        subgoal_dim: int,
        hidden_size: int = 256,
        subgoal_horizon: int = 10,
    ):
        """
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            subgoal_dim: Subgoal dimension
            hidden_size: Hidden size
            subgoal_horizon: Steps between subgoals
        """
        super().__init__()

        self.subgoal_horizon = subgoal_horizon
        self.step_count = 0

        # High-level policy (state -> subgoal)
        self.high_level = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, subgoal_dim),
        )

        # Low-level policy (state + subgoal -> action)
        self.low_level = GoalConditionedPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=subgoal_dim,
            hidden_size=hidden_size,
        )

        self.current_subgoal = None

    def forward(
        self,
        state: torch.Tensor,
        final_goal: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get action from hierarchical policy.

        Args:
            state: Current state
            final_goal: Final goal (optional, for conditioning high-level)

        Returns:
            Tuple of (action, current_subgoal)
        """
        # Update subgoal if needed
        if self.step_count % self.subgoal_horizon == 0 or self.current_subgoal is None:
            if final_goal is not None:
                high_input = torch.cat([state, final_goal], dim=-1)
            else:
                high_input = state

            self.current_subgoal = self.high_level(state)

        self.step_count += 1

        # Get low-level action
        action = self.low_level(state, self.current_subgoal)

        return action, self.current_subgoal

    def reset(self):
        """Reset policy state."""
        self.current_subgoal = None
        self.step_count = 0
