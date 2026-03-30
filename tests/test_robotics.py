"""Tests for complexity.robotics module."""

import pytest
import torch


class TestActionSpace:
    """Test action space components."""

    def test_action_config(self):
        """Test ActionConfig."""
        from complexity.robotics import ActionConfig

        config = ActionConfig(
            action_dim=7,
            num_bins=256,
            action_range=(-1.0, 1.0),
        )

        assert config.action_dim == 7
        assert config.num_bins == 256

    def test_action_tokenizer(self):
        """Test ActionTokenizer for discretizing continuous actions."""
        from complexity.robotics import ActionTokenizer, ActionConfig

        config = ActionConfig(
            action_dim=7,
            num_bins=256,
            action_range=(-1.0, 1.0),
        )
        tokenizer = ActionTokenizer(config)

        # Tokenize continuous actions
        actions = torch.randn(2, 10, 7)  # [batch, seq, action_dim]
        tokens = tokenizer.encode(actions)

        assert tokens.shape == (2, 10, 7)
        assert tokens.dtype == torch.long
        assert tokens.min() >= 0
        assert tokens.max() < 256

        # Decode back
        decoded = tokenizer.decode(tokens)
        assert decoded.shape == actions.shape

    def test_continuous_action_space(self):
        """Test ContinuousActionSpace."""
        from complexity.robotics import ContinuousActionSpace

        action_space = ContinuousActionSpace(
            dim=7,
            low=-1.0,
            high=1.0,
        )

        assert action_space.dim == 7

        # Sample
        samples = action_space.sample(batch_size=4)
        assert samples.shape == (4, 7)
        assert samples.min() >= -1.0
        assert samples.max() <= 1.0

    def test_discrete_action_space(self):
        """Test DiscreteActionSpace."""
        from complexity.robotics import DiscreteActionSpace

        action_space = DiscreteActionSpace(num_actions=10)

        assert action_space.num_actions == 10

        # Sample
        samples = action_space.sample(batch_size=4)
        assert samples.shape == (4,)
        assert samples.min() >= 0
        assert samples.max() < 10

    def test_action_head(self):
        """Test ActionHead for predicting actions."""
        from complexity.robotics import ActionHead, ActionConfig

        config = ActionConfig(
            action_dim=7,
            num_bins=256,
        )
        head = ActionHead(hidden_size=256, config=config)

        # Forward
        hidden = torch.randn(2, 10, 256)
        logits = head(hidden)

        assert logits.shape == (2, 10, 7, 256)


class TestStateEncoder:
    """Test state encoder components."""

    def test_state_config(self):
        """Test StateConfig."""
        from complexity.robotics import StateConfig

        config = StateConfig(
            proprio_dim=14,
            hidden_size=256,
            num_cameras=2,
        )

        assert config.proprio_dim == 14
        assert config.hidden_size == 256

    def test_proprioception_encoder(self):
        """Test ProprioceptionEncoder."""
        from complexity.robotics import ProprioceptionEncoder, StateConfig

        config = StateConfig(
            proprio_dim=14,
            hidden_size=256,
        )
        encoder = ProprioceptionEncoder(config)

        # Forward
        proprio = torch.randn(2, 10, 14)  # [batch, seq, proprio_dim]
        encoded = encoder(proprio)

        assert encoded.shape == (2, 10, 256)

    def test_state_encoder(self):
        """Test StateEncoder."""
        from complexity.robotics import StateEncoder, StateConfig

        config = StateConfig(
            proprio_dim=14,
            hidden_size=256,
        )
        encoder = StateEncoder(config)

        # Forward with proprio only
        proprio = torch.randn(2, 10, 14)
        encoded = encoder(proprio=proprio)

        assert encoded.shape == (2, 10, 256)

    def test_temporal_state_encoder(self):
        """Test TemporalStateEncoder."""
        from complexity.robotics import TemporalStateEncoder, StateConfig

        config = StateConfig(
            proprio_dim=14,
            hidden_size=256,
            history_length=5,
        )
        encoder = TemporalStateEncoder(config)

        # Forward
        proprio = torch.randn(2, 10, 14)
        encoded = encoder(proprio)

        assert encoded.shape[0] == 2
        assert encoded.shape[-1] == 256


class TestTrajectory:
    """Test trajectory components."""

    def test_trajectory_config(self):
        """Test TrajectoryConfig."""
        from complexity.robotics import TrajectoryConfig

        config = TrajectoryConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
            num_layers=6,
        )

        assert config.state_dim == 256
        assert config.action_dim == 7

    def test_trajectory_encoder(self):
        """Test TrajectoryEncoder."""
        from complexity.robotics import TrajectoryEncoder, TrajectoryConfig

        config = TrajectoryConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
        )
        encoder = TrajectoryEncoder(config)

        # Forward
        states = torch.randn(2, 20, 256)
        actions = torch.randn(2, 20, 7)
        encoded = encoder(states, actions)

        assert encoded.shape == (2, 20, 512)

    def test_trajectory_transformer(self):
        """Test TrajectoryTransformer."""
        from complexity.robotics import TrajectoryTransformer, TrajectoryConfig

        config = TrajectoryConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
            num_layers=4,
            num_heads=8,
        )
        model = TrajectoryTransformer(config)

        # Forward
        states = torch.randn(2, 20, 256)
        actions = torch.randn(2, 19, 7)  # One less action
        predicted = model(states, actions)

        assert "actions" in predicted
        assert predicted["actions"].shape == (2, 20, 7)

    def test_goal_conditioned_policy(self):
        """Test GoalConditionedPolicy."""
        from complexity.robotics import GoalConditionedPolicy, TrajectoryConfig

        config = TrajectoryConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
        )
        policy = GoalConditionedPolicy(config)

        # Forward
        state = torch.randn(2, 256)
        goal = torch.randn(2, 256)
        action = policy(state, goal)

        assert action.shape == (2, 7)


class TestRobotTransformer:
    """Test robot transformer models."""

    def test_robot_config(self):
        """Test RobotConfig."""
        from complexity.robotics import RobotConfig

        config = RobotConfig(
            state_dim=512,
            action_dim=7,
            hidden_size=768,
            num_layers=12,
        )

        assert config.state_dim == 512
        assert config.action_dim == 7

    def test_robot_transformer(self):
        """Test RobotTransformer."""
        from complexity.robotics import RobotTransformer, RobotConfig

        config = RobotConfig(
            state_dim=512,
            action_dim=7,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
        )
        model = RobotTransformer(config)

        # Forward with state only
        states = torch.randn(2, 10, 512)
        output = model(states)

        assert "actions" in output
        assert output["actions"].shape[-1] == 7

    def test_rt1_model(self):
        """Test RT1Model (simplified)."""
        from complexity.robotics import RT1Model, RobotConfig

        config = RobotConfig(
            state_dim=512,
            action_dim=7,
            hidden_size=256,
            num_layers=4,
        )
        model = RT1Model(config)

        # Forward
        states = torch.randn(2, 6, 512)  # 6 frames
        output = model(states)

        assert "actions" in output

    def test_rt2_model(self):
        """Test RT2Model (simplified)."""
        from complexity.robotics import RT2Model, RobotConfig

        config = RobotConfig(
            state_dim=512,
            action_dim=7,
            hidden_size=256,
            num_layers=4,
            vocab_size=1000,
        )
        model = RT2Model(config)

        # Forward
        states = torch.randn(2, 6, 512)
        output = model(states)

        assert "actions" in output or "logits" in output


class TestImitationLearning:
    """Test imitation learning components."""

    def test_bc_config(self):
        """Test BCConfig."""
        from complexity.robotics import BCConfig

        config = BCConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
        )

        assert config.state_dim == 256
        assert config.action_dim == 7

    def test_behavior_cloning(self):
        """Test BehaviorCloning."""
        from complexity.robotics import BehaviorCloning, BCConfig

        config = BCConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
            num_layers=3,
        )
        bc = BehaviorCloning(config)

        # Forward
        states = torch.randn(2, 256)
        actions = bc(states)

        assert actions.shape == (2, 7)

    def test_behavior_cloning_loss(self):
        """Test BehaviorCloning loss computation."""
        from complexity.robotics import BehaviorCloning, BCConfig

        config = BCConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
        )
        bc = BehaviorCloning(config)

        # Compute loss
        states = torch.randn(4, 256)
        target_actions = torch.randn(4, 7)

        loss = bc.compute_loss(states, target_actions)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_inverse_dynamics(self):
        """Test InverseDynamics model."""
        from complexity.robotics import InverseDynamics, BCConfig

        config = BCConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
        )
        model = InverseDynamics(config)

        # Forward: predict action from state pair
        state = torch.randn(2, 256)
        next_state = torch.randn(2, 256)
        action = model(state, next_state)

        assert action.shape == (2, 7)


class TestWorldModel:
    """Test world model components."""

    def test_world_model_config(self):
        """Test WorldModelConfig."""
        from complexity.robotics import WorldModelConfig

        config = WorldModelConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
            latent_dim=64,
        )

        assert config.state_dim == 256
        assert config.latent_dim == 64

    def test_latent_dynamics(self):
        """Test LatentDynamics."""
        from complexity.robotics import LatentDynamics, WorldModelConfig

        config = WorldModelConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=512,
            latent_dim=64,
        )
        dynamics = LatentDynamics(config)

        # Forward
        latent = torch.randn(2, 64)
        action = torch.randn(2, 7)
        next_latent = dynamics(latent, action)

        assert next_latent.shape == (2, 64)

    def test_reward_predictor(self):
        """Test RewardPredictor."""
        from complexity.robotics import RewardPredictor, WorldModelConfig

        config = WorldModelConfig(
            state_dim=256,
            action_dim=7,
            latent_dim=64,
        )
        predictor = RewardPredictor(config)

        # Forward
        latent = torch.randn(2, 64)
        reward = predictor(latent)

        assert reward.shape == (2, 1) or reward.shape == (2,)

    def test_world_model(self):
        """Test WorldModel."""
        from complexity.robotics import WorldModel, WorldModelConfig

        config = WorldModelConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=256,
            latent_dim=64,
        )
        model = WorldModel(config)

        # Forward
        states = torch.randn(2, 10, 256)
        actions = torch.randn(2, 10, 7)
        output = model(states, actions)

        assert "latents" in output or "predictions" in output

    @pytest.mark.skip(reason="DreamerV3 is complex - skip for basic tests")
    def test_dreamer_v3(self):
        """Test DreamerV3 world model."""
        from complexity.robotics import DreamerV3, WorldModelConfig

        config = WorldModelConfig(
            state_dim=256,
            action_dim=7,
            hidden_size=256,
            latent_dim=64,
        )
        model = DreamerV3(config)

        # Forward
        states = torch.randn(2, 10, 256)
        actions = torch.randn(2, 10, 7)
        output = model(states, actions)

        assert output is not None
