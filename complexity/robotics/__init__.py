"""
Robotics module for framework-complexity.

Provides ML components for robotics applications:
- Action tokenization and prediction
- State estimation and representation
- Trajectory modeling
- Behavior cloning / Imitation learning
- Robot transformers (RT-1, RT-2 style)
- Multi-task robot learning

Developed by INL for embodied AI research.

Usage:
    from complexity.robotics import (
        RobotTransformer,
        ActionTokenizer,
        StateEncoder,
        TrajectoryTransformer,
    )

    # Robot transformer for manipulation
    robot = RobotTransformer(
        state_dim=512,
        action_dim=7,  # 6-DOF + gripper
        num_layers=12,
    )

    # Predict actions from observations
    actions = robot(images, language_instruction)
"""

from .action_space import (
    ActionTokenizer,
    ContinuousActionSpace,
    DiscreteActionSpace,
    ActionConfig,
    ActionHead,
    DiffusionActionHead,
)

from .state_encoder import (
    StateEncoder,
    StateConfig,
    ProprioceptionEncoder,
    MultimodalStateEncoder,
    TemporalStateEncoder,
)

from .trajectory import (
    TrajectoryTransformer,
    TrajectoryConfig,
    TrajectoryEncoder,
    TrajectoryDecoder,
    GoalConditionedPolicy,
)

from .robot_transformer import (
    RobotTransformer,
    RobotConfig,
    RT1Model,
    RT2Model,
    OctoModel,
)

from .imitation import (
    BehaviorCloning,
    BCConfig,
    DAgger,
    GAIL,
    InverseDynamics,
)

from .world_model import (
    WorldModel,
    WorldModelConfig,
    DreamerV3,
    LatentDynamics,
    RewardPredictor,
)

__all__ = [
    # Action space
    "ActionTokenizer",
    "ContinuousActionSpace",
    "DiscreteActionSpace",
    "ActionConfig",
    "ActionHead",
    "DiffusionActionHead",
    # State encoder
    "StateEncoder",
    "StateConfig",
    "ProprioceptionEncoder",
    "MultimodalStateEncoder",
    "TemporalStateEncoder",
    # Trajectory
    "TrajectoryTransformer",
    "TrajectoryConfig",
    "TrajectoryEncoder",
    "TrajectoryDecoder",
    "GoalConditionedPolicy",
    # Robot transformer
    "RobotTransformer",
    "RobotConfig",
    "RT1Model",
    "RT2Model",
    "OctoModel",
    # Imitation learning
    "BehaviorCloning",
    "BCConfig",
    "DAgger",
    "GAIL",
    "InverseDynamics",
    # World model
    "WorldModel",
    "WorldModelConfig",
    "DreamerV3",
    "LatentDynamics",
    "RewardPredictor",
]
