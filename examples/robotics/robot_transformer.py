"""
Robot Transformer Example - RT-1/RT-2 style model for robot control.

This example shows how to:
1. Create a RobotTransformer model
2. Process sequences of observations
3. Predict actions

Usage:
    python robot_transformer.py
"""

import torch
import torch.nn as nn

from complexity.robotics import (
    RobotTransformer,
    RobotConfig,
    RT1Model,
    ActionTokenizer,
    ActionConfig,
    StateEncoder,
    StateConfig,
)


def demo_robot_transformer():
    """Demonstrate RobotTransformer usage."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    state_dim = 512  # Encoded observation dimension
    action_dim = 7   # 6-DOF + gripper
    seq_len = 6      # History length (RT-1 uses 6 frames)

    print("\n" + "=" * 60)
    print("ROBOT TRANSFORMER DEMO")
    print("=" * 60)

    # Create RobotTransformer
    print("\n1. Creating RobotTransformer...")
    config = RobotConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        num_action_bins=256,  # Discretize actions into 256 bins
    )
    model = RobotTransformer(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # Simulate observation sequence
    print("\n2. Simulating observation sequence...")
    # In practice, this would come from a vision encoder
    observations = torch.randn(2, seq_len, state_dim).to(device)  # [batch, seq, state_dim]
    print(f"   Observation shape: {observations.shape}")

    # Predict actions
    print("\n3. Predicting actions...")
    model.eval()
    with torch.no_grad():
        output = model(observations)

    if "actions" in output:
        actions = output["actions"]
        print(f"   Predicted actions shape: {actions.shape}")
        print(f"   Action sample: {actions[0, -1].cpu().numpy()}")  # Last timestep

    if "action_logits" in output:
        logits = output["action_logits"]
        print(f"   Action logits shape: {logits.shape}")

    return model


def demo_rt1():
    """Demonstrate RT-1 style model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("RT-1 MODEL DEMO")
    print("=" * 60)

    # RT-1 config
    config = RobotConfig(
        state_dim=512,
        action_dim=7,
        hidden_size=256,
        num_layers=8,
        num_heads=8,
        num_action_bins=256,
    )

    print("\n1. Creating RT1Model...")
    model = RT1Model(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")

    # RT-1 uses 6 frames of history
    print("\n2. Processing 6-frame history...")
    frames = torch.randn(1, 6, 512).to(device)

    model.eval()
    with torch.no_grad():
        output = model(frames)

    print(f"   Output keys: {list(output.keys())}")

    return model


def demo_action_tokenizer():
    """Demonstrate action tokenization (RT-1 style)."""
    print("\n" + "=" * 60)
    print("ACTION TOKENIZER DEMO")
    print("=" * 60)

    # Action tokenizer config
    config = ActionConfig(
        action_dim=7,
        num_bins=256,
        action_range=(-1.0, 1.0),
    )

    print("\n1. Creating ActionTokenizer...")
    tokenizer = ActionTokenizer(config)

    # Continuous actions
    print("\n2. Tokenizing continuous actions...")
    continuous_actions = torch.tensor([
        [0.5, -0.3, 0.8, 0.0, 0.2, -0.1, 1.0],  # gripper closed
        [-0.2, 0.4, -0.6, 0.1, -0.5, 0.3, 0.0],  # gripper open
    ])
    print(f"   Continuous actions:\n   {continuous_actions}")

    tokens = tokenizer.encode(continuous_actions)
    print(f"   Tokenized: {tokens}")

    # Decode back
    print("\n3. Decoding tokens back to continuous...")
    decoded = tokenizer.decode(tokens)
    print(f"   Decoded actions:\n   {decoded}")

    # Check reconstruction error
    error = (continuous_actions - decoded).abs().max()
    print(f"   Max reconstruction error: {error:.4f}")

    return tokenizer


def demo_full_pipeline():
    """Demonstrate full robot control pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("FULL ROBOT CONTROL PIPELINE")
    print("=" * 60)

    # 1. State encoder (proprioception)
    print("\n1. State Encoder...")
    state_config = StateConfig(
        proprio_dim=14,  # joint pos + vel
        hidden_size=256,
    )
    state_encoder = StateEncoder(state_config).to(device)

    # 2. Robot transformer
    print("2. Robot Transformer...")
    robot_config = RobotConfig(
        state_dim=256,  # Output of state encoder
        action_dim=7,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
    )
    robot = RobotTransformer(robot_config).to(device)

    # 3. Action tokenizer
    print("3. Action Tokenizer...")
    action_config = ActionConfig(action_dim=7, num_bins=256)
    action_tokenizer = ActionTokenizer(action_config)

    # Simulate episode
    print("\n4. Simulating episode...")
    batch_size = 1
    seq_len = 6

    # Raw proprioception
    proprio = torch.randn(batch_size, seq_len, 14).to(device)
    print(f"   Proprioception shape: {proprio.shape}")

    # Encode states
    encoded_states = state_encoder(proprio=proprio)
    print(f"   Encoded states shape: {encoded_states.shape}")

    # Predict actions
    with torch.no_grad():
        output = robot(encoded_states)

    if "actions" in output:
        actions = output["actions"]
        print(f"   Predicted actions shape: {actions.shape}")
        print(f"   Action to execute: {actions[0, -1].cpu().numpy()}")

    print("\nPipeline complete!")


if __name__ == "__main__":
    demo_robot_transformer()
    demo_rt1()
    demo_action_tokenizer()
    demo_full_pipeline()
