"""
World Model Example - Learn environment dynamics for planning.

This example shows how to:
1. Create a WorldModel for learning dynamics
2. Train on state-action-next_state transitions
3. Use for imagination/planning

Usage:
    python world_model.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from complexity.robotics import (
    WorldModel,
    WorldModelConfig,
    LatentDynamics,
    RewardPredictor,
)


def generate_transition_data(num_samples=10000, state_dim=32, action_dim=7):
    """Generate synthetic transition data."""
    # Simple dynamics: next_state = state + action + noise
    states = torch.randn(num_samples, state_dim)
    actions = torch.randn(num_samples, action_dim)

    # Linear dynamics with some nonlinearity
    A = torch.randn(state_dim, state_dim) * 0.1
    B = torch.randn(state_dim, action_dim) * 0.1

    next_states = states @ A.T + actions @ B.T + torch.randn(num_samples, state_dim) * 0.01
    next_states = torch.tanh(next_states)  # Nonlinearity

    # Simple reward: distance to origin
    rewards = -torch.norm(next_states, dim=-1, keepdim=True)

    return states, actions, next_states, rewards


def train_world_model():
    """Train a world model on transition data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    state_dim = 32
    action_dim = 7
    latent_dim = 64
    hidden_size = 256

    print("\n" + "=" * 60)
    print("WORLD MODEL TRAINING")
    print("=" * 60)

    # Generate data
    print("\n1. Generating transition data...")
    states, actions, next_states, rewards = generate_transition_data(
        num_samples=50000,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Split
    train_size = 40000
    train_data = TensorDataset(
        states[:train_size],
        actions[:train_size],
        next_states[:train_size],
        rewards[:train_size],
    )
    val_data = TensorDataset(
        states[train_size:],
        actions[train_size:],
        next_states[train_size:],
        rewards[train_size:],
    )

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=256)

    print(f"   Train samples: {train_size}")
    print(f"   Val samples: {len(states) - train_size}")

    # Create world model
    print("\n2. Creating WorldModel...")
    config = WorldModelConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        latent_dim=latent_dim,
    )
    model = WorldModel(config).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training
    print("\n3. Training...")
    print("-" * 50)

    for epoch in range(50):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            s, a, ns, r = [x.to(device) for x in batch]

            optimizer.zero_grad()

            # Forward: predict next state and reward
            output = model.forward_single(s, a)

            # Loss: reconstruction + reward prediction
            loss = 0.0
            if "next_state_pred" in output:
                loss += nn.functional.mse_loss(output["next_state_pred"], ns)
            if "reward_pred" in output:
                loss += nn.functional.mse_loss(output["reward_pred"], r)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3d} | Loss: {train_loss:.4f}")

    print("-" * 50)
    print("Training complete!")

    return model


def demo_imagination():
    """Demonstrate imagination/rollout with world model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("IMAGINATION ROLLOUT DEMO")
    print("=" * 60)

    # Create world model
    config = WorldModelConfig(
        state_dim=32,
        action_dim=7,
        hidden_size=256,
        latent_dim=64,
    )
    model = WorldModel(config).to(device)
    model.eval()

    # Initial state
    print("\n1. Starting from initial state...")
    state = torch.randn(1, 32).to(device)
    print(f"   Initial state norm: {state.norm().item():.4f}")

    # Imagine trajectory
    print("\n2. Imagining 10-step trajectory...")
    trajectory = [state]
    rewards = []

    with torch.no_grad():
        current_state = state
        for t in range(10):
            # Random action (could be from policy)
            action = torch.randn(1, 7).to(device) * 0.5

            # Predict next state
            output = model.forward_single(current_state, action)

            if "next_state_pred" in output:
                next_state = output["next_state_pred"]
            else:
                # Fallback
                next_state = current_state + action[:, :32] * 0.1 if current_state.shape[-1] >= 7 else current_state

            if "reward_pred" in output:
                reward = output["reward_pred"].item()
            else:
                reward = 0.0

            trajectory.append(next_state)
            rewards.append(reward)
            current_state = next_state

    print(f"   Trajectory length: {len(trajectory)}")
    print(f"   Total imagined reward: {sum(rewards):.4f}")

    # Show state evolution
    print("\n3. State evolution:")
    for i, s in enumerate(trajectory):
        print(f"   t={i}: norm={s.norm().item():.4f}")

    return model


def demo_latent_dynamics():
    """Demonstrate latent dynamics model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n" + "=" * 60)
    print("LATENT DYNAMICS DEMO")
    print("=" * 60)

    # Create latent dynamics
    config = WorldModelConfig(
        state_dim=32,
        action_dim=7,
        latent_dim=64,
        hidden_size=256,
    )
    dynamics = LatentDynamics(config).to(device)

    print("\n1. Creating latent dynamics model...")
    num_params = sum(p.numel() for p in dynamics.parameters())
    print(f"   Parameters: {num_params:,}")

    # Test forward
    print("\n2. Testing forward pass...")
    latent = torch.randn(2, 64).to(device)
    action = torch.randn(2, 7).to(device)

    next_latent = dynamics(latent, action)
    print(f"   Input latent shape: {latent.shape}")
    print(f"   Action shape: {action.shape}")
    print(f"   Output latent shape: {next_latent.shape}")

    # Multiple steps
    print("\n3. Multi-step rollout...")
    current = latent[0:1]
    for i in range(5):
        action = torch.randn(1, 7).to(device)
        current = dynamics(current, action)
        print(f"   Step {i+1}: latent norm = {current.norm().item():.4f}")

    return dynamics


if __name__ == "__main__":
    # Train world model
    model = train_world_model()

    # Demo imagination
    demo_imagination()

    # Demo latent dynamics
    demo_latent_dynamics()
