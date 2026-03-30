"""
Behavior Cloning Example - Train a robot policy from demonstrations.

This example shows how to:
1. Create a BehaviorCloning policy
2. Train on demonstration data
3. Evaluate the policy

Usage:
    python behavior_cloning.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from complexity.robotics import (
    BehaviorCloning,
    BCConfig,
    StateEncoder,
    StateConfig,
)


def generate_demo_data(num_samples=1000, state_dim=14, action_dim=7):
    """Generate synthetic demonstration data."""
    # Simple linear policy: action = W @ state + noise
    W = torch.randn(action_dim, state_dim) * 0.1

    states = torch.randn(num_samples, state_dim)
    actions = states @ W.T + torch.randn(num_samples, action_dim) * 0.01
    actions = torch.tanh(actions)  # Clip to [-1, 1]

    return states, actions


def train_bc():
    """Train a behavior cloning policy."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Config
    state_dim = 14  # e.g., joint positions + velocities
    action_dim = 7  # e.g., 6-DOF + gripper
    hidden_size = 256

    # Generate demo data
    print("Generating demonstration data...")
    states, actions = generate_demo_data(
        num_samples=10000,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    # Split train/val
    train_states, val_states = states[:8000], states[8000:]
    train_actions, val_actions = actions[:8000], actions[8000:]

    # DataLoader
    train_dataset = TensorDataset(train_states, train_actions)
    val_dataset = TensorDataset(val_states, val_actions)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64)

    # Create BC policy
    print("Creating BehaviorCloning policy...")
    config = BCConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        num_layers=3,
    )
    policy = BehaviorCloning(config).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # Training loop
    print("\nTraining...")
    print("-" * 50)

    for epoch in range(100):
        # Train
        policy.train()
        train_loss = 0.0
        for batch_states, batch_actions in train_loader:
            batch_states = batch_states.to(device)
            batch_actions = batch_actions.to(device)

            optimizer.zero_grad()
            loss = policy.compute_loss(batch_states, batch_actions)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validate
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(device)
                batch_actions = batch_actions.to(device)

                loss = policy.compute_loss(batch_states, batch_actions)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    print("-" * 50)
    print("Training complete!")

    # Test inference
    print("\nTesting inference...")
    policy.eval()
    with torch.no_grad():
        test_state = torch.randn(1, state_dim).to(device)
        predicted_action = policy(test_state)
        print(f"Input state shape: {test_state.shape}")
        print(f"Predicted action shape: {predicted_action.shape}")
        print(f"Predicted action: {predicted_action.squeeze().cpu().numpy()}")

    return policy


if __name__ == "__main__":
    train_bc()
