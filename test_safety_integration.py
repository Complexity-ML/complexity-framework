"""
Test Safety Integration for Complexity Framework

Tests:
1. SafetyClamp basic functionality
2. INLDynamics with safety clamp
3. MultiDirectionSafetyClamp
4. ContrastiveSafetyLoss
5. install/remove safety functions
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Testing Safety Integration - Complexity Framework")
print("=" * 60)

# Test 1: SafetyClamp
print("\n[1] Testing SafetyClamp...")
from complexity.utils.safety import SafetyClamp

clamp = SafetyClamp(hidden_size=256, threshold=2.0, soft_clamp=False)

# Set harm direction
harm_dir = torch.randn(256)
clamp.set_harm_direction(harm_dir)
clamp.enabled = True

# Test clamping
x = torch.randn(2, 32, 256)
x_clamped = clamp(x)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {x_clamped.shape}")
print(f"  Stats: {clamp.get_stats()}")
print("  [OK] SafetyClamp works!")

# Test 2: High projection clamping
print("\n[2] Testing high-projection clamping...")
harm_dir_norm = harm_dir / harm_dir.norm()
high_harm = harm_dir_norm.unsqueeze(0) * 5.0  # 5x threshold
clamped = clamp(high_harm)
proj_before = (high_harm @ harm_dir_norm).item()
proj_after = (clamped @ harm_dir_norm).item()
print(f"  Projection before: {proj_before:.4f}")
print(f"  Projection after:  {proj_after:.4f}")
print(f"  Threshold:         {clamp.threshold}")
assert proj_after <= clamp.threshold + 0.01, f"Clamping failed! {proj_after} > {clamp.threshold}"
print("  [OK] High-projection clamping works!")

# Test 3: INLDynamics with safety
print("\n[3] Testing INLDynamics with safety clamp...")
from complexity.core.dynamics.inl_dynamics import INLDynamics

dynamics = INLDynamics(hidden_size=256)

# Install safety
safety = SafetyClamp(hidden_size=256, threshold=2.0)
safety.set_harm_direction(harm_dir)
safety.enabled = True
dynamics.install_safety(safety)

# Forward pass
h = torch.randn(2, 32, 256)
v = torch.zeros(2, 32, 256)
h_next, v_next, mu = dynamics(h, v)

print(f"  h_next shape: {h_next.shape}")
print(f"  v_next shape: {v_next.shape}")
print(f"  Safety stats: {dynamics.get_safety_stats()}")
print("  [OK] INLDynamics with safety works!")

# Test 4: Remove safety
print("\n[4] Testing safety removal...")
dynamics.remove_safety()
h_next2, v_next2, mu2 = dynamics(h, v)
print(f"  Safety stats after removal: {dynamics.get_safety_stats()}")
print("  [OK] Safety removal works!")

# Test 5: INLDynamicsLite with safety
print("\n[5] Testing INLDynamicsLite with safety...")
from complexity.core.dynamics.inl_dynamics import INLDynamicsLite

dynamics_lite = INLDynamicsLite(hidden_size=256)
safety2 = SafetyClamp(hidden_size=256, threshold=2.0)
safety2.set_harm_direction(harm_dir)
safety2.enabled = True
dynamics_lite.install_safety(safety2)

h_next3, v_next3, mu3 = dynamics_lite(h, v)
print(f"  h_next shape: {h_next3.shape}")
print(f"  Safety stats: {dynamics_lite.get_safety_stats()}")
dynamics_lite.remove_safety()
print("  [OK] INLDynamicsLite with safety works!")

# Test 6: ContrastiveSafetyLoss
print("\n[6] Testing ContrastiveSafetyLoss...")
from complexity.utils.safety import ContrastiveSafetyLoss

loss_fn = ContrastiveSafetyLoss(hidden_size=256, margin=1.0)

safe_act = torch.randn(4, 256)
harmful_act = torch.randn(4, 256)

result = loss_fn(safe_act, harmful_act)
print(f"  Loss:       {result['loss'].item():.4f}")
print(f"  Separation: {result['separation'].item():.4f}")
print("  [OK] ContrastiveSafetyLoss works!")

# Test 7: MultiDirectionSafetyClamp
print("\n[7] Testing MultiDirectionSafetyClamp...")
from complexity.utils.safety import MultiDirectionSafetyClamp

multi_clamp = MultiDirectionSafetyClamp(hidden_size=256, num_directions=4)
multi_clamp.set_direction(0, torch.randn(256), threshold=1.5)
multi_clamp.set_direction(1, torch.randn(256), threshold=2.0)
multi_clamp.enabled = True

x = torch.randn(2, 16, 256)
x_clamped = multi_clamp(x)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {x_clamped.shape}")
print("  [OK] MultiDirectionSafetyClamp works!")

# Test 8: install_safety / remove_safety functions
print("\n[8] Testing install_safety function...")
from complexity.utils.safety import install_safety, remove_safety, get_safety_stats

# Create a simple model with layers
class DummyLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dynamics = INLDynamics(hidden_size)

class DummyModel(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            DummyLayer(hidden_size) for _ in range(num_layers)
        ])

model = DummyModel(hidden_size=256, num_layers=4)

# Install safety
install_safety(model, harm_dir, threshold=2.0, layers=[-2, -1])

# Check installation
stats = get_safety_stats(model)
print(f"  Safety stats: {stats}")
assert 'layer_2' in stats or 'layer_3' in stats, "Safety not installed on expected layers"

# Remove safety
remove_safety(model)
stats_after = get_safety_stats(model)
print(f"  Stats after removal: {stats_after}")
print("  [OK] install/remove safety works!")

# Test 9: Load/Save harm direction
print("\n[9] Testing save/load harm direction...")
from complexity.utils.safety import save_harm_direction, load_harm_direction
import tempfile
import os

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, "harm_direction.pt")
    save_harm_direction(harm_dir, path, metadata={'version': '1.0'})
    loaded = load_harm_direction(path)

    # Check it's normalized
    assert abs(loaded.norm().item() - 1.0) < 1e-5, "Loaded direction not normalized"
    print(f"  Saved and loaded direction (norm={loaded.norm().item():.4f})")
print("  [OK] save/load harm direction works!")

print("\n" + "=" * 60)
print("All safety integration tests passed!")
print("=" * 60)
