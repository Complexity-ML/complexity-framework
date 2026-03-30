# INL Dynamics

**Velocity tracking for LLM training stability.**

## The Problem

After ~400k training steps, models can explode (NaN loss) due to:
- **Small words** (the, a, is) causing oscillations
- **Unbounded beta** that tends to infinity with softplus
- **Velocity accumulation** without damping

## The Solution: INL Dynamics

Second-order dynamical system for stable representation evolution:

```
error = h - mu                      # deviation from equilibrium
v_next = alpha * v - beta * error   # velocity update
h_next = h + dt * gate * v_next     # position update
```

Key parameters:
- **alpha** (inertia): smooths movements
- **beta** (correction): pulls back to equilibrium
- **gate** (amplitude): controls force
- **velocity**: absorbs shocks from small words

## Usage

### Full Version (adaptive parameters)

```python
from complexity.api import INLDynamics

dynamics = INLDynamics(
    hidden_size=768,
    # Initial parameters
    init_alpha=0.9,      # High inertia = smooth
    init_beta=0.1,       # Low correction = stable
    init_gate=0.5,       # Medium amplitude
    dt=0.1,              # Integration step
    # CRITICAL: Stability constraints
    beta_max=2.0,        # Clamp beta to [0, 2]!
    velocity_max=10.0,   # Limit velocity
    mu_min=0.0,          # Min equilibrium
    mu_max=2.0,          # Max equilibrium
)

# Forward
h_next, v_next = dynamics(hidden_states, velocity_states)
```

### Lite Version (fixed parameters)

```python
from complexity.api import INLDynamicsLite

dynamics = INLDynamicsLite(
    hidden_size=768,
    alpha=0.9,
    beta=0.1,  # Fixed, no explosion possible
    gate=0.5,
    dt=0.1,
)
```

## Critical Constraints

| Parameter | Range | Why |
|-----------|-------|-----|
| `alpha` | [0, 1] | sigmoid |
| **`beta`** | **[0, 2]** | **softplus → ∞ without clamp!** |
| `gate` | [0, 1] | sigmoid |
| `mu` | [0, 2] | bounded equilibrium |
| `velocity` | [-10, 10] | prevents runaway |

### The Bug Discovered After 400k Steps

```python
# BAD - causes explosion!
beta = F.softplus(beta_raw)  # [0, inf) ❌

# GOOD - stable
beta = torch.clamp(F.softplus(beta_raw), max=2.0)  # [0, 2] ✓
```

## Integration in a Decoder Layer

```python
class DecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CUDA.flash(...)
        self.dynamics = INLDynamics(config.hidden_size, beta_max=2.0)
        self.mlp = SwiGLU(...)
        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, h, velocity=None):
        # 1. Attention
        residual = h
        h = self.norm1(h)
        h, _ = self.attn(h)

        # 2. INL Dynamics - CRUCIAL!
        h, velocity = self.dynamics(h, velocity)
        h = residual + h

        # 3. MLP
        residual = h
        h = self.norm2(h)
        h = self.mlp(h)
        h = residual + h

        return h, velocity
```

## Monitoring During Training

```python
# Check dynamics stats
for layer in model.layers:
    stats = layer.dynamics.get_dynamics_stats()
    print(f"mu: [{stats['mu_min']:.2f}, {stats['mu_max']:.2f}]")

# If mu goes outside [0, 2] → problem!
```

## Why It Works

1. **Velocity = damper**: Absorbs shocks from frequent tokens
2. **Clamped beta**: No correction explosion
3. **Smooth trajectories**: No jerky movements
4. **Learnable equilibrium**: Model learns where to converge

## Without INL Dynamics

- Training from scratch costs more
- Instability on small words
- Risk of explosion after long training

## With INL Dynamics

- Stable training up to 1M+ steps
- Better generalization
- Slightly higher compute cost (~5%)
