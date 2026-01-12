# Mixture of Experts (MoE)

Token-Routed MLP for efficient model scaling.

## Concept

Instead of a single MLP, you have **N experts** and a **router** that selects top-k experts per token:

```
Token → Router → [Expert 1, Expert 3] → Weighted Sum → Output
```

**Benefits:**
- Scale parameters without scaling compute (sparse activation)
- Each expert can specialize
- Used by Mixtral, GPT-4, Switch Transformer

## Basic Usage

```python
from complexity.api import MLP, TokenRoutedMLP

# Via factory
moe = MLP.moe(
    hidden_size=4096,
    num_experts=8,    # 8 experts
    top_k=2,          # 2 active experts per token
)

# Direct
moe = TokenRoutedMLP(
    hidden_size=4096,
    intermediate_size=11008,
    num_experts=8,
    top_k=2,
)

# Forward
output, aux_loss = moe(hidden_states)
```

## Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `hidden_size` | Hidden state dimension | - |
| `intermediate_size` | Expert inner dimension | `hidden_size * 4` |
| `num_experts` | Total number of experts | 8 |
| `top_k` | Active experts per token | 2 |
| `expert_capacity` | Max tokens per expert | `None` (auto) |
| `load_balancing_weight` | Balancing loss weight | 0.01 |

## Load Balancing Loss

To prevent all tokens from going to the same experts:

```python
output, aux_loss = moe(hidden_states)

# Add to total loss
total_loss = ce_loss + aux_loss  # aux_loss encourages balancing
```

## TokenRoutedMLPParallel

Optimized version for multi-GPU:

```python
from complexity.api import TokenRoutedMLPParallel

# Experts distributed across GPUs
moe = TokenRoutedMLPParallel(
    hidden_size=4096,
    num_experts=64,      # More experts
    top_k=2,
    expert_parallel=True,
)
```

## Full Architecture

```python
import torch.nn as nn
from complexity.api import (
    GQA, TokenRoutedMLP, RMSNorm, RoPE,
    INLDynamics
)

class MoETransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Standard attention
        self.attn = GQA(config)

        # MLP replaced by MoE
        self.moe = TokenRoutedMLP(
            hidden_size=config.hidden_size,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )

        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, x, **kwargs):
        # Attention
        h = self.norm1(x)
        x = x + self.attn(h, **kwargs)[0]

        # MoE
        h = self.norm2(x)
        moe_out, aux_loss = self.moe(h)
        x = x + moe_out

        return x, aux_loss


class MoEModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(config)
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

        # INL Dynamics for stability
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            beta_max=2.0,
        )

    def forward(self, input_ids, velocity=None):
        h = self.embed(input_ids)

        if velocity is None:
            velocity = torch.zeros_like(h)

        total_aux_loss = 0.0
        for block in self.blocks:
            h, aux_loss = block(h)
            total_aux_loss += aux_loss

            # Dynamics for stability
            h, velocity = self.dynamics(h, velocity)

        logits = self.head(self.norm(h))

        return logits, total_aux_loss, velocity
```

## Recommended Configs

| Model | Experts | Top-k | Total Params | Active Params |
|-------|---------|-------|--------------|---------------|
| Small MoE | 4 | 1 | ~500M | ~200M |
| Medium MoE | 8 | 2 | ~2B | ~500M |
| Large MoE | 16 | 2 | ~8B | ~1B |
| Mixtral-style | 8 | 2 | ~47B | ~13B |

## Tips

1. **Load balancing**: Always use `aux_loss` otherwise some experts die
2. **Capacity factor**: If tokens are dropped, increase `expert_capacity`
3. **top_k=1**: More efficient but less stable
4. **top_k=2**: Standard, good balance
5. **With INL Dynamics**: Velocity tracking helps stabilize routing

## See Also

- [API Reference](api.md) - Full reference
- [Efficient Training](efficient.md) - Memory optimizations
- [INL Dynamics](dynamics.md) - Training stability
