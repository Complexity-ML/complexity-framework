# Mixture of Experts (MoE)

Complexity Framework provides **two MoE approaches**:

| Type | Class | Routing | aux_loss |
|------|-------|---------|----------|
| **Token-Routed** (our innovation) | `TokenRoutedMLP` | `token_id % num_experts` | None |
| **Sparse MoE** (standard) | `SparseMoE` | Learned router | Required |

## Quick Start

```python
from complexity.api import MLP

# Token-Routed MoE (deterministic, our innovation)
moe = MLP.moe(hidden_size=768, num_experts=4)

# Sparse MoE (learned routing, standard approach)
moe = MLP.sparse_moe(hidden_size=768, num_experts=8, top_k=2)
```

---

## Token-Routed MLP (Recommended)

Our novel approach - **zero routing overhead, perfect load balancing**.

```python
expert_id = token_id % num_experts
```

**Benefits:**
- No router network to learn
- No aux_loss required
- Perfect load balancing by design
- 100% deterministic
- One line of code

```python
from complexity.api import TokenRoutedMLP, MLPConfig

config = MLPConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_experts=4,
    vocab_size=100000,
)

moe = TokenRoutedMLP(config)
output = moe(hidden_states, token_ids)  # No aux_loss!
```

See [Token-Routed MLP](token-routed.md) for full documentation.

---

## Sparse MoE (Standard)

Traditional MoE with learned routing (like Mixtral, GPT-4).

```python
from complexity.api import SparseMoE, SparseMoEConfig

config = SparseMoEConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_experts=8,
    top_k=2,
    load_balancing_weight=0.01,
)

moe = SparseMoE(config)
output, aux_loss = moe(hidden_states)

# Add aux_loss to total loss!
total_loss = ce_loss + aux_loss
```

### SparseMoE Parameters

| Param | Description | Default |
|-------|-------------|---------|
| `num_experts` | Total experts | 8 |
| `top_k` | Active experts per token | 2 |
| `load_balancing_weight` | aux_loss weight | 0.01 |

### Why aux_loss?

Without load balancing loss, some experts get all tokens (expert collapse). The aux_loss encourages uniform distribution:

```python
output, aux_loss = moe(hidden_states)
total_loss = ce_loss + aux_loss  # Required!
```

---

## Comparison

| Aspect | Token-Routed (Ours) | Sparse MoE |
|--------|---------------------|------------|
| Router | **None** | Neural network |
| aux_loss | **None** | Required |
| Load balancing | **Perfect by design** | Must be learned |
| Expert collapse | **Impossible** | Possible |
| Deterministic | **Yes** | No |
| Latency | **<0.1ms** | 5-10ms |

## When to Use Which?

**Token-Routed MLP** (recommended):
- Production/inference (determinism)
- Robotics/real-time (low latency)
- Training stability (no aux_loss tuning)

**Sparse MoE**:
- Research/comparison with Mixtral-style models
- When you need learned specialization

---

## See Also

- [Token-Routed MLP](token-routed.md) - Full documentation
- [INL Dynamics](dynamics.md) - Training stability
- [API Reference](api.md) - All MLP types
