# Mixture of Experts (MoE) - Background

This document explains **standard MoE with learned routing** for reference.

> **Complexity Framework uses [Token-Routed MLP](token-routed.md)** - our deterministic approach without learned routing.

## Standard MoE (Not Implemented)

Traditional MoE uses a learned router to select experts:

```
Token → Router Network → [Expert 1, Expert 3] → Weighted Sum → Output
```

**Examples**: Mixtral, GPT-4, Switch Transformer

### Problems with Learned Routing

| Issue | Description |
|-------|-------------|
| **Expert collapse** | Some experts get all tokens, others die |
| **Load imbalance** | Requires aux_loss to fix |
| **Non-deterministic** | Different runs give different outputs |
| **Routing overhead** | Router forward pass adds latency |
| **Complex tuning** | Capacity factors, top-k, balancing weights |

## Our Solution: Token-Routed MLP

Instead of learning which expert to use, we use the **token ID**:

```python
expert_id = token_id % num_experts
```

**No router. No aux_loss. No load balancing. Perfect distribution.**

See [Token-Routed MLP](token-routed.md) for full documentation.

## Comparison

| Aspect | Learned MoE | Token-Routed (Ours) |
|--------|-------------|---------------------|
| Router | Neural network | **None** |
| aux_loss | Required | **None** |
| Load balancing | Must be learned | **Perfect by design** |
| Expert collapse | Possible | **Impossible** |
| Deterministic | No | **Yes** |
| Implementation | Complex | **One line** |

## Usage in Complexity Framework

```python
from complexity.core.mlp import TokenRoutedMLP, MLPConfig

# Our deterministic MoE
config = MLPConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_experts=4,
    vocab_size=100000,
)

moe = TokenRoutedMLP(config)

# Forward - no aux_loss!
output = moe(hidden_states, token_ids)
```

## See Also

- [Token-Routed MLP](token-routed.md) - Our implementation
- [INL Dynamics](dynamics.md) - Training stability
