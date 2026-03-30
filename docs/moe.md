# Mixture of Experts (MoE)

Two MoE approaches available:

| Type | Class | Routing | aux_loss |
|------|-------|---------|----------|
| **Token-Routed** (ours) | `TokenRoutedMLP` | Zipf-balanced deterministic | None |
| **Mixtral-style** (baseline) | `MixtralMoE` | Learned router + top-1 | Required |

## Token-Routed MLP (Recommended)

Sort-and-split dispatch with deterministic routing. Zero overhead, perfect load balancing.

```python
from complexity.core.mlp import TokenRoutedMLP, MLPConfig

config = MLPConfig(
    hidden_size=768,
    intermediate_size=2048,
    num_experts=4,
    vocab_size=32000,
    shared_expert=True,
    token_frequencies=freqs,  # Zipf-balanced routing
)
mlp = TokenRoutedMLP(config)
output = mlp(hidden_states, token_ids=input_ids)  # No aux_loss!
```

See [Token-Routed MLP](token-routed.md) for full documentation.

## Mixtral-style MoE (Baseline)

Learned router with load balancing loss. For comparison with standard MoE.

```python
from complexity.core.mlp import MixtralMoE, MLPConfig

config = MLPConfig(
    hidden_size=768,
    intermediate_size=2048,
    num_experts=4,
    vocab_size=32000,
    shared_expert=True,
)
mlp = MixtralMoE(config)
output = mlp(hidden_states)
# aux_loss stored in mlp.last_aux_loss
```

## Comparison

| Aspect | Token-Routed (ours) | Mixtral-style |
|--------|---------------------|---------------|
| Router | **None (table lookup)** | nn.Linear + softmax |
| aux_loss | **None** | Required |
| Load balancing | **Perfect by design** | Must be learned |
| Expert collapse | **Impossible** | Possible |
| Deterministic | **Yes** | No |
| CUDA graph safe | **Yes** | Needs special handling |
| Dispatch | Sort-and-split (bmm) | Sort-and-split (bmm) |

### Training Results (500M tokens, 700 steps avg)

| Configuration | Avg Loss |
|---------------|----------|
| **TR + Mu + Zipf** | **5.026** |
| Mixtral (learned) | 5.110 |
| Dense baseline | 5.205 |

Token-Routed converges faster because experts specialize immediately without learning a router.

![Loss Curves](../figures/fig_loss_curves.png)

## See Also

- [Token-Routed MLP](token-routed.md)
- [Mu-Guidance](dynamics.md)
- [Training](training.md)
