# MoE implementations

The framework contains two different MoE families.

| Family | Class | Routing | Auxiliary balancing loss |
| --- | --- | --- | --- |
| TR-MoE | `TokenRoutedMLP` | fixed lexical tables or experimental LSH | no |
| Learned-router control | `MixtralMoE` | learned logits and top-k selection | implementation-dependent |

## TR-MoE

TR-MoE is the main research path. It combines a dense shared SwiGLU branch with
deterministically selected narrow experts. See
[TR-MoE internals](token-routed.md).

```python
from complexity.core.mlp import MLPConfig, TokenRoutedMLP

mlp = TokenRoutedMLP(
    MLPConfig(
        hidden_size=768,
        intermediate_size=512,
        shared_intermediate_size=2048,
        vocab_size=32_000,
        num_experts=4,
        shared_expert=True,
        routing_strategy="modulo_balanced_secondary",
        top_k=2,
        top_k_primary_weight=0.5,
    )
)
output = mlp(hidden_states, token_ids=input_ids)
```

## Learned-router control

`MixtralMoE` exists for controlled comparisons with a learned router. Its
configuration and loss handling must be inspected for the specific experiment;
the framework does not claim that every historical run used a common,
production-equivalent Mixtral recipe.

```python
from complexity.core.mlp import MLPConfig, MixtralMoE

mlp = MixtralMoE(
    MLPConfig(
        hidden_size=768,
        intermediate_size=2048,
        num_experts=4,
    )
)
output = mlp(hidden_states)
```

## Fair comparisons

An MoE comparison should report:

- total trainable parameters;
- active parameters per token;
- token budget and data order;
- optimizer and learning-rate schedule;
- number of seeds;
- training and evaluation NLL;
- throughput on the same hardware;
- the realized route distribution;
- any auxiliary loss and its coefficient.

Historical figures without those fields are not used as headline evidence in
the current documentation.
