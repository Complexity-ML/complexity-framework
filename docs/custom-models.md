# Custom models

`ModelConfig` composes registered attention, MLP, normalization, and position
components into `ComplexityModel`.

## Compose TR-GQA

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=512,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=2,
    attention_type="gqa",
    vocab_size=32_000,
    mlp_type="tr_hash_engine",
    num_experts=4,
    intermediate_size=256,
    shared_expert=True,
    shared_intermediate_size=1536,
    routing_strategy="modulo_cyclic",
    top_k=2,
)
model = ComplexityModel(config)
```

## Compose TR-MHA

```python
config = ModelConfig(
    hidden_size=512,
    num_hidden_layers=12,
    num_attention_heads=8,
    num_key_value_heads=8,
    attention_type="mha",
    vocab_size=32_000,
    mlp_type="tr_hash_engine",
    num_experts=4,
    intermediate_size=256,
    shared_expert=True,
    shared_intermediate_size=1536,
    routing_strategy="modulo_cyclic",
    top_k=2,
)
model = ComplexityModel(config)
```

## Register an MLP

```python
import torch.nn as nn

from complexity.core.mlp.base import MLPBase
from complexity.core.registry import register_mlp


@register_mlp("my_ffn")
class MyFFN(MLPBase):
    def __init__(self, config):
        super().__init__(config)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        return self.proj(hidden_states)
```

Use it through:

```python
config = ModelConfig(mlp_type="my_ffn")
```

## Register attention

Custom attention must follow the `AttentionBase` return contract:

```text
(hidden_states, new_cache_or_state)
```

Register with:

```python
from complexity.core.registry import register_attention

@register_attention("my_attention")
class MyAttention(...):
    ...
```

The transformer block passes `token_ids` only to attention types explicitly
listed as token-aware in `complexity/models/block.py`. A new token-aware
attention implementation must update that dispatch contract and add tests.

## Parameter matching

Before comparing a custom model:

```python
print(model.num_parameters())
```

Matching total parameters is not the same as matching active parameters per
token. Report both for sparse models.

## Test expectations

A new component should cover:

- construction through the registry;
- forward shape and dtype;
- backward pass;
- determinism where claimed;
- cache/state behavior;
- save/load round trip;
- CPU fallback;
- target accelerator parity;
- parameter and active-compute accounting.
