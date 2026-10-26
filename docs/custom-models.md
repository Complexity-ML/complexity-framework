# Custom models

`ModelConfig` describes the decoder, and `ComplexityModel` allocates its
weights. The default is the single composed TR-HASH architecture.

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=64, num_hidden_layers=2, vocab_size=256,
    max_position_embeddings=64, use_custom_kernels=False, use_cggr=False,
)
model = ComplexityModel(config)
print(model.num_parameters())
```

TR-HASH uses four query heads, one KV head, four experts, two fixed equal routes,
and shared/routed scales one/two. Shared and expert widths can be resized;
changing a routing algorithm or substituting MHA does not create this model.
See [the formulas](tr-hash-engine.md).

To construct a dense control, explicitly select `mlp_type="dense"`. Total
parameter matching does not by itself match active computation per token.

The registry supports separately named custom components. An MLP follows
`MLPBase.forward(hidden_states, **kwargs)`; attention follows
`(hidden_states, cache_or_state)`. Token-aware attention must receive original
IDs and carry its context in the cache. Add causal prefix, gradient and cached
continuation checks before using a new component in a comparison.

`Model.create(config, **overrides)` copies the configuration and validates
updates before allocating weights. Unknown authoring fields are rejected.
