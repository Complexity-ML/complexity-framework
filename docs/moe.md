# TR-HASH shared and routed experts

TR-HASH combines a dense shared SwiGLU with four narrow experts. Each token
activates two experts with fixed equal weights; their mixture has output scale
two, while the shared branch has scale one. Routing is causal, deterministic,
and based on original IDs, with no learned router or balancing objective.

```python
from complexity.core.mlp import MLPConfig, TRHashEngineMLP

mlp = TRHashEngineMLP(MLPConfig(
    hidden_size=64, intermediate_size=64, shared_intermediate_size=128,
    vocab_size=256, routing_strategy="token_id_composed_modulo",
))
output = mlp(hidden_states, token_ids=input_ids)
```

`intermediate_size` is aggregate routed width, so this example has four
width-16 experts. Full decoder models also include routed GQA attention and
V channel binding. See [TR-HASH engine](tr-hash-engine.md).

Comparisons report all seeds and metrics, total and active parameters, data
order, masks, optimizer, update/token budgets and measured runtime. A learned
MoE control in a research notebook is a comparison model, not another TR-HASH
mode. Its auxiliary training loss must not be mixed into validation CE.
