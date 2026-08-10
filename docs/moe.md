# MoE implementations

The framework is scoped to a single MoE family: TR-Hash (deterministic
token-ID / hash-table routing). A learned-router MoE (`MixtralMoE`) and the
historical `TokenRoutedMLP` dispatch implementation existed at earlier points
and were removed from the code entirely — not just discouraged — to keep the
canonical path unambiguous. Both are planned to return later as explicit,
clearly-labeled comparison baselines against TR-Hash, once the framework
around them has settled.

## TR-Hash MoE

`TRHashEngineMLP` (`mlp_type="tr_hash_engine"` / `"tr_hash_moe"`) combines a
dense shared SwiGLU branch with deterministically selected narrow experts —
no learned router, no auxiliary balancing loss. See
[TR-Hash Engine](tr-hash-engine.md) for the full contract (backend selection,
CUDA Graph buckets, dynamic capacity) and
[TokenRoutedMLP (removed) and migrating to TR-Hash](token-routed.md) if
you're working with an existing checkpoint from before this change.

```python
from complexity.core.mlp import MLPConfig, TRHashEngineMLP

mlp = TRHashEngineMLP(
    MLPConfig(
        hidden_size=768,
        intermediate_size=512,
        shared_intermediate_size=2048,
        vocab_size=32_000,
        num_experts=4,
        shared_expert=True,
        routing_strategy="token_id_balanced_hash",
        top_k=2,
        top_k_primary_weight=0.5,
    )
)
output = mlp(hidden_states, token_ids=input_ids)
```

## Fair comparisons

When the learned-router and dense baselines return, an MoE comparison
against TR-Hash should report:

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
