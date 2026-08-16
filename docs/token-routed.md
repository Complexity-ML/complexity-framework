# TokenRoutedMLP (removed) and migrating to TR-Hash

`TokenRoutedMLP` was the original, feature-complete Token-Routed MoE
dispatch implementation (`complexity.core.mlp.token_routed`). It predated
`complexity.tr_hash` and has been removed — new models use `TRHashEngineMLP`
(`mlp_type="tr_hash_engine"` / `"tr_hash_moe"`), documented in
[`tr-hash-engine.md`](tr-hash-engine.md). This page exists to help migrate
anything still referring to the old implementation.

## Why it was removed

The framework is now scoped to `complexity.tr_hash.TRHashEngine` as the
single canonical MoE execution path. `TokenRoutedMLP` had accumulated
several ablation-only features (`zipf`/`round_robin`/`random`/`lsh_hidden`
routing, `modulo_balanced_secondary`/`modulo_frequency_balanced_secondary`
frequency-aware secondary routing, `expert_initialization="legacy_kaiming"`)
that existed only to replay historical experiments and don't belong in the
canonical path going forward.

Constructing a config with the old `mlp_type` values (`token_routed`,
`sort_split`, `sort_split_moe`, `deterministic_moe`, `complexity`) now raises
a clear `ValueError` pointing here instead of silently failing deeper in the
stack.

## Loading an existing `token_routed` checkpoint

Do **not** try to construct `ModelConfig(mlp_type="token_routed", ...)` — it
will raise. Instead convert the checkpoint's tensors to `TRHashEngineMLP`'s
layout:

```python
from complexity.utils.token_routed_conversion import convert_token_routed_checkpoint_dir

model = convert_token_routed_checkpoint_dir("/path/to/old/checkpoint")
```

This works directly on the checkpoint directory's raw `config.json` +
`model.safetensors` — it never needs the removed `TokenRoutedMLP` class to be
importable. Lower-level entry points, if you're working with an in-memory
state dict/config dict instead of a directory:

```python
from complexity.utils.token_routed_conversion import (
    convert_token_routed_checkpoint,  # (state_dict, config_dict) -> model
    convert_token_routed_config,      # config_dict -> ModelConfig(mlp_type="tr_hash_engine")
    convert_token_routed_state_dict,  # state_dict -> (converted_state_dict, route_tables)
)
```

### What the conversion does

`TokenRoutedMLP` and `TRHashEngineMLP` store per-expert weights in the same
`[num_experts, hidden_size, expert_width]` / `[num_experts, expert_width,
hidden_size]` tensor layout — the conversion is a rename, not a reshape or
numeric remap:

| `token_routed` tensor | `tr_hash_engine` tensor |
|---|---|
| `mlp.gate_proj_w` | `mlp.engine.expert_gate` |
| `mlp.up_proj_w` | `mlp.engine.expert_up` |
| `mlp.down_proj_w` | `mlp.engine.expert_down` |
| `mlp.shared_gate.weight` | `mlp.engine.shared_gate.weight` |
| `mlp.shared_up.weight` | `mlp.engine.shared_up.weight` |
| `mlp.shared_down.weight` | `mlp.engine.shared_down.weight` |
| `mlp.topk_token_to_expert` | `mlp.engine.route_table` (installed via `TRHashEngine.load_route_table`, not the state dict) |

The routing table is transplanted exactly rather than regenerated: the two
implementations' route-table constructions are independent algorithms with
no guarantee of agreeing, so regenerating from `routing_strategy` would
silently pair an expert's trained weights with a different set of tokens
than it was trained on. `load_route_table` also recompiles fused-CUDA pair
metadata to match, so the converted model stays correct on the fused path
too.

Any tensor the conversion doesn't recognize (e.g. `hash_pair_gate_logits`
from `learn_hash_pair_gates`, an ablation-only feature with no
`tr_hash_engine` equivalent) raises rather than being silently dropped.

### Verified equivalence

Converting a real 500M-parameter `token_routed` checkpoint and comparing
logits against the original (same prompt, same weights) produced bit-exact
output in BF16. A small synthetic model shows a few-millipoint fp32 drift
from the two implementations summing expert contributions in a different
order — not a correctness issue; see
`tests/test_token_routed_to_tr_hash_conversion.py`.

## Feature gaps versus the old implementation

A few things `TokenRoutedMLP` had are not (yet) in `TRHashEngineMLP`:

- **Per-expert telemetry** (`expert_counts`, `training.moe_telemetry`
  aggregation) and **tensor-parallel expert-weight sharding**
  (`parallel.tensor_parallel`) are duck-typed against the old class's
  attribute names. They now permanently return empty/no-op for any model
  (verified: this degrades gracefully, it does not crash) until equivalent
  hooks are added to `TRHashEngine`.
- **LSH/semantic routing**, **learned hash-pair/channel gates**, and
  **static export capacity** (`static_expert_capacity`) have no equivalent —
  though `TRHashEngineMLP`'s single dispatch path is already unconditionally
  export/pipeline-trace-safe, so the last one specifically was never needed.
