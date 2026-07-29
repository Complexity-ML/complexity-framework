# TR-MoE internals

TR-MoE is the deterministic Token-Routed Mixture-of-Experts feed-forward path
implemented by `complexity.core.mlp.TokenRoutedMLP`.

It can be paired with GQA (**TR-GQA**) or MHA (**TR-MHA**).

## Computation

Each layer contains:

1. an optional dense shared SwiGLU path;
2. \(E\) narrow SwiGLU experts;
3. one or more deterministic route tables;
4. a dispatch implementation.

```text
contextual hidden state x ───────────────► shared SwiGLU ──┐
                                                           ├─► sum / gated sum
token ID t ─► layer route table ─► selected experts(x) ────┘
```

Token identity selects parameters. Every selected branch still transforms the
current contextual hidden state.

## Route construction

### `zipf`

When `token_frequencies` is supplied, tokens are sorted by descending
frequency and greedily assigned to the currently lightest expert. This aims to
balance expected frequency mass.

When no frequency tensor is supplied, `zipf` intentionally falls back to
token-ID modulo routing. The framework does not infer corpus frequencies.

### `modulo`

The primary route starts from `token_id % num_experts`, followed by a
deterministic layer-specific permutation. Auxiliary routes advance cyclically.

### `modulo_balanced_secondary`

The primary route is modulo-based. Secondary routes are built greedily from
the frequency artifact while excluding experts already selected for that token.
This is the route used in the short matched TR-MHA pilot.

### Controls

- `round_robin`: assignment over frequency rank when frequencies exist;
- `random`: seeded deterministic lexical partition;
- `lsh_hidden`: experimental fixed random-hyperplane routing on hidden states.

`lsh_hidden` is contextual rather than token-ID routing and should be labeled
as a separate control.

## Per-layer route variation

Each lexical table receives a deterministic layer-specific permutation of
expert labels. This preserves its load distribution while allowing a token to
reach different expert parameters at different depths.

## Top-k routing

`top_k=1` selects one expert. For `top_k>1`, the framework precomputes distinct
route tables and combines them using `top_k_primary_weight`.

For top-2 with primary weight \(p\):

\[
y_{\mathrm{routed}}
=p\,E_{r_1(t)}(x)+(1-p)\,E_{r_2(t)}(x).
\]

No learned router is executed at runtime for lexical strategies.

## Shared path

With `shared_expert=True`:

```text
output = shared_output + routed_output
```

With `use_shared_routed_gates=True`:

```text
output = shared_gate * shared_output + routed_gate * routed_output
```

`shared_expert_chunk_tokens` chunks the dense shared computation across the
token dimension to lower peak activation memory without changing the
mathematical result.

## Dispatch paths

### Universal PyTorch path

The implementation sorts tokens by expert and falls back to a PyTorch
expert-compute path when custom kernels are unavailable. The current fallback
is designed for portability and autograd correctness; it is not claimed to be
the fastest path on every device.

### CGGR CUDA/Triton path

When CUDA, Triton, and the autograd-aware grouped-GEMM kernel are available,
`use_cggr="auto"` may select CGGR. The actual path is stored in
`last_dispatch_path` and logged once.

### Static capacity

`static_expert_capacity=True` selects an export-friendly path intended for
pipeline tracing. It changes dispatch mechanics, not the route definition.

## Telemetry

`collect_moe_telemetry=True` records:

- token counts per route expert;
- shared output RMS;
- routed output RMS;
- scheduled top-k and gate values.

Telemetry is off by default because reductions and host-side logging can reduce
throughput.

## Model example

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    num_key_value_heads=2,
    attention_type="gqa",
    vocab_size=200_019,
    mlp_type="token_routed",
    num_experts=4,
    intermediate_size=128,
    shared_expert=True,
    shared_intermediate_size=1536,
    routing_strategy="zipf",
    top_k=2,
    top_k_primary_weight=0.5,
)

model = ComplexityModel(config)
```

To build TR-MHA, set `attention_type="mha"` and
`num_key_value_heads=num_attention_heads`.

## Direct module example

```python
import torch

from complexity.core.mlp import MLPConfig, TokenRoutedMLP

mlp = TokenRoutedMLP(
    MLPConfig(
        hidden_size=64,
        intermediate_size=128,
        vocab_size=1_000,
        num_experts=4,
        shared_expert=True,
        shared_intermediate_size=128,
        routing_strategy="modulo",
        top_k=2,
        top_k_primary_weight=0.5,
    )
)

hidden = torch.randn(2, 16, 64)
token_ids = torch.randint(0, 1_000, (2, 16))
output = mlp(hidden, token_ids=token_ids)
```

## Claims that are intentionally not made

- Routing is not perfectly balanced for every finite batch.
- Deterministic routing does not make expert collapse mathematically
  impossible.
- CUDA graph compatibility depends on the selected dispatch and serving
  integration.
- A route table alone does not establish better generalization.

These questions require measurement under a named protocol.
