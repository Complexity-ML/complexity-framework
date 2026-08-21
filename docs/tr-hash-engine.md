# TR-Hash execution engine

`complexity.tr_hash` is the execution contract for shared computation plus
deterministic token-ID-routed residual experts. It is not a learned contextual
router.

`TRHashEngineMLP` (`mlp_type="tr_hash_engine"`, alias `tr_hash_moe`) adapts the
engine to each decoder block. New models should use this path. Historical
`TokenRoutedMLP` checkpoints require the explicit
[conversion guide](token-routed.md).

## Released 200M engine shape

The public 200M model uses, in each of 16 layers:

| Field | Value |
|---|---:|
| Hidden size | 896 |
| Vocabulary | 32,000 |
| Shared width | 3,072 |
| Stored experts | 4 |
| Per-expert width | 64 |
| Stored routed width | 256 |
| Active routes / width | top-2 / 128 |
| Route strategy | `token_id_multi_hash` |
| Hash channels | 2 |
| Route weights | 0.5 / 0.5 |

At the `ModelConfig`/`MLPConfig` layer, `intermediate_size=256` is the total
stored routed width. `TRHashEngineMLP` validates divisibility and passes
`expert_width=intermediate_size // num_experts`, or 64, to
`TRHashEngineConfig`.

## Computation

For contextual hidden state `x`, token ID `t`, shared branch `S`, expert `E`,
fixed layer route `R`, fixed route weights `a`, and fixed output scales:

```text
y = shared_scale × S(x)
  + routed_scale × Σ_j a_j E[R[layer, j, t]](x)
```

The hash selects parameters. Attention has already contextualized `x`; routing
does not reduce the model to an embedding lookup.

## Route construction

The model-block adapter supports four canonical deterministic strategies:

| Strategy | Construction |
|---|---|
| `modulo_cyclic` | layer-permuted modulo primary with cyclic distinct routes |
| `token_id_balanced_hash` | collision-free tuples with exact marginal balance over complete blocks |
| `token_id_multi_hash` | summed independent rendezvous scores followed by fixed top-k |
| `token_id_hierarchical_hash` | balanced parent routes with deterministic clone selection |

The released model uses multi-hash. Each token/expert pair receives one 32-bit
score per channel. Channel scores are summed, then the highest-scoring distinct
experts become the route. The result is compiled into the ordinary
`[top_k, vocab_size]` table, so additional hash channels do not add runtime
routing-model compute.

Persisted tables must stay paired with the expert weights they trained. Loading
or converting a checkpoint must transplant its exact table rather than
silently regenerate a different mapping.

## Supported engine matrix

| Axis | Contract |
|---|---|
| Experts | 1, 2, 4, 8, 16 |
| Active routes | top-1, top-2, top-4, never greater than experts |
| Hash channels | 2 through 8 for multi-hash |
| Widths | independent shared and per-expert widths |
| Precision | FP32, BF16, FP16 |
| Attention metadata | GQA or MHA |
| Training parallelism | one device or replicated DDP |
| Execution | PyTorch, CGGR/Triton, fused CUDA, CUDA Graph reference path |

FP8 and INT8 enum values fail explicitly in this engine until their grouped
GEMM implementations exist. TR-Hash-i64 has a separate inference quantization
contract; do not infer framework-training support from the serving runtime.

## Direct engine example

`TRHashEngineConfig.expert_width` is explicitly per expert:

```python
import torch

from complexity.tr_hash import (
    AttentionBackbone,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashStrategy,
)

engine = TRHashEngine(
    TRHashEngineConfig(
        hidden_size=896,
        vocab_size=32_000,
        num_experts=4,
        top_k=2,
        shared_width=3_072,
        expert_width=64,
        routing_strategy=TRHashStrategy.MULTI_HASH,
        route_hash_count=2,
        route_weights=(0.5, 0.5),
        routed_output_scale=2.0,
        attention_backbone=AttentionBackbone.GQA,
    )
)

hidden = torch.randn(2, 128, 896)
token_ids = torch.randint(0, 32_000, (2, 128))
output = engine(hidden, token_ids)
```

## Backend selection

`backend="auto"` selects the fastest tested compatible path and reports the
decision. Top-2 shapes with two to eight experts can use the hash-native fused
CUDA path. Other supported shapes may use general CGGR/Triton. CPU, MPS, ROCm
under conservative policy, or unsupported CUDA shapes use the universal
PyTorch reference.

Inspect the realized path:

```python
layer = model.layers[0].mlp
print(layer.engine.last_backend)
print(layer.capability_summary("cuda"))
```

Requested flags are not evidence that a kernel ran. Throughput reports must
record the selected backend from runtime state after warm-up.

## Dynamic capacity

An allocated engine can be reduced to a deterministic prefix-width
sub-network:

```python
engine.set_active_capacity(num_experts=2, expert_width=32)
```

The change re-derives routes for the active expert count and invalidates cached
CUDA Graphs. PyTorch and CGGR support reduced capacity. Fused CUDA requires the
full allocated capacity because its compiled pair metadata is shape-specific.
Dynamic capacity is an experimental runtime control; it does not describe the
released 200M checkpoint.

## CUDA Graph buckets

The framework reference manager requires:

- `phase="inference"`;
- `backend="cuda_graph"`;
- one or more static `(batch, sequence)` buckets;
- `eval()` and `torch.no_grad()`.

It pads to the smallest containing bucket, copies into persistent buffers,
replays the captured graph, and returns the unpadded slice. Production serving
and continuous batching are implemented separately by TR-Hash-i64.

## Correctness policy

The universal PyTorch implementation is the numerical reference. Every
optimized path must have forward and backward parity tests for its supported
shape and dtype. A missing optimized dependency must produce an explicit
fallback reason; a required production kernel must fail preflight rather than
silently relabel the fallback.

See [GPU and dispatch paths](cuda.md) and
[TR-HASH MoE 200M release](tr-hash-200m-release.md).
