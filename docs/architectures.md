# Architecture and naming

Complexity Framework composes attention and a feed-forward path in a pre-norm
causal decoder:

```text
x ─► RMSNorm ─► attention ─► residual add
└─► RMSNorm ─► TR-MoE     ─► residual add
```

Token identity selects routed expert parameters. It does not replace
attention: every selected expert transforms the current contextual hidden
state.

## Released TR-HASH MoE 200M

The current reference release is TR-GQA with the following shape:

```python
from scripts.train_tr_hash_200m_200b import make_config

config = make_config()
```

| Component | Released value |
|---|---:|
| Parameters | 201,194,368 |
| Layers / hidden size | 16 / 896 |
| Query / KV heads | 14 / 2 |
| Vocabulary / context | 32,000 / 2,048 |
| Shared SwiGLU width | 3,072 |
| Stored routed width | 256 total = 4 experts × 64 |
| Active routed width | 128 = top-2 × 64 |
| Route construction | 2-channel multi-hash rendezvous voting |

The release uses `intermediate_size=256`. In `TRHashEngineMLP`, this field is
the total stored routed width, not a per-expert width. It is divided by
`num_experts=4` when the engine is built.

## Public decoder names

### TR-GQA

TR-GQA combines grouped-query attention with TR-MoE:

```python
ModelConfig(
    attention_type="gqa",
    num_attention_heads=14,
    num_key_value_heads=2,
    mlp_type="tr_hash_engine",
)
```

Multiple query heads share each K/V head. The released 200M model is TR-GQA.

### TR-MHA

TR-MHA combines full multi-head attention with the same TR-MoE:

```python
ModelConfig(
    attention_type="mha",
    num_attention_heads=8,
    num_key_value_heads=8,
    mlp_type="tr_hash_engine",
)
```

Only the attention layout changes. TR-MHA is implemented, but the 200M release
metrics do not transfer to an untrained TR-MHA configuration.

### TR-MoE

For layer `l`, contextual hidden state `x`, token ID `t`, shared branch `S`,
expert `E`, fixed route table `R`, fixed route weights `a`, and output scales
`α` and `β`:

```text
TRMoE_l(x, t) = α S_l(x)
              + β Σ_j a_j E_l,R[l,j,t](x)
```

The released model uses two routes with `a=(0.5, 0.5)`, `α=1`, and `β=2`.
Both expert outputs therefore contribute equally. Routes are persisted
`[top_k, vocab_size]` lookup tables with no learned gate and no auxiliary
load-balancing loss.

## Multi-hash route construction

For each layer and token/expert pair, the release computes two independent
32-bit rendezvous scores. Scores are summed across channels, then the two
highest-scoring distinct experts become the fixed top-2 route. Construction is
deterministic from token ID, expert ID, layer index, and route seed.

The result is compiled at model construction or loaded from the checkpoint.
Inference does not execute a routing neural network. Optimized and PyTorch
paths consume the same route table.

## Required lexical refinement boundary

TR-HASH language-model releases use a mandatory full-parameter refinement
stage between base pretraining and instruction SFT. The routing table is
lexical and fixed: a token ID continues to address the same expert routes
during refinement, while attention still supplies the contextual hidden state
transformed by those experts.

Refinement reuses the exact pretraining `unique_core` once, with no replay,
augmentation or new source, and starts a fresh optimizer and learning-rate
schedule from the pretrained weights. This gives the shared SwiGLU path and
the fixed routed experts a clean same-corpus optimization pass before the
training objective changes to supervised instruction following.

This is enforced as an architecture-training contract by Complexity Framework:

```text
pretraining -> same-corpus full-parameter refinement -> full-parameter SFT
```

A different corpus is not refinement. It must be identified as continued
pretraining or supervised fine-tuning according to its objective.

## Other routing strategies

`TRHashEngineMLP` accepts:

- `modulo_cyclic`;
- `token_id_balanced_hash`;
- `token_id_multi_hash`;
- `token_id_hierarchical_hash`.

Compatibility aliases may parse in `ModelConfig`, but new release documents
should use one of the canonical values above. Historical stochastic,
frequency-aware, hidden-state LSH, and learned-router paths are not the current
TR-Hash execution contract.

## Dense controls and experimental attention adapters

`dense_deterministic` is a separately named deterministic-initialization dense
SwiGLU control. It is useful for bounded comparisons but is not TR-MoE.

The attention registry values `tr_mha` and `tr_mha_v2` add experimental
token-routed residual adapters inside MHA. They are not synonyms for
**TR-MHA = MHA + TR-MoE** and do not inherit the 200M release evidence.

Other sequence mixers and multimodal position-routed modules are experimental.
Their routing keys and evidence must be documented separately.

## Configuration invariants

- `hidden_size` is divisible by `num_attention_heads`;
- query-head count is divisible by KV-head count;
- MHA uses equal query and KV head counts;
- routed `intermediate_size` is divisible by `num_experts`;
- `top_k` does not exceed `num_experts`;
- `route_hash_count` is between 2 and 8;
- TR-MoE receives the original token IDs;
- persisted route tables must be loaded with the expert weights they trained;
- parameter and active-width claims are measured after construction.

## Related pages

- [TR-HASH MoE 200M release](tr-hash-200m-release.md)
- [TR-Hash engine](tr-hash-engine.md)
- [Getting started](getting-started.md)
- [Historical TokenRoutedMLP migration](token-routed.md)
