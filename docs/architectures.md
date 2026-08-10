# Architecture and naming

Complexity Framework composes a sequence mixer and a feed-forward path in a
pre-norm causal decoder:

```text
x ─► RMSNorm ─► attention ─► residual add
  └► RMSNorm ─► FFN       ─► residual add
```

Token identity may influence the FFN, attention, or an experimental lexical
residual, but these mechanisms are configured independently.

## Primary decoder families

### TR-GQA

TR-GQA combines grouped-query attention with TR-MoE:

```python
ModelConfig(
    attention_type="gqa",
    num_attention_heads=8,
    num_key_value_heads=2,
    mlp_type="tr_hash_engine",
)
```

Multiple query heads share each K/V head. The FFN contains a shared dense path
and deterministic token-selected experts.

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

Every query head has its own K/V head. Only the attention layout changes;
TR-MoE routing and expert computation remain the same.

### Dense controls

Removed. Dense (`mlp_type="swiglu"`/`"gelu"`/`"geglu"`/`"standard"`) was
fully removed from the codebase to scope the framework to TR-Hash MoE only;
it will return later as an explicit comparison baseline, reimplemented
against the current architecture rather than restored as-is.

## TR-MoE block

For hidden state \(x\) and token identifier \(t\):

\[
\mathrm{TRMoE}(x,t)
=g_s\,\mathrm{Shared}(x)
+g_r\sum_{k=1}^{K}w_k\,\mathrm{Expert}_{r_{l,k}(t)}(x).
\]

- \(r_{l,k}(t)\) is a deterministic layer-specific lookup.
- The selected experts process the contextual hidden state \(x\), not an
  embedding-only representation.
- The shared path is optional in code but enabled in the principal TR-GQA and
  TR-MHA configurations.
- Gates \(g_s\) and \(g_r\) may be fixed or learned.
- No learned MoE router or auxiliary load-balancing loss is required for
  lexical routing.

See [TR-MoE internals](token-routed.md).

## Experimental routed-attention adapters

The attention registry also exposes:

- `tr_mha` / `token_routed_mha`;
- `tr_mha_v2` / `token_routed_mha_v2`.

These keep a full MHA path and add low-rank token-routed Q/V residual adapters.
They are **not** the same configuration as MHA + `TRHashEngineMLP`. The first
prototype evaluates contextual logits across all route experts; v2 restricts
contextual reweighting to two fixed token-ID candidates and starts the routed
up-projection at zero.

See [`../TR_MHA.md`](../TR_MHA.md).

## Other implemented sequence mixers

| Registry value | Status | Description |
| --- | --- | --- |
| `gqa`, `mha`, `mqa` | baseline | Standard causal attention variants |
| `lexical_gqa`, `lexical_key_gqa` | experiment | Lexical residuals around GQA |
| `causal_conv`, `causal_state_conv` | experiment | Attention-free causal convolution |
| `causal_fast_weight_conv` | experiment | Fixed-state fast-weight convolution |
| `routed_gqa` | prototype | Routed GQA implementation |

These are research alternatives and should not be presented as equivalent
evidence without a matched run.

## Historical Mu-Guidance

`use_mu_guidance=True` enables an optional contextual state passed between
layers. It remains in the framework for reproducibility and ablation work, but
it is not part of the current TR-GQA or TR-MHA definition. See
[Historical Mu-Guidance control](dynamics.md).

## Configuration invariants

- `hidden_size` must be divisible by `num_attention_heads`.
- `num_attention_heads` must be divisible by `num_key_value_heads`.
- MHA requires equal query and K/V head counts.
- `top_k` cannot exceed `num_experts`.
- TR-MoE requires `token_ids` to preserve lexical routing.
- Exact parameter matching must be checked after model construction.

## Related pages

- [TR-MoE internals](token-routed.md)
- [Getting started](getting-started.md)
- [Run configurations](run_configs.md)
- [API reference](api.md)
