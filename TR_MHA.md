# Token-routed MHA adapter experiments

This page documents the attention registry values `tr_mha` and `tr_mha_v2`.

> Naming note: the main README uses **TR-MHA** for the complete
> `attention_type="mha"` + `mlp_type="token_routed"` architecture. The
> components below are a separate experimental branch that routes low-rank
> residual adapters inside MHA.

The matched result for the complete MHA + TR-MoE architecture is reported in
[`RESULTS_100M_MPS.md`](RESULTS_100M_MPS.md). It must not be attributed to the
attention-adapter prototypes documented on this page.

## Shared MHA path

Both adapter variants preserve standard full MHA:

```text
shared Q/K/V projections ─► causal attention ─► output projection
             │
             └─ token-routed low-rank residual applied to Q and/or V
```

Every query head keeps its own K/V head, so
`num_key_value_heads == num_attention_heads` is required.

## `tr_mha`

The first prototype:

- builds two layer-specific candidates from token ID;
- computes a contextual score over all route experts;
- combines fixed prior and contextual verification logits;
- selects top-k low-rank Q and V adapters;
- adds their weighted deltas to the shared MHA projections.

The routed branch is not guaranteed to be neutral at initialization.

## `tr_mha_v2`

The second prototype is baseline-preserving:

- the full MHA path is unchanged at initialization;
- the routed up-projection is initialized to zero;
- token ID fixes the two candidate experts;
- context only reweights those candidates;
- only selected expert/token pairs are evaluated;
- `tr_mha_targets` chooses `q`, `v`, or `qv`.

## Configuration

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=384,
    num_hidden_layers=10,
    num_attention_heads=8,
    num_key_value_heads=8,
    attention_type="tr_mha_v2",
    mlp_type="swiglu",
    intermediate_size=1440,
    vocab_size=200_019,
    tr_mha_num_experts=4,
    tr_mha_adapter_rank=4,
    tr_mha_top_k=2,
    tr_mha_targets="qv",
)
model = ComplexityModel(config)
```

Tracked MPS configurations:

- `configs/run_configs/experiments_100m/100m_params_tr_mha_qv_mps.yaml`
- `configs/run_configs/experiments_100m/100m_params_tr_mha_v2_mps.yaml`
- `configs/run_configs/experiments_100m/100m_params_tr_mha_v2_v_only_mps.yaml`

## Telemetry

The training runner can report:

- adapter output gate;
- contextual verifier strength and use;
- route entropy;
- per-expert route shares.

## Status

These are bounded attention experiments. They are not the default pretraining
architecture and should not inherit claims from MHA + TR-MoE pilots.
