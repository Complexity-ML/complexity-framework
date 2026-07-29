# Run configurations

The repository separates direct runner configurations, bounded experiments, and
cluster plans.

## Direct runner YAML

`cf-o200k-pretrain` accepts YAML defaults:

```bash
cf-o200k-pretrain \
  --config configs/run_configs/100m_o200k_tr_rocm_mi350x.yaml
```

The runner reads the top-level `run` mapping. CLI flags override YAML values:

```bash
cf-o200k-pretrain \
  --config configs/run_configs/100m_o200k_tr_rocm_mi350x.yaml \
  --steps 10 \
  --run-name smoke-override
```

Unknown keys fail immediately rather than being silently ignored.

## Architecture mapping

### TR-GQA

```yaml
run:
  attention_type: gqa
  mlp_type: token_routed
  num_attention_heads: 8
  num_key_value_heads: 2
  shared_expert: true
  routing_strategy: zipf
  top_k: 2
```

### TR-MHA

```yaml
run:
  attention_type: mha
  mlp_type: token_routed
  num_attention_heads: 8
  num_key_value_heads: 8
  shared_expert: true
  routing_strategy: modulo_balanced_secondary
  top_k: 2
```

### Dense controls

```yaml
run:
  attention_type: gqa  # or mha
  mlp_type: swiglu
  shared_expert: false
```

Match parameter counts explicitly by adjusting `intermediate_size` and
`shared_intermediate_size`.

## Routing strategies

| Value | Route source | Frequency artifact |
| --- | --- | --- |
| `zipf` | lexical lookup | used when available; otherwise modulo fallback |
| `modulo` | token ID | not required |
| `modulo_balanced_secondary` | modulo primary, greedy auxiliary | used when available |
| `round_robin` | frequency rank or token ID | optional |
| `random` | fixed seeded lexical partition | not required |
| `lsh_hidden` | hidden-state hash | not lexical routing |

The parser accepts only these values. Older documentation mentioning
`zipf_token_class` is obsolete.

## Profiles

The local runner defines `50m`, `100m`, `200m_32k`, `300m`, `1b`, and `8b`
profiles.
Profile names are approximate. The realized parameter count depends on
vocabulary size and explicit overrides and is recorded at launch.

The matched 200,082,688-parameter Dense/TR protocol, frozen 4B-token dataset
preparation, B200 launcher, artifact export, and server teardown checklist are
documented in [the 200M B200 runbook](200m-32k-b200-runbook.md).

## Experimental configurations

`configs/run_configs/experiments_100m` contains bounded MPS and architecture
experiments. Several files contain absolute local dataset paths. Treat them as
reproducibility records and update paths before launch.

`configs/run_configs/review_h200` contains matched review and evidence runs.

`configs/run_configs/ablations_100m` contains routing controls.

## Cluster plans

Cluster plans with top-level `model`, `parallel`, and `run` sections are
validated with:

```bash
cf-plan-cluster \
  --config configs/run_configs/8b_o200k_tr_32t_gb300_4608.yaml
```

The planner checks TP × PP × DP world size, global batch, steps, target tokens,
and overshoot. It does not launch Slurm, Kubernetes, Ray, or vendor jobs.

Do not pass a cluster-plan YAML directly to `cf-o200k-pretrain`.

## Resume contract

At first launch, the runner writes:

```text
runs/<run-name>/run_config.json
```

On resume it rejects differences in training-critical arguments,
`ModelConfig`, and frozen token-shard identities. Operational fields such as
logging cadence, evaluation cadence, save cadence, and output names may change.

Use `--force-resume` only when the mismatch is understood and documented.
