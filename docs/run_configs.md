# Run configurations

The repository contains two different configuration classes. They must not be
confused:

1. current SFT curriculum YAMLs consumed by `scripts/run_sft_curriculum.py`;
2. historical pretraining and cluster-plan YAMLs retained for evidence and
   resource arithmetic.

## Current SFT curricula

The active Card Corpus V2 profiles are:

| File | Purpose |
|---|---|
| `configs/sft_500m_32k_v2_balanced.yaml` | one full-shard LoRA epoch with 2D loss weighting |
| `configs/sft_500m_32k_v2_balanced_continuation.yaml` | two additional full passes from a selected model checkpoint |

Resolve a profile without training:

```bash
python -m scripts.run_sft_curriculum \
  --checkpoint /path/to/checkpoint \
  --sft-bin /path/to/tokenized/32k-v2 \
  --curriculum-config configs/sft_500m_32k_v2_balanced.yaml \
  --through-stage full-shard-weighted \
  --output-root artifacts/sft-plan \
  --tokenizer tokenizer \
  --world-size 8 --batch-size 24 --lora-rank 32 --lora-alpha 32 \
  --dry-run
```

The dry-run verifies full-shard exposure, visible supervised-token counts,
group targets, nested task targets, and the maximum task coefficient. See
[Two-dimensional full-shard SFT weighting](sft-full-shard-2d-weighting.md).

## Token-budget planner

`cf-plan-run` computes step and checkpoint arithmetic from a target token
budget. It does not load a model or launch training:

```bash
cf-plan-run \
  --tokens 20B \
  --gpus 8 \
  --batch-size 24 \
  --seq-len 512 \
  --tok-s 42000 \
  --save-steps 500
```

## Cluster-plan validator

Cluster plans with top-level `model`, `parallel`, and `run` sections are
validated with:

```bash
cf-plan-cluster \
  --config configs/run_configs/8b_o200k_tr_32t_gb300_4608.yaml
```

The validator checks TP × PP × DP world size, global batch, steps, target
tokens, and overshoot. It does not launch Slurm, Kubernetes, Ray, or a vendor
job.

## Historical pretraining YAMLs

Files below `configs/run_configs/` whose `run` mappings describe o200k
pretraining are preserved as experiment records. The `cf-o200k-pretrain`
driver, Dense baseline, learned-router baseline, and several routing strategies
used by those records were removed. Do not pass those files to a current
training command or describe them as runnable.

The following values are canonical for newly constructed TR-Hash models:

```yaml
run:
  attention_type: gqa        # use mha with equal Q and KV head counts for TR-MHA
  mlp_type: tr_hash_engine
  routing_strategy: token_id_balanced_hash  # or modulo_cyclic
  num_experts: 4
  top_k: 2
  shared_expert: true
```

Legacy `token_routed` checkpoints require conversion; see
[TokenRoutedMLP migration](token-routed.md). Historical `swiglu`, `mixtral`,
`zipf`, `round_robin`, `random`, and hidden-state LSH configurations are not
canonical current TR-Hash inputs.

## Reproducibility rule

Treat a tracked YAML as a claim about intent until a matching artifact records
the resolved configuration, code commit, dataset identity, hardware, metrics,
and selected checkpoint. Configuration files alone are not completed
experiments.
