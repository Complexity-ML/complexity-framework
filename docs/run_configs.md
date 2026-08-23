# Run configurations and launch profiles

The repository contains three kinds of run description. Treating them as one
format causes incorrect launch claims.

## 1. Current 200M production launchers

The released language-model lineage is encoded in shell launchers plus the
Python entry points they call:

| Operation | Launcher | Python entry point |
|---|---|---|
| Replay pretraining | `scripts/vast_pretrain_tr_hash_200m_70b_replay.sh` | `scripts.train_tr_hash_200m_200b` |
| Unique-token refinement | `scripts/vast_finetune_tr_hash_200m_70b_unique_phase2.sh` | `scripts.train_tr_hash_200m_200b` |
| Recompile SFT text for vocab 32,004 | — | `scripts.recompile_tr_hash_sft_32004` |
| Tokenize and audit SFT v3 | — | `scripts.tokenize_tr_hash_sft_32004` |
| Three-epoch full SFT v3 | `scripts/vast_sft_200m_32004_full_3e.sh` | `scripts.sft_tr` |
| Package/publish dataset release | — | `scripts.package_tr_hash_sft_32004_release` / `scripts.publish_tr_hash_sft_32004_dataset` |

The launchers resolve environment variables, validate manifests and exact
token counts, and then invoke DDP. They are production profiles for the
`/workspace` layout, not portable YAML specifications. See
[Training](training.md).

The refinement launcher uses the committed immutable plan
`configs/replay_plans/tr_hash_70b_unique_only_phase2.json`. It loads base model
weights only and starts a fresh optimizer and scheduler. The runtime compares
the exact `unique_core` fingerprint against the base plan; a matching token
count with different source rows is rejected. All text SFT launchers declare
their checkpoint provenance through `--source-stage`.

## 2. Historical o200k and cluster-plan YAMLs

Files under `configs/run_configs/` describe removed Dense/TR comparison runs or
cluster arithmetic. The `cf-o200k-pretrain` launcher no longer exists. Do not
present those YAMLs as runnable training jobs.

`cf-plan-run` still computes token/step arithmetic:

```bash
cf-plan-run \
  --tokens 130B \
  --gpus 8 \
  --batch-size 8 \
  --seq-len 1024 \
  --tok-s 600000 \
  --save-steps 0
```

`cf-plan-cluster` validates TP × PP × DP topology, global batch, target tokens,
and overshoot for a cluster-plan YAML:

```bash
cf-plan-cluster \
  --config configs/run_configs/8b_o200k_tr_32t_gb300_4608.yaml
```

Neither planner launches Slurm, Kubernetes, Ray, a cloud rental, or training.

## Canonical new-model values

```yaml
run:
  attention_type: gqa
  mlp_type: tr_hash_engine
  routing_strategy: token_id_multi_hash
  route_hash_count: 2
  num_experts: 4
  top_k: 2
  shared_expert: true
```

For the exact released 200M shape, use
`scripts.train_tr_hash_200m_200b.make_config`; do not reconstruct it from this
minimal routing fragment.

Legacy `token_routed` checkpoints require conversion; see
[TokenRoutedMLP migration](token-routed.md).

## Reproducibility rule

A configuration records intent until a matching artifact records the resolved
configuration, code commit, dataset identity and revision, tokenizer, hardware,
precision, kernel path, metrics, checkpoint-selection rule, and file hashes.
Configuration files alone are not completed experiments.
