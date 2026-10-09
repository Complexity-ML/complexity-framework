# Configuration types

Choose the format for the interface that consumes it. Model architecture,
training-loop settings, a runner's arguments and a preserved FSDP recipe are
separate contracts; a file extension alone does not identify them.

| Format | Shape / example | Consumer | Scope |
|---|---|---|---|
| Model architecture | flat JSON/YAML; `configs/examples/tr_gqa_cpu.json` | `ModelConfig.load(path, strict=True)` | new model shape; no weights/data |
| Model + training template | `model:` and `training:` mappings | `complexity train validate`; explicit Python `Trainer` | field validation; dataloader still supplied |
| Corpus preparation / replay | explicit source/layout or budget/pass/alignment mapping | `complexity data prepare` / `replay` | new local token shards or zero-copy selection metadata |
| Native SFT preparation | `format: complexity-sft-preparation-v1`, explicit split/source quotas | `prepare_sft_dataset`; `complexity data sft` compiles local JSONL | projected local records; historical 100K recipe is separate from the unresolved 200K assertion |
| Text lineage | `format: complexity-text-lineage-v1`, profile/stage/world size/settings | `complexity train lineage CONFIG` | audited explicit plan; execution requires `--run` and the declared world size |
| Full-parameter SFT experiment | `format: complexity-sft-experiment-v1` plus local source/data and `execution` | `complexity train sft CONFIG` | marked direct-SFT ablation; execution requires `--run` |
| Staged LoRA curriculum | `StackSFTConfig` fields plus named `stages` | `complexity train stack CONFIG` | explicit source stage and file-backed mixes; execution requires `--run` |
| Runner configuration | flat arguments or a `run:` mapping | selected runner's `--config` parser | that runner's flags, defaults and artifact guards |
| Cluster plan | `model:`, `parallel:`, `run:` and world size | `cf-plan-cluster --config` | topology and token arithmetic only |
| Preserved 1B FSDP recipe | `scripts/pretrain_1b_fsdp/recipe.json` | original FSDP runner; `train inspect` for local evidence | immutable model/data hashes, four-rank state and 70B horizon |
| Model provenance collection | `format: complexity-artifact-collection-v1` plus local source/output paths | `complexity checkpoint collect CONFIG` | model-only export with hashes; no training-state resume or publication |

A generic model loader must not silently reinterpret a run config or the 1B
recipe. The CLI reports the appropriate consumer when a format is unsuitable.

## Author a new model

```bash
complexity train validate configs/examples/tr_gqa_cpu.json --json
```

```python
from complexity import ModelConfig

config = ModelConfig.load("configs/examples/tr_gqa_cpu.json", strict=True)
config.save("/tmp/my-model.json")
```

Use canonical names: `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads` and `max_position_embeddings`. Strict authoring rejects
unknown fields and validates shape/routing invariants. The default historical
loader still tolerates extra adapter metadata when loading existing
checkpoints; strict authoring does not rewrite those artifacts.

For a released model, load the `config.json` supplied with its weights and
retain tokenizer IDs and trained route tables. Presets and model configs
construct random weights; they are not pretrained artifacts.

## Author a training template

```bash
complexity train create-config /tmp/my-training.yaml --size tiny
complexity train validate /tmp/my-training.yaml --json
```

The template separates `ModelConfig` fields from `TrainingConfig` fields.
Validation does not infer a dataset, create loaders or launch a job. Optional
`data`, `checkpoint` and `logging` mappings are script-owned metadata; CLI
validation does not prove their paths exist or implement their behavior.

Older CLI aliases are accepted with explicit migration warnings; new templates
use canonical fields. Pass validated model/training sections to the reusable
Python APIs with a real loader as shown in [workflows](workflows.md).

## Runner-specific YAML

Runners using `parse_args_with_yaml_config` accept flat defaults or a top-level
`run` mapping. Keys may use underscores or hyphens; they normalize to the
owning parser's destinations. CLI flags take precedence. Unknown destinations,
duplicate normalized keys and invalid types/choices are rejected.

A `run:` mapping is not a `ModelConfig`. Inspect the selected runtime's `--help`
to find its actual fields; an old YAML for a removed runtime cannot be launched
by a different runner merely because both accept `--config`.

Vision uses one task tree, for example:

```bash
complexity vision train detector -- --help
complexity vision train image -- --help
```

The separator forwards arguments to that runtime. Not all vision runners use
the same configuration schema; see [vision workflows](vision/index.md).

## Preserved paused 1B recipe

```bash
complexity train inspect scripts/pretrain_1b_fsdp/recipe.json --json
```

The preserved recipe targets 534,057 updates, four ranks, microbatch two and
context 16,384: 131,072 tokens/update and **69,999,919,104 tokens**, the
whole-update realization of the 70B-token budget.

Recipe inspection checks locally available linked assets and reports missing
evidence. It does not allocate weights, restore distributed state, regenerate
a schedule or launch training. Keep the original `model.json`, data plan,
schedule, source offsets and tokenizer with the recipe. Exact continuation
uses the [original FSDP runbook](../scripts/pretrain_1b_fsdp/README.md),
not a newly authored template or a planner's suggested budget.

## Plan arithmetic without training

`cf-plan-run` computes token/update arithmetic from explicit values:

```bash
cf-plan-run --tokens 70B --gpus 4 --batch-size 2 --seq-len 16384 --save-steps 0
```

This is an arithmetic illustration, not the preserved run's authoritative
schedule. Whole-update rounding may differ from the fixed FSDP recipe above.
Throughput/ETA estimates require a representative measured rate; model shape
or GPU count does not supply one.

The historical cluster plan below can be checked for TP × PP × DP topology,
global batch, target tokens and overshoot:

```bash
cf-plan-cluster --config configs/run_configs/8b_o200k_tr_32t_gb300_4608.yaml
```

Neither planner launches training, creates a rental or deploys a scheduler.
The cluster file is a planning artifact, not a demonstrated deployment.

## Historical release and research plans

The 200M replay/refinement profiles and the removed o200k Dense/TR comparison
YAMLs retain historical provenance. Several named 200M executables are absent
from this checkout; their former names are recorded in [training
lineage](training.md). `cf-o200k-pretrain` and `cf-check-pipeline` are removed.
Do not present these plans as runnable current entry points.

A configuration records intent until a matching artifact identifies resolved
settings, code revision, checkpoint and dataset hashes, tokenizer, hardware,
precision, selected kernels, measured metrics and checkpoint-selection rule.
See [workflows](workflows.md) for the validation boundaries.
