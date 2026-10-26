# Training and evaluation workflows

Choose the operation below, then supply its configuration and local artifacts.
Training commands perform a preflight first; add `--run` to execute.

| Task | Entry point |
|---|---|
| Check the complete local training/resume/export workflow | `complexity train demo --output PATH --device cpu` |
| Prepare text shards or replay selections | `complexity data prepare`, `inspect`, `replay` |
| Compile native SFT prompt/completion records | `complexity data sft` |
| Train or refine an explicit text experiment | `complexity train lineage` |
| Run full-parameter SFT | `complexity train sft` |
| Run staged LoRA | `complexity train stack` |
| Evaluate or compare checkpoints | `complexity eval choices`, `compare` |
| Inspect or collect checkpoints | `complexity checkpoint inspect`, `collect` |
| Train/evaluate/export vision models | [Vision command tree](vision/index.md) |

## Configuration and checkpoint types

| Input | Consumer | Contents |
|---|---|---|
| Flat model JSON/YAML | `ModelConfig.load(path, strict=True)` | architecture for fresh weights |
| Model + training template | Python `Trainer` | architecture and optimizer settings; explicit dataloaders |
| Experiment/cluster YAML | `cf-plan-run`, `cf-plan-cluster`, owning runner | budget and runner settings |
| Native export | `ComplexityModel.from_pretrained(path)` | configuration and model weights |
| Optimization checkpoint | its original trainer | model, optimizer, schedule, cursor and RNG state |
| Preserved 1B FSDP directory | `scripts/pretrain_1b_fsdp/` | distributed shards, rank state and manifest |

Use [run configurations](run_configs.md) when authoring a new experiment.
An export initializes a new stage; an optimization checkpoint continues its
original experiment with the same data, objective and schedule.

## Prepare and inspect text data

```bash
complexity data prepare configs/examples/corpus_mixture.json artifacts/local-mixture --dry-run
complexity data sft /path/to/jsonl-dataset /path/to/new-shards \
  --tokenizer /path/to/native-agentic-tokenizer \
  --sequence-length 2048 --assistant-supervision final_assistant_only
```

The dry run plans 64 rows / 65,536 tokens without creating artifacts.
Preparation requires explicit sources and a tokenizer; SFT compilation verifies
complete targets, final EOS, source hashes and its declared supervision policy.
See [data preparation](data-preparation.md) for schemas, native tool projection,
quota recipes, replay, collection and recorded metrics.

## New local training experiment

First check the installation with `complexity train demo --output artifacts/demo-cpu --device cpu`
(or `--device mps` on a compatible Mac). The packaged demo uses
local synthetic fixtures, checks continuation against an uninterrupted run and
reloads its native export. It requires a new output directory. See the
[local workflow guide](local-training.md).

Use `complexity.training.Trainer` with explicit dataloaders for a custom
experiment. The [complete CPU example](local-training.md) performs four
diagnostic optimizer updates and explains initialization versus resume.
Batches supply `input_ids` and optional masks/labels. Explicit labels are
already next-token targets; `-100` excludes positions from the loss.

## Refinement

The text stage sequence is pretraining → refinement with a fresh optimizer →
SFT. Refinement/refit remains a separate phase from the pretraining budget.

```bash
complexity train lineage /path/to/text-lineage.json
complexity train lineage /path/to/text-lineage.json --run
```

The `complexity-text-lineage-v1` configuration declares the model profile,
stage, world size, tokenizer, pinned corpus and checkpoint directory.
Refinement also requires an initialization checkpoint, pretraining plan and
complete `unique_core` pass. Preflight checks the horizon, tail and plan hash.
Distributed execution uses the reported `torchrun` command. See
[training lineage](training.md) for these contracts and historical recipes.

## Explicit SFT workflows

Full-parameter SFT uses `complexity-sft-experiment-v1`; staged LoRA uses
`StackSFTConfig`. Config-relative artifact paths and the source stage must be
explicit. The [authoring example](../configs/examples/sft_200m_no_refinement.json)
contains placeholder paths; the [200M comparison](experiments/tr-hash-200m-refinement-ablation.md)
records the pinned dataset and reference batch geometry.

```bash
complexity train sft /path/to/experiment.json
complexity train sft /path/to/experiment.json --run --steps 10
complexity train sft /path/to/experiment.json --run --resume /path/to/output/latest
```

`--steps` is an absolute optimizer update, retaining the full LR horizon.
Resume restores optimizer, schedule, cursor and RNG with the original identity.
The default runner uses one device and accumulates the declared reference-rank
batches. CPU/MPS use FP32; CUDA permits FP32 or BF16. Full execution requires
complete held-out evaluation; `eval_max_batches` is for bounded diagnostics.

### Distributed SFT

Select DDP explicitly and launch one process per declared reference rank:

```bash
complexity train sft /path/to/experiment.json --distributed
torchrun --standalone --nproc_per_node=4 -m complexity train sft \
  /path/to/experiment.json --distributed --run
```

Use the config's actual world size. DDP averages the same per-rank loss means,
clips once per update and writes metrics/exports on rank zero. `--resume`
continues a DDP checkpoint with matching runtime, world size and identities.
`--migrate-single --resume /path/to/single/latest` explicitly initializes a **new output**
from a single-device checkpoint, retaining its scientific plan and content
hashes. Migration permits relocated identical assets; it does not promise
bitwise equivalence across runtimes. Two-rank CPU/Gloo update, migration and
resume are tested locally; CUDA/NCCL and BF16 DDP remain to be qualified.

### Staged LoRA

```bash
complexity train stack /path/to/curriculum.yaml
complexity train stack /path/to/curriculum.yaml --run \
  --start-stage general --stop-after general
```

Stages declare steps, LR and weighted JSONL sources. `source_stage` must be
`refinement` or `supervised-finetuning`. The package masks literal prompts,
supervises completion/EOS and chains merged exports with tokenizer metadata.
Python `runner.run_stage(stage, checkpoint, resume=path)` restores a stage.

## Paused 1B pretraining

The existing run retains its original `scripts/pretrain_1b_fsdp/` recipe:
four ranks × microbatch 2 × context 16,384 × 534,057 updates =
**69,999,919,104 tokens**, the whole-update realization of the **70B budget**.

```bash
complexity train inspect scripts/pretrain_1b_fsdp/recipe.json --json
complexity checkpoint inspect /path/to/complete-step \
  --recipe scripts/pretrain_1b_fsdp/recipe.json --verify --json
```

These local commands inspect files and hashes. To resume later, follow the
[original FSDP runbook](../scripts/pretrain_1b_fsdp/README.md): recover one pinned
snapshot, preserve all original assets, prove a bounded four-rank advance and
fresh model/optimizer/RNG reload, then check sustained upload and storage.
The generic Trainer and SFT runners use different checkpoint contracts.

## Evaluation

Evaluate a fixed native export in a separate process with the original
tokenizer. Multiple-choice continuation scoring reports raw and normalized
likelihood accuracy, without generating answers or adding a chat template:

```bash
complexity eval choices /path/to/checkpoint --backend torch --device cpu \
  --tokenizer /path/to/tokenizer --tasks arc_easy,arc_challenge \
  --arc-easy-samples /path/to/arc_easy.jsonl \
  --arc-challenge-samples /path/to/arc_challenge.jsonl --output /path/to/report.json
complexity eval compare /path/to/checkpoint-a /path/to/checkpoint-b \
  --tokenizer /path/to/tokenizer --tasks piqa --piqa-data /path/to/piqa \
  --output /path/to/comparison.json
```

`compare` uses the same examples, encodings and portable Torch kernel policy
for both models, reporting B−A deltas only after all tasks finish. PIQA fixtures
contain `dev.jsonl` and `dev-labels.lst`. Other supported tasks include
HellaSwag and MMLU; dataset downloads occur only when local inputs are omitted.
The `choices` default remains MLX for an MLX-compatible bundle; choose Torch
explicitly for a native checkpoint. Tiny Torch CPU fixtures are tested locally.

For held-out perplexity, use `Evaluator(model).perplexity(held_out_loader)`
with token-weighted aggregation; [the API](api.md#training-and-evaluation)
documents target alignment. Keep checkpoint/tokenizer/source revisions,
sample counts and runtime with each report. Existing [200M measurements](tr-hash-200m-release.md)
remain separate from new experiments and the paused 1B model.

## Generation and vision

Use the [Transformers adapter](../integrations/transformers/tr_hash_moe/README.md)
or [external inference client](api.md#external-inference) with a compatible
runtime. Vision uses its own [package and command tree](vision/index.md),
configuration and checkpoint formats.
