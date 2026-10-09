# Public API

Use the native model/config API for reproducible research and import task
modules as needed. Concrete runners own dataloaders, distributed launch and
artifact-specific checkpoint restoration. The [framework architecture](framework-architecture.md)
explains these boundaries.

## Model configuration and construction

```python
from complexity import ComplexityModel, ModelConfig, get_preset

config = ModelConfig.load("configs/examples/tr_gqa_cpu.json", strict=True)
model = ComplexityModel(config)
```

`get_preset(name)` and `ComplexityModel.from_preset(name)` construct fresh
weights; a preset is not a released checkpoint. To inspect or recreate a
released shape, load its own `config.json`. The paused 1B run's exact shape is
in `scripts/pretrain_1b_fsdp/model.json`.

| Configuration method | Purpose |
|---|---|
| `ModelConfig(**fields)` | author architecture directly; validates invariants |
| `ModelConfig.from_dict(data, strict=True)` | reject unknown fields in a new architecture |
| `ModelConfig.load(path, strict=True)` | load a flat model JSON/YAML with strict authoring |
| `config.save(path)` / `config.to_dict()` | serialize architecture fields |
| Default `from_dict` / `load` | retain historical tolerance of checkpoint/adapter metadata |

A nested model+training template or experiment YAML is not a flat model
configuration. See [configuration types](run_configs.md).

### Principal model fields

| Field | Meaning |
|---|---|
| `hidden_size`, `num_hidden_layers` | residual width and decoder depth |
| `vocab_size`, `max_position_embeddings` | tokenizer vocabulary and context bound |
| `attention_type` | `gqa`, `mha` or `mqa` for the common decoder path |
| `num_attention_heads`, `num_key_value_heads` | query and K/V heads |
| `mlp_type` | canonical TR-MoE key `tr_hash_engine` |
| `num_experts` | stored routed expert count |
| `intermediate_size` | total stored routed width, divided by `num_experts` |
| `shared_expert`, `shared_intermediate_size` | dense shared branch and its width |
| `routing_strategy`, `route_hash_count` | deterministic route-table construction |
| `top_k`, `top_k_primary_weight` | deterministic expert routes and their blend |
| `shared_output_scale`, `routed_output_scale` | shared/routed residual scales |
| `use_custom_kernels`, `use_cggr` | dispatch policy; PyTorch remains the reference |
| `collect_moe_telemetry` | opt-in route/RMS diagnostics |

Canonical routing values are `modulo_cyclic`, `token_id_balanced_hash`,
`token_id_multi_hash` and `token_id_hierarchical_hash`. The source config
contains additional experimental fields. Changing routing/capacity in a
trained model is an experiment, not an automatic checkpoint-safe upgrade.

## Forward pass

```python
result = model(
    input_ids,
    attention_mask=None,
    past_key_values=None,
    use_cache=False,
    return_hidden_states=False,
    return_logits=True,
)
```

| Return key | Value |
|---|---|
| `logits` | `[batch, sequence, vocabulary]`, or `None` |
| `last_hidden_state` | final normalized hidden states |
| `past_key_values` | optional per-layer cache/state list |
| `hidden_states` | optional embedding and layer states |

Set `return_logits=False` for fused/chunked tied-head loss paths. The model
passes original token IDs to TR-MoE layers; selected experts transform the
current contextual states.

## Native weight exports

```python
model.save_pretrained("artifacts/model-export")
restored = ComplexityModel.from_pretrained("artifacts/model-export")
```

Native exports contain `config.json` and weights (`model.safetensors` by
default, or `pytorch_model.bin`). Loading requires actual compatible weights;
configuration alone does not qualify a checkpoint. Persisted route buffers
must stay with their trained expert weights. DTensor/FSDP export is collective:
every participating rank must enter the save call. A replicated DDP model can
be exported on rank zero alone. Failed DTensor gathering refuses to export a
local shard. Serialization prepares temporary files before replacing the
canonical files, and changing formats removes the former weight file. A commit
error restores the previous files; if restoration itself fails, backups are
preserved for recovery. In-place exports are not a crash-atomic directory swap.
Live routing/capacity controls are saved in the configuration, including
`layer_mlp_controls` when layers differ, and native reload restores them. The
Transformers exporter rejects per-layer controls it does not implement.

This format is weights only. Generic Trainer optimization checkpoints and the
preserved 1B distributed checkpoints require their respective original
loaders. See [workflows](workflows.md#configuration-and-checkpoint-types).
Generic Trainer checkpoints also record live MLP controls; exact resume and
model-only collection preserve them without adding model tensors.

## Training and evaluation

```python
from complexity.training import Trainer, TrainingConfig
from complexity.evaluation import EvalConfig, Evaluator
```

| Interface | Contract |
|---|---|
| `Trainer(model, config, train_dataloader, eval_dataloader=None)` | explicit reusable optimizer/update loop |
| `trainer.train()` | run this configured experiment |
| `TrainingConfig.resume_from` | restore this trainer's saved optimization state |
| `TrainingConfig.init_checkpoint` | weights-only initialization for a new stage |
| `Evaluator(model, tokenizer=None, config=None)` | model evaluation; loader still supplied explicitly |
| `evaluator.perplexity(dataloader)` | token-weighted finite-loader causal loss/perplexity |
| `EvalConfig.labels_are_shifted` | `True` by default for explicit next-token labels; `False` for full labels aligned with input IDs |

`TrainingConfig` uses `batch_size` per rank, `gradient_accumulation_steps`,
`max_steps`, `learning_rate`, `precision`, `grad_clip`, `save_steps`,
`eval_steps`, `checkpoint_dir` and `log_dir`. The [workflows guide](workflows.md)
links a [complete CPU example](local-training.md), resume boundaries and
evaluation requirements. Explicit labels supplied to the training
loop are already next-token targets (`S` or `S-1` positions); without labels,
it derives targets from `input_ids[:, 1:]`. Mask ignored positions with `-100`.
Evaluation uses the same default. Set `EvalConfig(labels_are_shifted=False)`
for a loader whose full label sequence is aligned with its input IDs.

### Convenience wrappers

```python
from complexity.api import Model, Trainer as SimpleTrainer, TrainerConfig

wrapped = Model.create(config)
wrapped.save("artifacts/wrapped-model")
loaded = Model.load("artifacts/wrapped-model")
```

`Model` wraps the native decoder. `Model.load` supports complete local native
exports, compatible legacy `model.pt` folders, or named presets with fresh
weights; it does not download a Hub model implicitly. Unknown overrides and
missing checkpoint weights raise errors.

`SimpleTrainer(wrapped, config=TrainerConfig(...)).train(dataset_or_loader)`
delegates to the reusable trainer. `save(path)` exports model weights;
`save_checkpoint(tag="manual")` saves resumable optimization state after a
trainer has been initialized. For an existing FSDP run, use its preserved
runner instead of either convenience wrapper.

## Task-specific SFT

`complexity.training.SFTTokenShardDataset`, `build_sft_plan` and
`execute_sft_config` implement the marked full-parameter ablation with the
v2/32K pre-shifted target contract. `complexity.training.stack_sft` implements
file-backed staged LoRA. Use `complexity train sft CONFIG` or
`complexity train stack CONFIG` for preflight, and add `--run` explicitly for
execution. `complexity.training.sft_distributed` adds explicit DDP execution
under `torchrun`, exact DDP resume and marked single-device migration.
See [SFT workflows](workflows.md#explicit-sft-workflows) for device,
resume and source-stage boundaries.

`sft_weighting.WeightedSFTTokenShardDataset` composes the v2 reader with
audited task weights and sidecar metadata. `sft_evaluation` provides coverage
checks, partition discovery and validation-state policies.
`export.native_checkpoint.load_native_checkpoint` normalizes explicit legacy
SafeTensors aliases and verifies trained routes and dimensions. These are
caller-selected APIs; they do not automatically enable weighting, early
stopping or another loader in the frozen direct-SFT runner.

## Corpus preparation and replay

```python
from complexity.training import (
    TextCorpusSource,
    TokenShardWriter,
    prepare_corpus_mixture,
    PretokenizedCorpusMixtureDataset,
    build_replay_plan,
    build_corrective_replay_plan,
)
```

`prepare_corpus_mixture` writes quota-controlled uint16 shards/manifests from
explicit sources and a supplied tokenizer. `TokenShardWriter` is the streaming
primitive. `PretokenizedCorpusMixtureDataset` consumes local manifests or a
pinned, lazily cached Hub dataset. Replay functions return zero-copy selection
and exposure plans; all budgets, passes and alignment are explicit inputs.
These helpers have no default training launch, fixed corpus or upload target.
Use `complexity data prepare`, `inspect` and `replay`; [workflows](workflows.md#prepare-and-inspect-text-data)
links their [config schemas](data-preparation.md) and a dry run.

`complexity.training.sft_preparation` owns native tool/final projection,
`encode_agentic_trajectory`, `prepare_sft_dataset` and `compile_sft_dataset`.
Quotas are supplied in a recipe, and shard compilation requires an explicit
assistant-supervision policy. `complexity.training.lineage_runner` owns audited
text pretraining/refinement config execution. See [workflows](workflows.md) for
the source, loss-policy and paused-run boundaries.

`agentic_corpus`, `agentic_sharding`, `agentic_candidates` and `agentic_plans`
provide source filtering, candidate staging, verified shard publication and
curriculum plans. They accept explicit manifests, tokenizer contracts and
publisher callbacks; local tests qualify sequential execution. `sft_examples`
provides reusable projections and bounded synthetic examples. Their historical
quota/source profiles are in `configs/examples`; profiles do not acquire or
publish data automatically.

`complexity.export.provenance.collect_run_artifacts` and `collect_from_config`
collect local model-only weights and selected metadata into a new folder.
`complexity.training.tensorboard.export_recorded_metrics` exports CSV/JSONL
scalars; `mirror_once` supports a caller-owned writer. Their task commands are
`complexity checkpoint collect CONFIG` and
`complexity jobs metrics FILE --logdir DIR`.

## External inference

`ComplexityModel.generate()` intentionally raises `RuntimeError`. Use an
external serving client or a compatible Transformers custom-model bundle.

```python
from complexity.inference import ExternalGenerationConfig, create_external_backend

backend = create_external_backend(
    "vllm",
    base_url="http://localhost:8000",
    model="my-model",
)
answer = backend.chat(
    [{"role": "user", "content": "Summarize the experiment."}],
    ExternalGenerationConfig(max_tokens=128),
)
```

The synchronous, non-streaming client accepts `vllm` or `sglang` labels and
calls an OpenAI-compatible endpoint. A label selects the HTTP adapter, not
model architecture support. The separate [TR-Hash-i64](https://github.com/Complexity-ML/TR-Hash-i64)
service implements the released model's routes. The [Transformers adapter](../integrations/transformers/tr_hash_moe/README.md)
documents custom-model packaging and diagnostic generation; neither interface
implies native upstream runtime support.

### Local Transformers export

```bash
python -m complexity.export.transformers \
  --config /path/to/config.json \
  --weights /path/to/model.safetensors \
  --tokenizer-dir /path/to/tokenizer \
  --output /path/to/separate-bundle
```

The exporter packages custom-model code and tokenizer files in a separate local
directory; it does not publish or reconstruct FSDP shards. It preserves tensor
bytes while adding SafeTensors framework metadata in the exported header.

## Components and vision

Advanced extensions use `complexity.core.registry` decorators and registries.
The canonical MLP is `TRHashEngineMLP`, backed by `complexity.tr_hash.TRHashEngine`.
Its diagnostics are `capability_summary()`, `training_telemetry()` and
`layer.engine.last_backend`. See [custom models](custom-models.md) and
[engine internals](tr-hash-engine.md).

`complexity.profiling.routing` exposes stratified token sampling, actual
weighted expert contributions and full-probe route/norm audits without plotting
dependencies. Diagnostics respect the layer's current capacity and mix weights.

`tr_hash_moe` is a compatibility alias of `tr_hash_engine`. Experimental
attention keys `tr_mha`/`tr_mha_v2`, lexical modules and `dense_deterministic`
controls do not inherit released checkpoint measurements. Removed
`token_routed` checkpoints use the explicit [conversion guide](token-routed.md).

Vision interfaces are grouped under `complexity.vision`; [the vision guide](vision/index.md)
indexes one command family: `complexity vision train`, `pretrain`, `evaluate`,
`export`, `generate`, `edit` and `serve`. Use `complexity vision list` for tasks
and `complexity vision train detector -- --help` for runtime arguments.
Compatibility imports do not create a second model implementation.

## CLI commands

From a source checkout, `python -m complexity ...` runs the same subcommands
from that tree, avoiding an older installed console script. The common CLI's
local checks do not start a training job:

```bash
complexity train validate configs/examples/tr_gqa_cpu.json --json
complexity train create-config /tmp/new-training.yaml --size tiny
complexity train inspect scripts/pretrain_1b_fsdp/recipe.json --json
complexity checkpoint inspect /path/to/complete-step --verify --json
```

`train validate` handles flat model configs and model+training templates;
`train inspect` also understands the preserved FSDP recipe and its asset
contract. `checkpoint inspect` reports FSDP manifest/state integrity without
restoring tensors; `--recipe PATH` additionally checks against the original
recipe. `--verify` hashes files and can take time for a multi-gigabyte step.

Generic `train run`/`train resume` require the original dataloader and runner;
they do not infer a dataset or substitute a new recipe. `--help` describes
conversion, external inference, tokenization, profiling and managed job tasks.

All packaged console entry points:

| Entry point | Role |
|---|---|
| `complexity` | common CLI and task discovery |
| `cf-plan-run` | experiment token/update budget validation |
| `cf-plan-cluster` | cluster-sizing arithmetic |
| `cf-image-train` | image-generation training |
| `cf-image-edit-train` | image-editing training |
| `cf-detector-train` | detector training |
| `cf-vision-pretrain` | vision pretraining |
| `cf-detector-serve` | detector HTTP service |
| `cf-sensor-fusion-train` | sensor fusion training |
| `cf-sensor-fusion-submit` | sensor fusion submission artifacts |

`cf-o200k-pretrain` and `cf-check-pipeline` are removed historical commands.
The remaining family-specific commands use their own configs and dependency
extras; they do not launch or resume the preserved text 1B run.

## MCP tooling

Install the optional `[tools]` extra and import from `complexity.mcp`:
`MCPTool`, `MCPToolResult`, `OfficialMCPStdioClient` and `OfficialMCPStdioConfig`.
The wrapper calls servers through the official Python SDK stdio transport.
