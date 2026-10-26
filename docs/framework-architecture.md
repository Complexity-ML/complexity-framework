# Framework architecture

The public package provides reusable model, configuration, training and
checkpoint interfaces. Concrete runners bind these components to a dataset,
distributed topology and an experiment's immutable contract.

## Package boundaries

| Module | Responsibility | Start with |
|---|---|---|
| `complexity.config` | validated architecture configuration and presets | `ModelConfig` |
| `complexity.models` | decoder construction, forward pass and native weight exports | `ComplexityModel` |
| `complexity.core` | attention, normalization, position and MLP components/registries | component classes and decorators |
| `complexity.tr_hash` | deterministic route tables and dispatch backends | `TRHashEngine`, `TRHashEngineConfig` |
| `complexity.training` | data preparation/replay, optimizer/update loop and training checkpoints | `Trainer`, `TrainingConfig`, `complexity data` |
| `complexity.evaluation` | held-out metrics and continuation scoring | `Evaluator`, `EvalConfig`, `zero_shot` |
| `complexity.export` | native model provenance, tokenizer assets and autonomous Transformers bundles | `provenance`, `transformers` |
| `complexity.inference` | clients for external OpenAI-compatible generation services | `create_external_backend` |
| `complexity.vision` | grouped vision model/training/export interfaces | [vision guides](index.md#multimodal-and-vision-guides) |
| `complexity.api` | convenience wrappers over canonical components | `Model`, convenience `Trainer` |
| `complexity.cli` | discoverability, validation, local inspection and task commands | `complexity --help` |

Begin with `from complexity import ComplexityModel, ModelConfig`. Import the
training, evaluation, vision or tooling modules for the task at hand. Optional
vision/MCP/serving dependencies should not be required for a text model's
basic construction.

## Model architecture versus framework architecture

A decoder is embeddings, contextual attention, shared/routed feed-forward
blocks, final normalization and a vocabulary head. **TR-GQA** and **TR-MHA**
name complete decoder combinations, while **TR-MoE** names their common MLP.
See [architecture and naming](architectures.md) for tensor widths, routes and
invariants; [TR-Hash engine](tr-hash-engine.md) documents dispatch backends.

Semantic/contextual computation remains in the hidden states and attention.
Lexical token identity selects deterministic per-layer expert routes. This
implemented mechanism is not evidence that less training data yields the same
quality; that hypothesis needs controlled research comparisons.

## Configuration boundaries

Architecture fields belong to `ModelConfig`; optimizer/update/checkpoint
fields belong to `TrainingConfig`. New authoring should use canonical field
names and strict loading. Historical checkpoint metadata may include adapter
keys, so the existing default loader remains tolerant for compatibility.

A run-planning YAML describes a runner's arguments and execution budget; it is
not a `ModelConfig` and must not be silently translated into a different
training recipe. The preserved 1B `recipe.json` additionally ties model/data
hashes, world size and scheduler horizon to checkpoint state. See
[configuration guide](run_configs.md).

## Checkpoint boundaries

| Format | API | Resume optimization? |
|---|---|---|
| Native `config.json` + weights | `save_pretrained` / `from_pretrained` | no, weights only |
| Generic trainer state | original `Trainer` checkpoint manager | model, optimizer, scheduler and saved state |
| 1B FSDP distributed step | original `scripts/pretrain_1b_fsdp/` loader | model/optimizer shards, cursor and per-rank RNG |
| Transformers custom-model bundle | external adapter | inference/evaluation; not an FSDP resume |

Preserve tokenizer IDs, tensor names and trained route buffers together. An
inspector can check metadata and file integrity without loading a model; that
is weaker evidence than an actual distributed restore and round trip.

## Runners and research scope

`scripts/pretrain_1b_fsdp/` remains the owner of the existing paused 1B run.
The framework refactor must keep its model shape, recipe hash, original corpus
schedule and 70B-token budget compatible. Launching training and uploading or
mutating published checkpoints are separate operational actions.

Vision is grouped under one public tree. Existing `complexity.generative`
imports remain compatibility surfaces where required; text, image generation,
audio and sensor research do not share interchangeable checkpoint formats.
Released evidence and experimental features are labelled in the
[documentation index](index.md).

The separate 200M release and older comparisons retain their original
provenance. Useful preparation, training and evaluation responsibilities belong
in task modules inside the framework. One-off release launchers retain their
historical recipes; their absence does not make a reusable responsibility
obsolete. See [refactor status](refactor-status.md) for unresolved contracts.

## Extending the framework

Register a new attention or MLP component through `complexity.core.registry`,
keep its configuration explicit, and state whether it is an experiment or a
checkpoint-compatible extension. Changing expert capacity, routing or
optimizer grouping can alter behavior or resume state; a passing small-model
forward pass does not qualify those changes for an existing large run.

See [custom models](custom-models.md), [public API](api.md) and
[workflows](workflows.md) for the task interfaces.
