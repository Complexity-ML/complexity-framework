# Complexity Framework documentation

Use the framework guides for local development and the release pages for
checkpoint-specific evidence. The paused 1B run and the released 200M lineage
have separate recipes and checkpoint formats.

## Start with your task

| Task | Guide |
|---|---|
| Install, validate the environment and run a CPU forward pass | [Getting started](getting-started.md) |
| Find public interfaces and module ownership | [Framework architecture](framework-architecture.md), [API reference](api.md) |
| Choose the correct configuration type | [Run configurations](run_configs.md) |
| Prepare data, train, resume or evaluate | [Workflows](workflows.md) |
| Prepare or compile local datasets | [Data preparation](data-preparation.md) |
| Train with explicit Python dataloaders | [Local training](local-training.md) |
| Inspect the preserved paused 1B/70B recipe | [FSDP 1B runbook](../scripts/pretrain_1b_fsdp/README.md) |
| Understand TR-GQA, TR-MHA and deterministic experts | [Architecture and naming](architectures.md), [TR-Hash engine](tr-hash-engine.md) |
| Select CPU, MPS, CUDA or ROCm behavior | [GPU and dispatch paths](cuda.md) |

`complexity train validate`, `complexity train inspect` and
`complexity checkpoint inspect` provide local checks without starting a job.
File/hash inspection, CPU regression tests and a successful GPU restore are
separate levels of evidence.

## Model lineages and document status

| Area | Status | Entry point |
|---|---|---|
| TR-HASH-MoE-1B, 70B-token target | preserved recipe; training paused and resumed periodically | [1B FSDP](../scripts/pretrain_1b_fsdp/README.md), [workflows](workflows.md) |
| Public model/config/trainer interfaces | reusable framework APIs | [API](api.md), [framework architecture](framework-architecture.md) |
| 200M architecture and recorded metrics | released, checkpoint-specific evidence | [200M release](tr-hash-200m-release.md) |
| 200M full SFT v2 | released epoch 3 / step 5,982 | [SFT v2](tr-hash-200m-clean-sft-v2.md) |
| 200M launch profiles | historical recipe evidence; not the 1B workflow | [Training lineage](training.md), [streaming data](tr-hash-200m-streaming-data.md) |
| TokenRoutedMLP conversion | compatibility only | [Migration](token-routed.md) |
| o200k Dense/TR comparison plans | historical, non-runnable | [Run configurations](run_configs.md), [B200 runbook](200m-o200k-b200-runbook.md) |
| Multimodal/generative modules | experimental unless a release guide states otherwise | [Multimodal index](multimodal.md) |
| Vision detector | separate lineage and validation gates | [Object detection](vision/tr-hash-object-detection.md) |

The 1B pretraining budget remains 70B tokens, represented by 69,999,919,104
whole-update tokens in its preserved recipe. Its schedule and checkpoint state
are never replaced by a small-model example or the separate 200M lineage.
Pretraining, same-corpus refinement and SFT are different stages; resuming
pretraining does not start either later stage.

## Release evidence

The 200M source lineage is approximately 162.07B token exposures before SFT.
Its refinement was interrupted and its full-parameter SFT v2 completed three
epochs. Consult the [release page](tr-hash-200m-release.md) and [SFT
measurements](tr-hash-200m-clean-sft-v2.md) before quoting results. Training
loss, held-out SFT loss and downstream accuracy measure different properties.

Historical experiment files preserve provenance. A command in an older
runbook is not an available entry point unless its executable is present and
its environment/artifact requirements have been checked.

## Supporting guides

- [Refactor status and remaining work](refactor-status.md)
- [200M experiment: refinement versus direct SFT](experiments/tr-hash-200m-refinement-ablation.md)
- [Custom models and registries](custom-models.md)
- [Efficient training](efficient.md)
- [Optimizer and resume boundaries](training-boundaries.md)
- [Portable Supervisor jobs](supervisor-jobs.md)
- [Hugging Face organization card](huggingface-org-card.md)
- [Use of generative AI tools](claude-role.md)

## Multimodal and vision guides

- [Vision: one package and command tree](vision/index.md)
- [Multimodal prototypes](multimodal.md)
- [TR-Hash image editor](vision/tr-hash-image-editor.md)
- [TR-Hash image-text-to-text](vision/tr-hash-image-text-to-text.md)
- [TR-Hash text-to-image](vision/tr-hash-text-to-image.md)
- [TR-Hash object detection and serving](vision/tr-hash-object-detection.md)
- [TR-Hash Vision ONNX deployment](vision/onnx_deploy.md)
- [Vision v8 COCO accuracy gates](vision/vision-v8-coco-accuracy-gates.md)
- [Vision v8 ONNX quantization](vision/vision-v8-onnx-quantization.md)
- [Detector specialization and ablations](vision/TR_HASH_DETECTOR_SPECIALIZATION.md)
- [TR-Hash sensor fusion](tr_hash_sensor_fusion.md)
- [Vision dependency stack](vision/vision-dependency-stack.md)

## Evidence vocabulary

- **Reusable API**: maintained package interface, independent of one experiment.
- **Preserved recipe**: an existing run's immutable model/training contract.
- **Released**: a named artifact with the measurements specified in its release guide.
- **Compatibility**: retained for loading or converting an older format.
- **Historical**: records an earlier experiment; may require archived executables.
- **Experimental**: implementation without release-level evidence.
- **Planned**: a protocol or target without a completed result.

Every numerical claim should identify the checkpoint, data or token exposure,
evaluation split, runtime and retained artifact that supports it.
