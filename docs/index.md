# Documentation

Complexity Framework is a PyTorch research stack for deterministic
TR-Hash Mixture-of-Experts (TR-MoE) language and multimodal models.

## Start here

1. [Architecture and naming](architectures.md)
2. [Getting started](getting-started.md)
3. [TR-Hash execution engine](tr-hash-engine.md)
4. [TokenRoutedMLP (removed) and migrating to TR-Hash](token-routed.md)
5. [Training](training.md)
6. [Two-dimensional full-shard SFT weighting](sft-full-shard-2d-weighting.md)
7. [Run configurations](run_configs.md)
8. [GPU and dispatch paths](cuda.md)
9. [API reference](api.md)
10. [Claude's role in this project](claude-role.md)

## Architecture vocabulary

| Public name | Attention | FFN |
| --- | --- | --- |
| TR-GQA | GQA | TR-MoE |
| TR-MHA | MHA | TR-MoE |

TR-GQA and TR-MHA share the same `TRHashEngineMLP`. They differ only in the
attention head layout. The framework is currently scoped to TR-Hash MoE only
— dense and learned-router baselines were removed and will return later as
explicit comparisons.

The registry values `tr_mha` and `tr_mha_v2` refer to experimental routed
residual adapters inside attention. They are documented separately in
[`../TR_MHA.md`](../TR_MHA.md) to avoid conflating them with the main
MHA + TR-MoE architecture.

## Additional guides

- [MoE comparison](moe.md)
- [Custom models and registries](custom-models.md)
- [Efficient training](efficient.md)
- [Multimodal prototypes](multimodal.md)

## Document status

| Area | Status | Entry point |
|---|---|---|
| Architecture and TR-Hash runtime | current | [Architecture](architectures.md), [TR-Hash engine](tr-hash-engine.md) |
| LoRA instruction tuning | current | [Training](training.md), [2D full-shard weighting](sft-full-shard-2d-weighting.md) |
| CUDA and dispatch | current | [GPU and dispatch paths](cuda.md) |
| Multimodal and generative modules | experimental | [Multimodal prototypes](multimodal.md) and the modality guides below |
| TokenRoutedMLP migration | compatibility only | [Migration guide](token-routed.md) |
| o200k/Dense pretraining records | historical, non-runnable | [Run configurations](run_configs.md), [200M B200 runbook](200m-o200k-b200-runbook.md) |

Historical documents preserve evidence provenance. Their commands must not be
treated as supported entrypoints unless a current guide explicitly says so.

## Multimodal and generative models

- [TR-Hash image editor](tr-hash-image-editor.md)
- [TR-Hash image-text-to-text](tr-hash-image-text-to-text.md)
- [TR-Hash text-to-image](tr-hash-text-to-image.md)
- [TR-Hash object detection and serving](tr-hash-object-detection.md)
- [TR-Hash detector specialization and ablations](TR_HASH_DETECTOR_SPECIALIZATION.md)
- [TR-Hash sensor fusion](tr_hash_sensor_fusion.md)

## Operations

- [200M o200k B200 runbook](200m-o200k-b200-runbook.md)
- [Hugging Face org card](huggingface-org-card.md)

## Evidence policy

A configuration file or model implementation is not a completed experiment.
Documentation distinguishes:

- **implemented**: represented in code and tests;
- **pilot**: bounded evidence, usually short or single-seed;
- **controlled comparison**: matched protocol with tracked metrics;
- **planned**: a launch or cluster plan without completed metrics.

Claims should point to the tracked configuration, metrics, or artifact that
supports them.
