# Documentation

Complexity Framework is a PyTorch research stack for deterministic
TR-Hash Mixture-of-Experts (TR-MoE) language and multimodal models.

## Start here

1. [Architecture and naming](architectures.md)
2. [Getting started](getting-started.md)
3. [TR-Hash execution engine](tr-hash-engine.md)
4. [TokenRoutedMLP (removed) and migrating to TR-Hash](token-routed.md)
5. [Training](training.md)
6. [Run configurations](run_configs.md)
7. [GPU and dispatch paths](cuda.md)
8. [API reference](api.md)

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
- [Historical Mu-Guidance control](dynamics.md)

## Multimodal / image

- [TR-Hash image editor](tr-hash-image-editor.md)
- [TR-Hash image-text-to-text](tr-hash-image-text-to-text.md)
- [TR-Hash text-to-image](tr-hash-text-to-image.md)
- [TR-Hash object detection and serving](tr-hash-object-detection.md)

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
