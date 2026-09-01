# Complexity Framework documentation

Complexity Framework is the PyTorch research and training stack behind the
released TR-HASH MoE 200M language-model lineage and separate experimental
multimodal systems.

## Start here

1. [TR-HASH MoE 200M release](tr-hash-200m-release.md) — architecture,
   checkpoints, metrics, evaluation protocol, and limitations.
2. [Getting started](getting-started.md) — install, construct, load, and test a
   model.
3. [Architecture and naming](architectures.md) — TR-GQA, TR-MHA, and TR-MoE.
4. [TR-Hash execution engine](tr-hash-engine.md) — routes, widths, backends,
   and CUDA Graph constraints.
5. [Training](training.md) — base pretraining, refinement, full SFT, evaluation,
   export, and resume boundaries.
6. [GPU and dispatch paths](cuda.md) — PyTorch fallback, Triton/CGGR, Liger,
   ROCm, and reporting.
7. [API reference](api.md) — public Python and CLI surfaces.
8. [Released clean SFT v2](tr-hash-200m-clean-sft-v2.md) — audited 300K
   mixture, reusable 32K token shards, completed training, and epoch metrics.

All non-Vision release recipes follow `pretraining -> same-corpus refinement
-> SFT`. Vision is the documented exception because its clean-image
refinement is already integrated into the detector recipe.

## Current 200M release path

```text
130B replay-scheduled base pretraining
        ↓ fresh optimizer, weights only
32.07B unique-token full-parameter refinement (stopped at step 8,156)
        ↓ full checkpoint weights
3 epochs audited 300K full-parameter instruction SFT v2
        ↓ PIQA selection
epoch 3 / step 5,982 copied to the root F32 SafeTensors release
        ↓
TR-Hash-i64 OpenAI-compatible serving
```

The source-token lineage is approximately 162.07B exposures. SFT v2 then uses
202,948,693 tokenized training tokens per epoch, for 608,846,079 token
exposures over three epochs. The SFT is full parameter, not LoRA. See the
[release page](tr-hash-200m-release.md) before quoting any metric.

## Document status

| Area | Status | Entry point |
|---|---|---|
| 200M release and metrics | **current** | [Release](tr-hash-200m-release.md) |
| Architecture and TR-Hash runtime | **current** | [Architecture](architectures.md), [engine](tr-hash-engine.md) |
| 200M pretraining, refinement, full SFT | **current** | [Training](training.md), [streaming data](tr-hash-200m-streaming-data.md) |
| 200M clean SFT v2 | **released; epoch 3 at repository root** | [SFT v2](tr-hash-200m-clean-sft-v2.md) |
| CUDA, Triton, Liger, fallback | **current** | [GPU and dispatch](cuda.md) |
| Public Python surface | **current** | [API reference](api.md) |
| TokenRoutedMLP conversion | compatibility only | [Migration](token-routed.md) |
| o200k Dense/TR comparison plans | historical, non-runnable | [Run configurations](run_configs.md), [B200 runbook](200m-o200k-b200-runbook.md) |
| Multimodal and generative modules | experimental | [Multimodal index](multimodal.md) |
| Vision detector | separate released lineage | [Object detection](tr-hash-object-detection.md) |

Historical documents preserve evidence provenance. Their commands must not be
treated as supported entry points unless a current page explicitly says so.

## Architecture vocabulary

| Public name | Attention | Feed-forward path |
|---|---|---|
| TR-GQA | grouped-query attention | TR-MoE |
| TR-MHA | multi-head attention | TR-MoE |
| TR-MoE | attention-independent | shared SwiGLU plus fixed token-ID experts |

The registry values `tr_mha` and `tr_mha_v2` are experimental token-routed
residual adapters inside attention. They are not synonyms for the released
GQA + TR-MoE architecture.

## Supporting guides

- [Custom models and registries](custom-models.md)
- [Efficient training](efficient.md)
- [Run configurations and planners](run_configs.md)
- [Historical TokenRoutedMLP migration](token-routed.md)
- [Hugging Face organization card](huggingface-org-card.md)
- [Use of generative AI tools](claude-role.md)

## Multimodal and vision guides

- [Multimodal prototypes](multimodal.md)
- [TR-Hash image editor](tr-hash-image-editor.md)
- [TR-Hash image-text-to-text](tr-hash-image-text-to-text.md)
- [TR-Hash text-to-image](tr-hash-text-to-image.md)
- [TR-Hash object detection and serving](tr-hash-object-detection.md)
- [TR-Hash Vision ONNX deployment](onnx_deploy.md)
- [Vision v8 COCO accuracy gates](vision-v8-coco-accuracy-gates.md)
- [Detector specialization and ablations](TR_HASH_DETECTOR_SPECIALIZATION.md)
- [TR-Hash sensor fusion](tr_hash_sensor_fusion.md)
- [Vision dependency stack](vision-dependency-stack.md)

## Evidence policy

A configuration or implementation is not a completed experiment. Documents
use these labels:

- **current**: matches the released code and artifacts;
- **compatibility**: supported only for migration or conversion;
- **historical**: preserves an earlier experiment but is not the default path;
- **experimental**: implemented without release-level evidence;
- **planned**: a protocol or launch shape without completed metrics.

Every numerical claim should identify the checkpoint, data or token exposure,
evaluation split, runtime, and artifact that supports it.
