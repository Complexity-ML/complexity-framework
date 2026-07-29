# Documentation

Complexity Framework is a PyTorch research stack for deterministic
Token-Routed Mixture-of-Experts (TR-MoE) language models.

## Start here

1. [Architecture and naming](architectures.md)
2. [Getting started](getting-started.md)
3. [TR-MoE internals](token-routed.md)
4. [Training](training.md)
5. [Run configurations](run_configs.md)
6. [GPU and dispatch paths](cuda.md)
7. [API reference](api.md)

## Architecture vocabulary

| Public name | Attention | FFN |
| --- | --- | --- |
| TR-GQA | GQA | TR-MoE |
| TR-MHA | MHA | TR-MoE |
| Dense GQA | GQA | dense SwiGLU |
| Dense MHA | MHA | dense SwiGLU |

TR-GQA and TR-MHA share the same `TokenRoutedMLP`. They differ only in the
attention head layout.

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

## Evidence policy

A configuration file or model implementation is not a completed experiment.
Documentation distinguishes:

- **implemented**: represented in code and tests;
- **pilot**: bounded evidence, usually short or single-seed;
- **controlled comparison**: matched protocol with tracked metrics;
- **planned**: a launch or cluster plan without completed metrics.

Claims should point to the tracked configuration, metrics, or artifact that
supports them.
