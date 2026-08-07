---
license: cc-by-nc-4.0
language:
- en
pipeline_tag: text-generation
library_name: vllm
tags:
- causal-lm
- mixture-of-experts
- token-routing
- gqa
- tr-hash
---

# TR-HASH-MOE-100M

`TR-HASH-MOE-100M` is a 99.5M-parameter causal base language model from the
Complexity TR-Hash experiments. It combines GQA with a dense shared MLP path
and small residual experts selected by a deterministic balanced hash of token
identity. The selected route is known without evaluating a learned contextual
router.

This checkpoint is **pretrained only**. It is not instruction-tuned and should
not be expected to follow chat prompts reliably.

## Architecture

- Parameters: 99.5M
- Hidden size: 384
- Layers: 10
- Attention: GQA, 8 query heads / 2 key-value heads
- Vocabulary: o200k, 200,019 entries
- Context length: 2,048
- Shared MLP width: 1,520
- Residual experts: 4 experts, top-2, 0.5 / 0.5 route weights
- Routing: deterministic balanced token-ID hash
- Hash channel modulation: enabled
- Tied token embeddings / LM head
- Weight dtype in this release: BF16

## Training evidence

The model was trained for approximately 2B FineWeb-Edu tokens with seed 42.
Training used a separate fixed evaluation shard. At the 10,000-step evaluation,
TR-Hash reached 3.6680 NLL; the matched Dense-GQA reference reached 3.6685.
Measured training throughput was about 375.8k tokens/s for TR-Hash and 397.5k
tokens/s for Dense-GQA on the recorded run.

These are single-seed experimental measurements, not evidence of a general
quality or speed advantage.

## Runtime

The exported checkpoint uses the `DeepForCausalLM` contract implemented in the
Complexity vLLM fork. Standard Transformers does not natively implement this
TR-Hash architecture.

- Training and architecture: https://github.com/Complexity-ML/complexity-framework
- CUDA inference: https://github.com/Complexity-ML/vllm-cuda_graph

The included tokenizer metadata points to the `o200k_base` encoding. This base
checkpoint deliberately contains no chat template.

## Intended use and limitations

This release is intended for architecture research, evaluation, continued
pretraining, and controlled fine-tuning. A 99.5M-parameter base model has
limited factual knowledge and generation quality. Outputs may be incorrect,
repetitive, biased, or unsafe and should not be relied on for consequential
decisions.

## License

Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
