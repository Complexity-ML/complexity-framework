# TR-HASH MoE 200M clean SFT v2

> **Released result.** The three-epoch full-parameter run completed at step
> 5,982. Epoch 3 is copied to the model repository root as F32 SafeTensors.
> This page is a historical record. Its 32,000-token launch, evaluation and
> synchronization scripts were removed after the append-only 32,004-token
> migration. See [the SFT 32,004 recipe](tr_hash_200m_sft_32004_recipe.md) for
> the current pipeline.

## Purpose

The first 200M full-parameter SFT established a working assistant checkpoint,
but qualitative testing exposed incomplete code, weak arithmetic and brittle
multi-turn recall. The root cause audit also found that its 512-token training
windows truncated large parts of several code, math and chat completions.

SFT v2 therefore changes the data contract rather than repeating the same run:

- exactly 300,000 training and 3,000 validation examples;
- the model's native 2,048-token context;
- one complete final assistant turn per example;
- no prompt or completion slicing;
- short conversations expanded at assistant-turn boundaries;
- verifier-backed math and execution-filtered code;
- refusal, repetition, template-artifact and exact-duplicate rejection.

## Mixture

| Capability | Training examples |
|---|---:|
| General and precise instruction following | 100,000 |
| Verified mathematics | 50,000 |
| Execution-filtered code | 40,000 |
| Constraint following | 30,000 |
| Python algorithms | 20,000 |
| Rewriting | 20,000 |
| Summarization | 20,000 |
| STEM | 10,000 |
| Short multi-turn conversation | 5,000 |
| English/French bilingual instruction | 5,000 |
| **Total** | **300,000** |

The exact upstream datasets, revisions, row caps, licenses and rejection
counts remain recorded in the immutable published v2 manifest and repository
history. The aggregate is multi-license; row provenance is retained in
`source` and `capability` fields.

## Published dataset

The public dataset is
[`AETHORIA-AI/TR-HASH-MoE-200M-SFT-v3-32004-300K`](https://huggingface.co/datasets/AETHORIA-AI/TR-HASH-MoE-200M-SFT-v3-32004-300K).
The historical revision contained two equivalent representations:

1. `train.jsonl` / `eval.jsonl`, for inspection and independent processing;
2. `tokenized/tr-hash-32k-v2-2048/`, pre-encoded with the former TR-HASH
   **32,000-entry** tokenizer.

The tokenized representation is directly readable by `scripts.sft_tr`:

- `input_ids.bin`: little-endian unsigned 32-bit token IDs;
- `labels.bin`: little-endian signed 32-bit labels, with prompt/history labels
  set to `-100`;
- `examples.jsonl`: offsets, lengths, source and capability;
- `sft.idx.json`: format, counts, hashes and no-truncation contract;
- `tokenizer/` and `chat_template.json`: exact encoding assets.

For this tokenizer, `</s>` is EOS token ID 0. The token-shard manifest records
that spelling explicitly so prior assistant turns use the same one-token
separator during SFT and Hugging Face/Space inference; the historical literal
`<|endoftext|>` would split into ordinary tokens and is rejected by preflight.

## Dataset preflight

The historical source revision and its immutable manifest remain provenance
records. Current preflight is implemented by
`scripts/package_tr_hash_sft_32004_release.py` and enforced again by
`scripts/vast_sft_200m_32004_full_3e.sh`.

## Completed training

The removed historical launcher required:

- the Refinement step-8,156 checkpoint as a weights-only source;
- full-parameter training, never LoRA/QLoRA;
- 3 epochs, BF16, packed 2,048-token windows;
- Liger and the required custom CUDA/Triton path;
- a fresh optimizer and continuous cosine schedule;
- complete held-out evaluation and one checkpoint per epoch.

## Measured result

The run used 202,948,693 tokenized training tokens per epoch. All 201.2M
parameters were trainable. Evaluation at the source checkpoint and every epoch
boundary produced:

| Epoch | Step | Held-out SFT loss | SFT PPL | PIQA acc | PIQA acc_norm |
|---:|---:|---:|---:|---:|---:|
| 1 | 1,994 | 0.990943 | 2.69 | 67.90% | 68.93% |
| 2 | 3,988 | 0.963912 | 2.62 | 67.85% | 68.82% |
| 3 | 5,982 | **0.959617** | **2.61** | **68.01%** | **69.10%** |

PIQA uses all 1,838 validation examples, zero-shot causal continuation
likelihood, no chat template, FP16 weights, and a 2,048-token maximum length.
The source checkpoint's matched held-out loss was 1.722610 (PPL 5.60).

The archived regression report supplements the public loss and PIQA
measurements; it is not shown as a pass/fail column in the release table.
