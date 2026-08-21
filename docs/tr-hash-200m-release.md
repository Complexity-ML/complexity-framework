# TR-HASH MoE 200M release

> **Current reference release.** This page describes the public 201.2M model
> lineage and the artifacts that are served today. It supersedes the older
> 492.1M/20B and 200M-o200k planning documents as the default language-model
> reference for Complexity Framework.

## Released lineage

| Stage | Released progress | Training signal | PIQA acc / acc_norm |
|---|---:|---:|---:|
| Base pretraining | step 165,298; 129,995,636,736 token exposures | final logged minibatch loss 2.652628; PPL 14.19 | 65.45% / 65.61% |
| Full-parameter refinement | step 8,156 / 17,802; +32,069,495,945 unique-token exposures | terminal displayed loss 2.3208; PPL 10.2 | 68.66% / 68.39% |
| Full-parameter instruction SFT | 3 epochs; 238.9M supervised tokens | epoch-3 held-out loss 1.220861; PPL 3.39 | **68.82% / 69.31%** at promoted epoch 2 |

The base and refinement stages represent exactly 162,065,132,681 source-token
exposures. `160B` in repository names is a rounded lineage label. The
refinement stopped at 45.8% of its planned 70B-token pass; the release does not
claim that the pass completed.

The instruction model is a **full-parameter SFT**, not LoRA or QLoRA. All
201,194,368 parameters were trainable. Epoch 2 was selected by downstream PIQA;
epoch 3 has the lowest matched SFT validation loss but was not silently
substituted for the benchmark-selected checkpoint.

## Architecture contract

| Field | Released value |
|---|---:|
| Parameters | 201,194,368 |
| Decoder layers / hidden size | 16 / 896 |
| Attention | GQA, 14 query heads / 2 KV heads, QK normalization |
| Vocabulary / context | 32,000 / 2,048 |
| Shared SwiGLU width | 3,072 |
| Routed experts | 4 stored, width 64 each |
| Routed width | 256 stored; 128 active with top-2 |
| Routing | `token_id_multi_hash`, 2 rendezvous-hash channels |
| Route weights | 0.5 / 0.5 |
| Embeddings | input/output tied |

`ModelConfig.intermediate_size=256` means the **total stored routed width**.
`TRHashEngineMLP` divides it by `num_experts=4`, producing four width-64 expert
branches. Every token also traverses the width-3,072 shared branch. The two
selected expert outputs are summed with fixed equal weights and a routed output
scale of 2.0 in the released configuration.

Multi-hash routing is resolved once per layer into a persisted
`[top_k, vocab_size]` table. It has no learned router, no auxiliary
load-balancing loss, and no per-token routing network at inference time.

## Checkpoints and artifacts

- [130B base checkpoint](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-130B)
- [Step-8,156 refinement checkpoint](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement)
- [Promoted full-parameter SFT](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT)
- [Luciole 16-way SFT dataset](https://huggingface.co/datasets/AETHORIA-AI/luciole-16way-sft-209k)
- [Release paper](https://www.complexity-ai.fr/papers/tr-hash-200m-multi-hash-routing.pdf)
- [Live TR-Hash-i64 chat](https://www.complexity-ai.fr/ai-lab)

The root SFT `model.safetensors` is epoch 2 / step 926. Its published SHA-256
is `2dd5a35624dda4cb267d872f974fd448c9e085004c163b31772515bfd9573caa`.
The three training checkpoints remain available for audit and exact metric
provenance.

## Evaluation protocol

PIQA uses the complete 1,838-example validation split. Each candidate ending
is scored as a zero-shot causal continuation of the prompt. Accuracy compares
total continuation log-likelihood; `acc_norm` compares length-normalized
log-likelihood. Evaluation uses no chat template, caps sequences at 2,048
tokens, and runs the same checkpoint weights in FP16 eager PyTorch or MLX.

PIQA is a narrow physical-commonsense checkpoint-selection sanity check. It is
not an evaluation of instruction following, safety, code generation, factual
recall, or open-ended chat quality. Training minibatch loss, matched SFT
validation loss, and PIQA are different measurements and must not be compared
as if they shared one sampling process.

## Reproduction entry points

| Operation | Entry point |
|---|---|
| Base replay pretraining | `scripts/vast_pretrain_tr_hash_200m_70b_replay.sh` |
| Unique-token refinement | `scripts/vast_finetune_tr_hash_200m_70b_unique_phase2.sh` |
| Luciole dataset tokenization | `scripts/vast_tokenize_upload_luciole_16way_sft.sh` |
| Three-epoch full SFT | `scripts/vast_sft_200m_luciole_16way_full_3e.sh` |
| PIQA for all SFT epochs | `scripts/eval_full_sft_piqa_3.sh` |
| SFT release export | `scripts/export_full_sft_release.py` |
| Hugging Face synchronization | `scripts/vast_sync_200m_luciole_full_sft.sh` |

The `vast_*` launchers are production profiles for `/workspace` machines and
expect the environment variables documented in [Training](training.md). Read
and adapt their paths before using them on another cluster.

## Limitations

This is one architecture and one training lineage. There is no
parameter-matched dense or learned-router control, multi-seed replication,
significance test, benchmark-contamination audit, systematic safety study, or
claim of architectural superiority. The 200M assistant can hallucinate,
repeat, refuse benign requests, fail arithmetic or code, and lose coherence on
long outputs. The public demo is a research artifact, not a production agent.
