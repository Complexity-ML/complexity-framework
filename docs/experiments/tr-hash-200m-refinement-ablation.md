# 200M refinement ablation after SFT

**Incomplete research comparison; no completed new benchmark.** The general research
hypothesis is that combining lexical routing and semantic computation can allow
pretraining with fewer tokens, followed by a distinct refit that improves
benchmarks toward models of the same size available in 2026. This hypothesis
has not yet been demonstrated.

The current experiment starts from the **same TR-HASH 200M base pretrained on
130B tokens**: direct instruction SFT versus that same base followed by refit
and the same SFT. It measures whether refit supplies the benchmark gain sought
for this model. Refit is a distinct stage after the pretraining budget. The
paused 1B/70B run is outside this experiment.

## Two branches

| Branch | Source weights | Following stage |
|---|---|---|
| A — direct SFT | [200M/130B base](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-130B) | fresh full-parameter SFT, without refit |
| B — refit + SFT | the same 200M/130B base | distinct same-corpus refit, then the same full-parameter SFT |

The [published 160B-SFT epoch 3](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT)
is the recorded reference for branch B, including the step-8,156 refit.
`160B` is retained as a historical repository identifier; it does not denote
a different base pretrained on 160B tokens in this comparison.

Branch A intentionally starts from **model weights only**, with a fresh
optimizer/scheduler and a new experiment directory. Its research metadata is
`source_stage=pretraining`, `release_eligible=false`; the normal promoted
release contract still includes refinement. Retain the base tokenizer,
32,000 embedding rows and trained route tables unchanged.

## Shared SFT contract

Use exactly
[AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-32K-300K](https://huggingface.co/datasets/AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-32K-300K):
300,000 training examples, 3,000 held-out examples, the 32K tokenizer, final
assistant-only labels (`-100` masks for prompt/history), no response truncation,
packed 2,048-token sequences, **three epochs and all parameters trainable**.
The repository metadata pins dataset snapshot
`084a658ec47e4ee872f6d67fdbad3602f599424b`. Verify downloaded payloads against
that snapshot's manifests before execution. Stored labels are already
next-token targets: retain their alignment and supervised EOS, rather than
shifting them a second time.

Use identical SFT sample order, packing, optimizer, LR/warmup/schedule,
precision and checkpoint/evaluation policy across compared branches. The
[recorded v2 recipe](../tr-hash-200m-clean-sft-v2.md) is the starting reference.
The old short dataset name `...SFT-v2-300K` redirects to v3/32,004 vocabulary;
its unpinned contents are unsuitable for this 32K comparison.

## Confirmed reference geometry

The published training index and complete example lengths establish the
following counts; these are data/plan evidence, not a completed new run:

| Per-epoch quantity | Value |
|---|---:|
| Training examples | 300,000 |
| Source token IDs | 202,948,693 |
| Supervised targets | 96,674,982 |
| Next-fit packs of 2,048, with one masked EOS separator between examples | 127,554 |
| Reference global batch | 64 packs = 4 ranks × 16 packs per rank |
| Optimizer updates | 1,994 per epoch |
| Three-epoch horizon | 5,982 updates |

The last global batch is padded under the reference sampler contract. Source
unique-token totals, inserted separators, padding and repeated tail packs are
different quantities; preserve the packing/sampler contract when matching the
published SFT. Changing the microbatch may require accumulation while keeping
the reference global batch and update horizon.

## Current framework interface

Use [explicit SFT workflows](../workflows.md#explicit-sft-workflows):
`complexity train sft CONFIG` inspects local artifacts and returns the plan;
`--run` explicitly starts execution. The single-device runner accumulates the
reference rank batches; the four-rank reference geometry does not imply a
distributed launch. CPU update/resume tests pass. The final refactored package
still requires GPU qualification; the complete 200M comparison is unfinished.

## Resolve before execution

- Pin immutable revisions and verify weights, config, tokenizer and route
  hashes for both model sources; verify the exact v2-32K dataset manifest,
  split/bin hashes, row counts and masks.
- Match the published branch's remaining execution details: seed/sample order,
  accumulation, optimizer groups, kernel/precision settings and evaluation
  boundaries. The reference global batch and update horizon above are fixed;
  server memory, microbatch suitability and actual hardware readiness still
  need validation.
- Run `complexity train sft CONFIG` against the pinned local assets before
  deciding execution settings. Package tests cover label alignment, parameter
  updates and exact local continuation on tiny CPU models; they do not
  establish 200M/GPU quality, historical parity or a four-rank server restart.

A direct-SFT run on the earlier frozen wheel was observed at update 124 of
5,982, with a complete checkpoint at update 100. A small CUDA gradient check
also passed on that server. These observations do not qualify the final
refactored package or establish a completed comparison. Later SSH attempts
were refused; current server state and successful shutdown are unverified.
No new rental or training launch is part of this framework refactor. The 1B
run and published checkpoints were not modified.

## Comparison and scientific limits

Measure each source and each SFT epoch on the same finite held-out split; keep
loss, perplexity and token counts. Evaluate both final models with the same
PIQA continuation-likelihood protocol and a fixed instruction/code/math
regression panel. Retain source/revision hashes and results; compare equivalent
epoch boundaries rather than choosing a favorable checkpoint after inspection.

The published B artifact is a historical reference until its exact SFT data
and execution contract are matched. Unmatched runtime, sample order or
optimization makes the comparison observational; a controlled B rerun would
be a separate execution decision.

Compare the post-SFT benchmark results against criteria defined before inspecting
the outcomes. This comparison measures the contribution of the distinct refit
stage for this 200M model; it neither assumes a gain nor establishes that refit
is universally necessary. A benchmark gain here would not alone demonstrate
the broader hypothesis of fewer pretraining tokens or parity with other
same-size models available in 2026. Those claims require their own controlled
comparisons. One pair also does not establish statistical significance,
market readiness or lower cost at equal quality.
