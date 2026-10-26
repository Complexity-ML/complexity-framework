# Training lineage and stage contracts

For current installation, training, resume and evaluation tasks, use
[workflows](workflows.md). This page records the **separate 200M release
lineage** and framework stage boundaries. It is not the paused 1B recipe.

## Stage boundaries

The framework's non-Vision release contract is:

```text
pretraining → refinement (same corpus, fresh optimizer) → SFT
```

`validate_training_stage_transition` rejects direct pretraining-to-SFT
transitions. Refinement identity is checked against the complete `unique_core`
source/shard/row contract; an equal token total is insufficient. Vision's
augmentation anneal and clean-image phase supply its integrated refinement,
so that recipe does not receive a duplicate stage.

| Stage | Objective | State carried forward |
|---|---|---|
| Base pretraining | causal language modeling over the replay schedule | model weights |
| Refinement | causal language modeling over one same-corpus unique-token pass | weights; new optimizer/scheduler |
| Instruction SFT | final-assistant-only supervised causal loss | refinement weights; new optimizer/scheduler |

Token IDs continue to address the same trained expert routes while attention
provides contextual hidden states. A new corpus is continued pretraining or
SFT, not same-corpus refinement. Several historical filenames contain `sft`
for what the release records identify as full-parameter refinement.

## Preserved 1B run

The [1B FSDP runbook](../scripts/pretrain_1b_fsdp/README.md) owns the paused
`Pacific-i64/TR-HASH-MoE-1B-70B-Agentic-Pretraining` recipe. Its 70B budget is
534,057 updates × 131,072 tokens = 69,999,919,104 whole-update tokens.
Continue that same pretraining state when resuming; refinement/refit remains
a separate stage. The original model, data schedule, tokenizer, optimizer,
scheduler and distributed RNG contract are not replaced by the 200M lineage.

## Historical 200M recipe

The recorded source-token lineage is approximately 162.07B exposures:
base replay pretraining completed, unique-token refinement stopped at step
8,156 / 17,802, then audited full-parameter SFT v2 completed three epochs.
The `160B` label is rounded lineage naming, not a completed 70B refinement.

| Operation | Historical profile identifier | Present use |
|---|---|---|
| Base replay pretraining | `vast_pretrain_tr_hash_200m_70b_replay.sh`, `train_tr_hash_200m_200b` | archive/provenance; not a current generic launcher |
| Unique-token refinement | `vast_finetune_tr_hash_200m_70b_unique_phase2.sh` | archive/provenance |
| Three-epoch full SFT v2 | archived release manifest and checkpoint | reproduce from its exact recorded contract |

Some historical scripts are absent from this checkout. Retrieve the matching
archived revision and pinned artifact manifests before attempting reproduction;
do not substitute a differently named current runner. The current framework
API can construct/load the architecture without importing those scripts.

The historical server profiles used `/workspace`, `/venv/main`, DDP processes
per GPU, BF16 compute and the prescribed CUDA/Triton/Liger paths. These are
records of one environment, not installation defaults for a new machine.

## Audited full-parameter SFT v2

The dataset `AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-32K-300K` records 300,000 training
and 3,000 held-out examples. It uses the release's 32,000-token tokenizer,
packed 2,048-token sequences, EOS `</s>` (ID 0), final-assistant-only labels,
source/file hashes and a fail-closed no-truncation gate.

The recorded training mode was full parameter, not LoRA: BF16 AdamW, LR
`2e-5`, betas `0.9/0.95`, weight decay `0.1`, 3% warmup and one continuous
cosine schedule over three epochs. Liger/custom-kernel policy, complete
held-out evaluation and resumable epoch checkpoints belong to that exact
recipe. Experimental LoRA is a different optimization contract.

The completed run reached step 5,982. Its epoch-3 matched held-out loss was
0.959617 (PPL 2.61); the root release is the epoch-3 F32 SafeTensors artifact.
See [SFT v2 evidence](tr-hash-200m-clean-sft-v2.md) and the [200M
release](tr-hash-200m-release.md) for source/epoch measurements and PIQA protocol.
Those results do not establish the paused 1B model's quality.

## Experimental refinement ablation

The user-requested [200M ablation](experiments/tr-hash-200m-refinement-ablation.md)
adds a research branch from 130B base weights directly to full-parameter SFT.
It is explicitly `source_stage=pretraining`, `release_eligible=false`; its
protocol does not turn a bypass of refinement into a promoted release recipe.
The [packaged SFT workflow](workflows.md#explicit-sft-workflows) now supports
preflight and explicit local execution, with CPU update/resume tests. Artifact
preflight, GPU qualification and any actual experiment launch remain separate.

## Resume and export boundaries

An exact resume restores model, optimizer, scheduler, data cursor, distributed
RNG and compatible training-critical arguments. Weights-only initialization
starts a new stage. Changing corpus, objective, optimizer grouping or schedule
must be identified accordingly.

A native weight export retains configuration, tensor values, route buffers,
tokenizer and any applicable chat template. Publishing/synchronization is a
separate lifecycle operation, verified before cleanup. A directory with weights
alone does not restore optimization state; distributed FSDP shards use their
original reconstruction path.

## Measurement and evidence

A publishable experiment retains framework revision, checkpoint hashes,
dataset/tokenizer revisions, stage/mode, optimizer schedule, world size,
hardware, precision, effective batch, selected kernel backend, validation
protocol, all compared checkpoints and selection rule.

Training minibatch loss, matched held-out loss and downstream continuation
accuracy measure different things. PIQA is a narrow checkpoint-selection
sanity check, not general chat/agent quality. File/hash checks, local tests,
GPU restart qualification and sustained training/upload monitoring are separate
evidence scopes. See [optimizer boundaries](training-boundaries.md) and
[workflows](workflows.md#evaluation).
