# Training

> **Current path.** The released language-model recipe is 130B replay
> pretraining, fresh-optimizer full-parameter refinement, then three epochs of
> full-parameter instruction SFT on the audited 300K v2 dataset.

## Stage boundaries

This is the framework-wide release contract for every non-Vision model family:

```text
pretraining -> refinement (same corpus, fresh optimizer) -> SFT
```

Direct `pretraining -> SFT` transitions are rejected by
`validate_training_stage_transition`. Refinement identity is checked from the
complete `unique_core` source/shard/row selection; equal token totals are not
enough. Vision does not receive a duplicate stage: its canonical augmentation
anneal and clean-image phase already supply refinement inside that recipe.

This boundary is mandatory for lexical TR-HASH language models in Complexity
Framework. Because token identity selects fixed expert routes, refinement keeps
the vocabulary and persisted routes unchanged while giving all attention,
shared and routed parameters one clean pass over the exact pretraining corpus
with fresh optimization state. It is the last causal-language-modeling phase
before instruction data changes the objective.

| Stage | Objective | State carried forward |
|---|---|---|
| Base pretraining | causal language modeling over the replay schedule | final model weights |
| Refinement | causal language modeling over one unique-token pass | model weights only; fresh optimizer/scheduler |
| Instruction SFT | final-assistant-only supervised causal loss | full refinement weights; fresh optimizer/scheduler |

Refinement and instruction SFT are different stages. Several older filenames
contain `sft` for the refinement run; documentation and release cards use the
semantic name **full-parameter refinement**.

## Production environment

The tracked `vast_*` launchers default to:

- repository: `/workspace/complexity-framework`;
- virtual environment: `/venv/main`;
- model and dataset caches under `/workspace`;
- one DDP process per GPU;
- BF16 training on supported NVIDIA hardware.

Override documented environment variables when using another layout. Run the
preflight and a bounded smoke test before an expensive job. Do not embed access
tokens in scripts, supervisor configurations, logs, or Git history.

Long-running jobs use `complexity.training.SupervisorManager` and a validated
`SupervisorProgram`. The framework atomically generates the host configuration;
static instance-specific Supervisor files are not versioned. Never put tokens,
passwords, API keys, or other credentials in a program environment mapping.

## Stage 1: replay-scheduled base pretraining

Entry point:

```bash
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=8 \
GRADIENT_ACCUMULATION=8 \
TOKENIZED_CACHE_GB=24 \
scripts/vast_pretrain_tr_hash_200m_70b_replay.sh
```

The launcher builds or validates the immutable replay plan, verifies any
explicit `TARGET_TOKENS`, streams tokenized shards through a bounded local
cache, and launches `scripts.train_tr_hash_200m_200b` with DDP. Its default
schedule targets 40 token packs and the released 130B exposure count.

`RESUME=auto` resumes the most recent compatible checkpoint. A successful run
exports `final/`; optional cleanup happens only after successful completion.
Checkpoint deletion and Hugging Face synchronization are separate lifecycle
operations and must not race each other.

## Stage 2: full-parameter refinement

Entry point:

```bash
INIT_CHECKPOINT=artifacts/tr_hash_200m_70b_replay/final \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=32 \
GRADIENT_ACCUMULATION=1 \
scripts/vast_finetune_tr_hash_200m_70b_unique_phase2.sh
```

The committed plan
`configs/replay_plans/tr_hash_70b_unique_only_phase2.json` selects the same
69,997,690,880 unique tokens once, with no replay or augmentation. The launcher
loads phase-1 weights through `--init-checkpoint` but creates a fresh optimizer,
scheduler, and step count.

Before allocating a model, the trainer hashes both plans' exact `unique_core`
contracts and requires identical fingerprints. Changing one shard or row while
preserving the total token count fails closed.

The public refinement stopped at step 8,156 / 17,802 after approximately
32.07B additional exposures. `160B` in the repository name is a rounded source
lineage label, not a completed 70B refinement claim.

## Audited SFT v2 data

`AETHORIA-AI/TR-HASH-MoE-200M-SFT-v2-300K` provides the canonical
32,000-token release: 300,000 training examples and 3,000 held-out examples.
It uses the tokenizer stored with the released model and
`AETHORIA-AI/TR-HASH-Tokenizer-32K`. The manifest pins file hashes, the
2,048-token sequence cap, EOS `</s>` (ID 0), final-assistant-only supervision,
source revisions and a fail-closed no-truncation gate.

## Stage 3: three-epoch full-parameter SFT

The completed production run used `scripts.sft_tr` with the released
32,000-token refinement checkpoint and audited SFT v2 binary dataset:

```bash
BASE_CHECKPOINT=/workspace/tr-hash-refinement
SFT_BIN=/workspace/tr-hash-moe-200m-sft-v2/tokenized/tr-hash-32k-v2-2048

python -m scripts.sft_tr \
  --checkpoint "$BASE_CHECKPOINT" \
  --tokenizer /workspace/tr-hash-tokenizer-32k \
  --sft-bin "$SFT_BIN" \
  --source-stage refinement \
  --full-parameter \
  --steps 0 \
  --epochs 3 \
  --seq-len 2048
```

The production profile starts from weights only, validates the dataset
manifest, requires Liger, and invokes `scripts.sft_tr` with:

- `--full-parameter`;
- 3 epochs, packed 2,048-token sequences, BF16 training;
- AdamW, LR `2e-5`, betas `0.9/0.95`, weight decay `0.1`;
- 3% warmup and one continuous cosine schedule across all epochs;
- custom kernels required by policy;
- complete held-out evaluation and checkpointing at every epoch;
- no early stopping before the configured three-epoch comparison completes.

`--batch-size` is per rank. Gradient accumulation changes optimizer frequency
and effective batch size; it does not create memory for a larger per-rank
microbatch. Tune the microbatch from measured VRAM, then choose accumulation to
reach the intended global batch.

## Full-parameter versus LoRA mode

`scripts.sft_tr` supports both modes:

- `--full-parameter` updates all model parameters and rejects unexpected LoRA
  state on resume;
- without `--full-parameter`, a positive `--lora-rank` is required and only
  selected low-rank adapters are optimized.

The current 200M release uses the first mode. Experimental LoRA runs are not
release substitutes and should not be merged into a full-parameter model card.

## Historical released clean SFT v2 result

The three-epoch v2 run completed at step 5,982. Matched held-out loss decreased
from 1.722610 at the source checkpoint to 0.959617 (PPL 2.61) after epoch 3.
The root release is the epoch-3 checkpoint in F32 SafeTensors. See
[Clean SFT v2](tr-hash-200m-clean-sft-v2.md) for the dataset contract and the
complete epoch table.

## Evaluation and checkpoint selection

Matched SFT evaluation consumes the finite held-out split at the source
checkpoint and every epoch boundary. Rank zero writes
`runs/<run-name>/metrics.csv`; checkpoint roots contain `step_*`, optional
`best/`, and selection metadata.

The historical v2 evaluation covered steps 1,994, 3,988, and 5,982.
The release protocol uses the full 1,838-example PIQA validation split,
zero-shot continuation likelihood, no chat template, FP16, and maximum length
2,048. The regression panel remains a diagnostic artifact; it is not reported
as an extra public benchmark column.

Held-out SFT loss and PIQA select different properties. Report both; never
claim that the lowest SFT loss automatically produces the best downstream or
chat checkpoint.

## Export and synchronization

Export must preserve the 32,000 embedding rows, perform a tensor-equality round
trip, copy the canonical 32K tokenizer and chat template, and verify remote
hashes before local cleanup.

## Resume rules

- `--resume` means exact continuation: model, optimizer, scheduler, data cursor,
  distributed RNG, world size, and training-critical arguments must match.
- `--init-checkpoint` or `--checkpoint` means weights-only initialization for a
  new stage with fresh optimization state.
- every text SFT launcher must declare `--source-stage refinement`; an
  additional SFT-on-SFT stage declares `--source-stage supervised-finetuning`;
- changing epochs, dataset mixture, loss policy, or LR schedule creates a new
  stage; it is not an exact resume.
- never resume full SFT from an experimental LoRA checkpoint or silently load a
  different phase into the same output directory.

## Precision, kernels, and memory

- BF16 is the released training precision;
- chunked tied-head CE bounds vocabulary-logit memory;
- Liger is a production dependency for the released SFT launcher;
- `--use-custom-kernels true` makes the production kernel expectation explicit;
- PyTorch fallback remains the numerical reference for regression tests;
- lower per-rank batch before changing gradient accumulation when an OOM occurs;
- record the selected backend from runtime logs, not only the requested flag.

See [GPU and dispatch paths](cuda.md).

## Verification

```bash
pytest -q \
  tests/test_tr_hash_200m_pretraining.py \
  tests/test_sft_bin.py \
  tests/test_sync_checkpoints_to_hf.py
```

A publishable run records the framework commit, source checkpoint hash,
dataset revision, tokenizer, model mode, optimizer schedule, world size,
hardware, precision, effective batch, selected kernel path, all checkpoint
metrics, selection rule, exported SafeTensors hash, and known limitations.
