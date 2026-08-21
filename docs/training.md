# Training

> **Current path.** The released language-model recipe is 130B replay
> pretraining, fresh-optimizer full-parameter refinement, then three epochs of
> full-parameter Luciole instruction SFT. The framework is not LoRA-only.

## Stage boundaries

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

The public refinement stopped at step 8,156 / 17,802 after approximately
32.07B additional exposures. `160B` in the repository name is a rounded source
lineage label, not a completed 70B refinement claim.

## Prepare the Luciole 16-way SFT data

The released dataset has 209,000 training examples and 2,100 held-out examples
from 16 capped sources. Supervision includes only the final assistant answer;
the prompt and prior assistant turns are masked.

The production tokenizer wrapper is:

```bash
scripts/vast_tokenize_upload_luciole_16way_sft.sh
```

It pins the source revision, uses the refinement tokenizer, caps sequences at
512 tokens, enforces a minimum completion length, writes the tokenized view,
and can upload it to
`AETHORIA-AI/luciole-16way-sft-209k`. The wrapper expects a private token file
at `/workspace/.hf_token`; create it with mode `0600` and never commit it.

## Stage 3: three-epoch full-parameter SFT

The production recipe is tracked verbatim in
`scripts/vast_sft_200m_luciole_16way_full_3e.sh`:

```bash
BASE_CHECKPOINT=/workspace/tr-hash-refinement \
TOKENIZER=/workspace/tr-hash-refinement \
DATA_ROOT=/workspace/luciole-16way-sft \
NPROC_PER_NODE=8 \
BATCH_SIZE_PER_GPU=48 \
scripts/vast_sft_200m_luciole_16way_full_3e.sh
```

The launcher refuses `RESUME_FROM`, validates the dataset manifest, requires
Liger, and invokes `scripts.sft_tr` with:

- `--full-parameter`;
- 3 epochs, packed 512-token sequences, BF16;
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

## Evaluation and checkpoint selection

Matched SFT evaluation consumes the finite held-out split at the source
checkpoint and every epoch boundary. Rank zero writes
`runs/<run-name>/metrics.csv`; checkpoint roots contain `step_*`, optional
`best/`, and selection metadata.

Evaluate all three full-SFT checkpoints on PIQA:

```bash
scripts/eval_full_sft_piqa_3.sh
```

The script evaluates steps 463, 926, and 1,389 on separate GPUs when available.
The release protocol uses the full 1,838-example PIQA validation split,
zero-shot continuation likelihood, no chat template, FP16, and maximum length
2,048. Epoch 2 / step 926 is the promoted root model.

Held-out SFT loss and PIQA select different properties. Report both; never
claim that the lowest SFT loss automatically produces the best downstream or
chat checkpoint.

## Export and synchronization

`scripts.export_full_sft_release` verifies that the supplied checkpoint matches
PIQA selection, writes `model.safetensors`, performs a tensor-equality
round-trip, copies tokenizer and chat-template assets, and records hashes and
reports in the release manifest.

`scripts/vast_sync_200m_luciole_full_sft.sh` uploads complete checkpoints to
`AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT`. Synchronization must finish before
local checkpoint cleanup. The root model is a copied release artifact; the
resumable epoch directories remain available for provenance.

## Resume rules

- `--resume` means exact continuation: model, optimizer, scheduler, data cursor,
  distributed RNG, world size, and training-critical arguments must match.
- `--init-checkpoint` or `--checkpoint` means weights-only initialization for a
  new stage with fresh optimization state.
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

## Historical 500M LoRA curriculum

The Card Corpus V2 two-dimensional weighting pipeline and
`configs/sft_500m_32k_v2_balanced*.yaml` are preserved for reproducibility.
They are not the current release path. See
[Two-dimensional full-shard SFT weighting](sft-full-shard-2d-weighting.md) and
[Run configurations](run_configs.md).

## Verification

```bash
pytest -q \
  tests/test_tr_hash_200m_pretraining.py \
  tests/test_sft_bin.py \
  tests/test_tokenize_luciole_16way_sft.py \
  tests/test_sync_checkpoints_to_hf.py
```

A publishable run records the framework commit, source checkpoint hash,
dataset revision, tokenizer, model mode, optimizer schedule, world size,
hardware, precision, effective batch, selected kernel path, all checkpoint
metrics, selection rule, exported SafeTensors hash, and known limitations.
