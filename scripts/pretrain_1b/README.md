# TR-HASH 1B — 100B pretraining, 2 GPU DDP

Adapted from the existing `pretrain-1b-70b/train.py` recipe, formerly stored in
`Documents/Codex/2026-09-07/ok-x20/outputs/`. `model.json` is copied unchanged:
24 layers, hidden 1536, GQA 24/6, 8 experts/top-2, shared width 4352, total
routed width 3072, context 16384, vocabulary 32000. No refinement is launched.

Training uses the framework's `TrainRunner`, `Trainer`, `CheckpointManager`
and `sync_checkpoints_to_hf.sync_once`. The original 16K reader is retained:
no cross-shard concatenation, deterministic shuffled source schedule, no
intentional replay, fixed held-out prefix per source. Corpus weights are unchanged.

The original global update remains 131072 tokens: 2 GPUs × microbatch 2 ×
accumulation 2 × 16384. The 100B target is rounded down to complete updates:
762939 updates = 99,999,940,608 tokens. The LR uses the framework's cosine
scheduler with 2000 warmup updates, peak 3e-4, AdamW betas 0.9/0.95.
Checkpoint boundaries cross each multiple of 5B tokens, then save at completion.
There is no mandatory checkpoint after 10 or 1000 updates.

After moving to a new server, prepare the identical recipe and restore the
last uploaded snapshot with `python -m scripts.pretrain_1b.sync --restore`,
then launch normally. The download uses one pinned Hub revision and checks
every checkpoint file hash before making the snapshot available for resume.

## Installation and preparation

On the rented Linux server, copy this repository including these uncommitted
changes. A fresh clone of main does not yet contain this recipe. Install a
PyTorch CUDA build supporting this Blackwell GPU, then the framework:

```bash
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install_backend.sh cuda
hf auth login
export TR_HASH_RUN_DIR=/workspace/tr-hash-1b-100b
python -m scripts.pretrain_1b.prepare --create-repo
```

Preparation downloads only pinned metadata/tokenizer and writes the schedule.
The corpus is streamed through a bounded 64 GiB cache. Keep the same
`TR_HASH_RUN_DIR` in all shells. Reserve substantial persistent disk space for
three full checkpoints, the checkpoint being written, the token cache and the
final export. Do not delete this directory when stopping the rental.

## GPU validation, launch and resume

Start the uploader in another persistent terminal/session with the same virtual
environment and `TR_HASH_RUN_DIR`:

```bash
python -m scripts.pretrain_1b.sync
```

First run two real updates and stop at a complete optimizer boundary:

```bash
bash scripts/pretrain_1b/run.sh --stop-after 2
```

Inspect loss, GPU memory and logs, then relaunch to verify that the runner
resumes update 2 and stops after update 3:

```bash
bash scripts/pretrain_1b/run.sh --stop-after 3
```

Launch the full pretraining after this GPU gate:

```bash
bash scripts/pretrain_1b/run.sh
```

`--resume auto` is the default. The same recipe, world size, batch and schedule
are required. SIGINT/SIGTERM request a checkpoint at the next full update;
SIGKILL/power loss recover only the last committed checkpoint. Optimizer load
errors fail the run rather than silently resetting moments. Hidden staging
directories are never eligible for resume or upload. RNG is saved per rank.

The uploader preserves the three latest uploaded checkpoints locally, plus
`final/`. If upload fails, unsent checkpoints remain locally (possibly more than
three). Remote history is not purged. Final weights are exported only when the
full token target is reached; a bounded test is not labelled final.

No RTX PRO 6000 run or throughput estimate is claimed from CPU tests. The 16K
microbatch still needs the actual two-GPU memory/communication validation above.
