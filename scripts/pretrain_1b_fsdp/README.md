# TR-HASH 1B pretraining — 70B tokens

This four-rank FSDP recipe uses the framework's composed-modulo TR-HASH:
four query heads, one KV head, four experts and two routes per token.
Refinement and instruction SFT are separate stages.

| Setting | Value |
|---|---:|
| Model repository | `Pacific-i64/TR-HASH-MoE-1B-70B-Agentic-Pretraining` |
| World size | 4 |
| Per-rank microbatch | 2 |
| Context | 16,384 |
| Tokens per optimizer update | 131,072 |
| Optimizer updates | 534,057 |
| Scheduled token exposure | 69,999,919,104 (approximately 70B) |

`model.json` defines the current TR-HASH implementation; `recipe.json` pins
its exact hash. The data plan, token budget, optimizer and schedule are
unchanged. `data-plan.json`, `schedule.npy`, `source-update-offsets.npy` and
the tokenizer must match the checkpoint being restored.

Resume accepts only a checkpoint with the same model and recipe fingerprint.
Checkpoints from the retired routing implementation are rejected rather than
loaded into a different architecture.

## Inspect without starting a run

From the framework root:

```bash
complexity train inspect scripts/pretrain_1b_fsdp/recipe.json --json
complexity checkpoint inspect /path/to/complete/step_directory \
  --recipe scripts/pretrain_1b_fsdp/recipe.json --verify --json
```

The first command reports the token arithmetic and available linked assets.
The second checks checkpoint JSON, rank files and manifest integrity without
loading tensors. Neither starts training, downloads a checkpoint or uploads
anything. File integrity alone does not demonstrate a successful FSDP restore.

## Resume later on the selected GPU server

Use the CUDA/PyTorch environment matched to the original runtime, with
Triton, Liger, Hugging Face Hub, NumPy and TensorBoard. Authenticate through
`hf auth login`. Choose one pinned repository revision that actually exists:
old revisions and old checkpoint paths may have been removed by retention.

The lightweight base install does not include Hub/data integrations or
TensorBoard. Install the `pretrain-data,tracking` extras in the matched backend
environment, retaining the original run's pinned runtime versions. The small
`complexity train demo` is an installation check, not this FSDP resume path.

1. Run `python scripts/pretrain_1b_fsdp/restore.py --revision <revision>`.
   The restore retrieves the original assets and verifies recipe, plan,
   schedule, checkpoint-file sizes and hashes. It refuses to overwrite an
   existing output checkpoint.
2. Inspect the restored complete checkpoint and read its actual step `N`.
   Never infer it from a historical example or regenerate assets with
   `prepare.py`.
3. Run `bash scripts/pretrain_1b_fsdp/run.sh --limit <N+2> --verify-resume`.
   Replace `<N+2>` with the numeric absolute target step. This verifies a
   bounded advance and save/reload of model, optimizer and rank RNG state.
   The learning-rate horizon remains the original 70B run; the launcher
   refuses an accidental zero-step start.
4. Once this bounded check passes on the intended four-rank environment,
   the original `run.sh` continues the latest complete local checkpoint.
   This is an operational action, separate from framework inspection.

Configure and verify the uploader before a sustained run. Training waits when
three complete local checkpoints have not been acknowledged as uploaded;
all ranks coordinate and poll every 30 seconds. During that wait:

- `runtime/request-stop` saves the current update and stops;
- `runtime/request-checkpoint` saves the current update, then rechecks upload
  capacity.

`upload.py` keeps three regular checkpoints and one interruption checkpoint,
removing older remote paths. Historical Hub revision purging is disabled by
`purge_history_after_eviction: false`. The uploader reads `runtime/hf-token`.
Inspection commands never start this uploader or apply its retention policy.

## Historical records

An audit on 27 September 2026 concerned `step_0018000` at revision
`0ba0e730f9ef078a669e5fd84277bdb60903f57f`, with 2,359,296,000 tokens consumed.
That is a historical observation, not the current resume target.
`README.original.md` and `TRAINING.md` describe the earlier server.
For shared model/export/resume terminology, read the
[framework workflows](../../docs/workflows.md).
