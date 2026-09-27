---
license: other
library_name: complexity
language:
- en
- fr
pipeline_tag: text-generation
---
# TR-HASH MoE 1B — 70B-token pretraining

From-scratch pretraining of a 1,011,823,104-parameter TR-HASH model with 8 routed experts (top-2), a shared branch, and full causal attention at 16,384 tokens. Training is in progress; this is not an instruction-tuned release.

The fixed schedule contains 534,057 optimizer updates of 131,072 tokens, or 69,999,919,104 tokens. It selects corpus positions without intentional replay from the pinned AETHORIA 125B Agentic 32K corpus, maintaining its source proportions. Global text deduplication is not asserted. Each source reserves its first 131,072 positions for validation. Shard tails shorter than a full 16K sequence are excluded; sequences never cross shard boundaries.

`recipe.json`, `data-plan.json`, the schedule arrays and the tokenizer manifest pin the training and data contracts. Training uses four RTX 5090 GPUs, FSDP2, FP32 sharded master weights and AdamW state, BF16 compute, Liger fused loss and the framework's TR-HASH CUDA kernels. Learning rate: linear warmup over 2,000 updates to 3e-4, then cosine decay to 3e-5; AdamW betas 0.9/0.95, weight decay 0.1, global gradient clipping 1.0.

## Checkpoints and resumption

`step_XXXXXXX/` contains a complete distributed model/optimizer checkpoint, RNG state for all four ranks, the next update and per-source position metadata, and a SHA-256 manifest. `latest.json` points to the latest checkpoint whose remote file sizes and hashes were verified. The latest three regular checkpoints and latest interruption checkpoint are retained (at most four distinct snapshots). A checkpoint saved at a regular boundary during interruption can occupy both roles. Intermediate checkpoints are resumable PyTorch distributed states; they are not standalone Transformers weights. Hosted snapshots use `distributed/metadata.json` with an explicit JSON schema and the provided reader in `checkpoint_json.py`. Original `.metadata` files stay local. Tensor shards and RNG files retain their PyTorch serialization; this is not a fully pickle-free checkpoint format. Final consolidated weights are exported only after completion.

Snapshots are saved after startup verification, at update 10, every 1,000 updates, and on a checkpoint request. An upload worker atomically adds each completed snapshot, removes obsolete snapshot paths and updates `latest.json`. It verifies the surviving remote snapshots before replacement and verifies the resulting snapshot set afterward. The uploader purges obsolete Git/LFS history after each eviction; quota updates can take up to 36 hours on Hugging Face. `latest.json` uses a manifest digest and the same repository snapshot as its checkpoint, so a history squash does not invalidate restoration. A backlog of three unuploaded snapshots pauses training until upload catches up.

For a new instance, copy the contents of the Hub `training/` directory into `/workspace/pretrain-1b-70b/`, install the pinned environment, and provide the private credential file before restoring. The dataset prefetch looks up to 1,536 updates ahead, refreshed every 32 updates, within the same bounded cache.

On the configured instance:

```bash
supervisorctl status pretrain_1b pretrain_upload
tail -f /var/log/portal/pretrain_1b.log
tail -f /var/log/portal/pretrain_upload.log
```

Request a checkpoint and continue:

```bash
touch /workspace/pretrain-1b-70b/runtime/request-checkpoint
```

Request a checkpoint and stop at an update boundary:

```bash
touch /workspace/pretrain-1b-70b/runtime/request-stop
```

Wait for `graceful_stop` in the training log and `upload_verified` for that step before stopping or replacing the instance. Resume on the same instance by removing `runtime/request-stop` and starting `pretrain_1b`. A fresh instance must install the pinned framework/CUDA environment and download this bundle; `restore.py` downloads and verifies the latest complete checkpoint before `run.sh` resumes it. This recipe requires the same four-rank layout and batch contract.

The operational policy lives in `retention.json`; changing it does not change the training recipe or dataset positions.

The instance-side `runtime/hf-token` is a private credential file (mode 600) and is never included in this repository.
