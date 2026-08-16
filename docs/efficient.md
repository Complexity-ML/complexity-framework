# Efficient training

Efficiency depends on active parameters, sequence length, vocabulary size,
device kernels, loss implementation, and communication topology. Profile the
realized run instead of relying on a model-size label.

## Current LoRA-SFT controls

1. Use BF16 on supported devices with `--bf16`.
2. Enable activation checkpointing with `--grad-ckpt` when memory-bound.
3. Bound tied-head loss memory with `--loss-chunk-tokens`.
4. Keep the SFT loss in FP32 with `--sft-fp32-loss` when stability matters.
5. Use `--use-custom-kernels auto` on tested NVIDIA CUDA environments.
6. Use DDP when one complete model fits on every device.
7. Measure the full eval and checkpoint boundary separately from steady-state
   training throughput.

## Full-shard loss weighting

Two-dimensional weighting does not alter row count or forward-pass cost. Every
example is still consumed. It changes per-example loss coefficients and
normalizes by visible weighted-token mass. Extremely sparse task targets can
increase gradient variance, so production curricula enforce
`max_task_loss_weight` and fail before training when the cap is exceeded.

See [Two-dimensional full-shard SFT weighting](sft-full-shard-2d-weighting.md).

## DDP

`scripts/run_sft_curriculum.py --world-size 8` launches eight processes through
`torch.distributed.run`. Batch size is per rank. The loader shards one shuffled
full-shard stream deterministically across ranks and computes the exact
complete-pass boundary.

TP/PP/DP files under `configs/run_configs/` are planning contracts validated by
`cf-plan-cluster`; they need an external launcher.

## TR-MoE kernels

The portable path uses PyTorch/autograd. NVIDIA CUDA can select custom
Triton/CUDA paths when shape, dtype, and capability checks pass. ROCm remains
on the conservative fallback under `auto` unless explicitly opted in. Never
infer the selected backend from the requested flag; record the runtime backend
metadata and `TRHashEngineMLP.engine.last_backend`.

See [GPU and dispatch paths](cuda.md).

## Measurement protocol

Record:

- framework and runtime commits;
- device model and count;
- selected kernel path;
- total and trainable parameter counts;
- vocabulary, sequence length, and batch per rank;
- precision and loss precision;
- gradient checkpointing;
- loss chunk size;
- telemetry and compilation state;
- warm-up exclusion and synchronization boundaries;
- whether evaluation or checkpoint I/O is included.

Do not compare training throughput that includes a complete held-out evaluation
against a steady-state interval that excludes it.
