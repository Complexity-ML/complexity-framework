# Efficient training

Efficiency depends on active parameters, sequence length, vocabulary size,
device kernels, loss implementation, and communication topology. Profile the
realized run instead of relying on a model-size label.

## Current 200M full-SFT controls

1. Use BF16 on supported devices with `--bf16`.
2. Bound tied-head loss memory with `--loss-chunk-tokens`.
3. Use `--full-parameter` explicitly for the released recipe.
4. Require Liger and `--use-custom-kernels true` in the production profile;
   retain PyTorch fallback as the numerical regression reference.
5. Use DDP when one complete model plus optimizer state fits on every device.
6. Tune the per-rank microbatch from measured VRAM, then use gradient
   accumulation only to reach the intended global batch.
7. Measure the full eval and checkpoint boundary separately from steady-state
   training throughput.

Activation checkpointing and FP32 chunked SFT loss remain available for
memory- or stability-sensitive experiments, but they were not the defining
settings of the promoted 200M full-SFT run.

## Full-shard loss weighting

Two-dimensional weighting does not alter row count or forward-pass cost. Every
example is still consumed. It changes per-example loss coefficients and
normalizes by visible weighted-token mass. Extremely sparse task targets can
increase gradient variance, so production curricula enforce
`max_task_loss_weight` and fail before training when the cap is exceeded.

This optional weighting mechanism remains available through the generic SFT
runner, but the retired 500M experiment-specific launchers are no longer kept.

## DDP

The released launcher uses `torch.distributed.run` with one process per GPU.
Batch size is per rank. The loader shards one shuffled stream deterministically
across ranks and computes the exact complete-pass boundary. The historical
`scripts/run_sft_curriculum.py --world-size 8` path follows the same per-rank
batch convention for 500M LoRA reproduction.

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
