# Efficient training

Efficiency depends on active parameters, sequence length, vocabulary size,
device kernels, and communication topology. Profile the realized run instead
of relying on a model-size label.

## First controls to use

1. BF16 on supported devices.
2. Chunked or fused linear cross entropy for large vocabularies.
3. Shared-expert token chunking.
4. Gradient checkpointing when activation memory dominates.
5. DDP when the model fits on one device.
6. CGGR only after parity and throughput validation.

## BF16

The `cf-o200k-pretrain` runner this referred to was removed (see
[`training.md`](training.md)); backend-appropriate autocast is still
available directly via `complexity.utils.autocast`. BF16 support and
performance vary by device.

## Gradient checkpointing

Enabled by default in the runner:

```bash
--grad-ckpt
```

Disable for a small model that fits comfortably:

```bash
--no-grad-ckpt
```

Checkpointing trades additional forward computation for lower activation
memory.

## Shared-path chunking

TR-MoE's shared branch touches every token. Limit its peak activation memory:

```bash
--shared-expert-chunk-tokens 32768
```

Set zero for one dense pass. Chunking preserves the mathematical output.

## Large-vocabulary loss

```bash
--loss-backend auto --loss-chunk-tokens 1024
```

See [GPU and dispatch paths](cuda.md).

## DDP versus cluster plans

Use DDP when each worker can hold a complete model:

```bash
torchrun --nproc_per_node=8 \
  -m complexity.training.o200k_pretrain \
  --config path/to/direct-run.yaml
```

TP/PP/DP files are planning contracts validated with `cf-plan-cluster`; they
need an external cluster launcher.

## Measurement protocol

For a defensible throughput result, record:

- framework and PyTorch commits;
- device model and count;
- backend and kernel path;
- model and active parameter counts;
- vocabulary and sequence length;
- batch per device and world size;
- precision;
- gradient checkpointing;
- loss backend;
- telemetry state;
- compilation state;
- warm-up exclusion;
- synchronization boundaries.

Do not compare a telemetry-enabled run against a telemetry-disabled run as if
the numbers measured only architecture.
