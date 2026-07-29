# GPU and dispatch paths

Complexity Framework uses PyTorch for CPU, Apple MPS, NVIDIA CUDA, and AMD
ROCm. PyTorch exposes ROCm devices through `torch.device("cuda")`; the
framework records the logical backend separately.

## Install the correct PyTorch build

Use the helper:

```bash
./scripts/install_backend.sh cpu
./scripts/install_backend.sh cuda
./scripts/install_backend.sh rocm
```

or the Make targets:

```bash
make install-cpu
make install-cuda
make install-rocm
```

Review the selected wheel index before using these commands on a managed
cluster.

## Attention

GQA and MHA use PyTorch scaled dot-product attention when `use_sdpa=True`.
Available kernels depend on the installed PyTorch build, device, dtype, shape,
and mask.

The framework requests Flash, efficient, and math SDPA backends when available.
Override the preference list with:

```bash
export COMPLEXITY_SDPA_BACKENDS=flash,efficient,math
```

This is a preference, not proof that a specific kernel was selected. Record
profiler evidence when publishing throughput.

## TR-MoE dispatch

### `masked_dense`

Portable PyTorch/autograd path used when the custom grouped-GEMM path is
unavailable or disabled.

```yaml
run:
  use_custom_kernels: false
  cggr: false
```

### `cggr`

Autograd-aware CUDA/Triton grouped GEMM for routed experts. Selection requires:

- CUDA-visible tensors;
- custom kernels enabled;
- the `complexity_cuda` Triton import;
- the CGGR autograd wrapper;
- non-static dispatch.

```yaml
run:
  use_custom_kernels: auto
  cggr: auto
```

The selected path is logged and exposed as `TokenRoutedMLP.last_dispatch_path`.

### ROCm policy

`auto` leaves custom Triton disabled on ROCm by default. Opt in only after
testing the installed ROCm/Triton combination:

```bash
export COMPLEXITY_ALLOW_ROCM_TRITON=1
```

or:

```bash
cf-o200k-pretrain ... --use-custom-kernels true --cggr true
```

The opt-in is experimental and should be benchmarked against the fallback.

### Static dispatch

`static_expert_capacity=True` selects an export-friendly route intended for
pipeline tracing. It disables CGGR in the current implementation.

## Large-vocabulary loss

With o200k, materializing all logits for every token can dominate memory.

- `--loss-backend chunked` computes exact tied-head CE in token chunks.
- `--loss-backend liger` uses fused linear CE when installed.
- `--loss-backend auto` selects Liger when importable.

Record the active backend from `run_config.json`.

## Memory controls

- lower `batch_size` or `seq_len`;
- enable gradient checkpointing;
- use `shared_expert_chunk_tokens` for the dense shared path;
- use chunked or fused linear CE;
- use BF16 where supported;
- disable MoE telemetry for throughput measurements;
- avoid forcing cache emptying at short intervals unless diagnosing pressure.

## `torch.compile`

The runner supports:

```bash
--compile --compile-mode default
```

Compilation is shape- and backend-sensitive. The first step includes compile
time, so exclude warm-up from steady-state throughput. Verify numerical parity
before comparing speed.

## Serving

The repository provides checkpoint export and OpenAI-compatible clients.
Upstream vLLM/SGLang does not automatically understand a custom TR-MoE
checkpoint. A serving integration must implement the architecture and its
route tables.

No device-specific tokens/s value is treated as universal documentation.
Always report hardware, dtype, batch/concurrency, prompt length, generated
length, quantization, and runtime commit.
