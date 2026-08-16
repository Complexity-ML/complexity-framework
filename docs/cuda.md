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

# Detector/vision profile: Torchvision is resolved from the same backend index.
./scripts/install_backend.sh cuda vision
./scripts/install_backend.sh rocm vision
```

or the Make targets:

```bash
make install-cpu
make install-cuda
make install-rocm
make install-vision-cuda
make install-vision-rocm
```

The vision profile also installs the OpenCV, augmentation, COCO-evaluation,
monitoring and export primitives documented in
[`vision-dependency-stack.md`](vision-dependency-stack.md).

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

### `pytorch`

Portable PyTorch/autograd path used when the custom grouped-GEMM path is
unavailable or disabled.

```yaml
run:
  use_custom_kernels: false
  use_cggr: false
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
  use_cggr: auto
```

The selected path is logged and exposed as `TRHashEngineMLP.engine.last_backend`
(or `TRHashEngineMLP.capability_summary()["selected_backend"]`).

### ROCm policy

`auto` leaves custom Triton disabled on ROCm by default. Opt in only after
testing the installed ROCm/Triton combination:

```bash
export COMPLEXITY_ALLOW_ROCM_TRITON=1
```

or, in code, via `ModelConfig(use_custom_kernels=True, use_cggr=True)` (the
`--use-custom-kernels`/`--cggr` CLI flags belonged to the removed
`cf-o200k-pretrain`; see [`training.md`](training.md)).

The opt-in is experimental and should be benchmarked against the fallback.

### Static dispatch

`static_expert_capacity=True` selects an export-friendly route intended for
pipeline tracing. It disables CGGR in the current implementation.

## Large-vocabulary SFT loss

Materializing all logits for every token can dominate memory. The current
LoRA-SFT runner computes tied-head loss in token chunks; configure the chunk
size with `--loss-chunk-tokens`. `--sft-fp32-loss` keeps that loss calculation
in FP32 for stability while the model runs under BF16 autocast. The historical
`--loss-backend` selector belonged to the removed pretraining runner.

## Memory controls

- lower `batch_size` or `seq_len`;
- enable gradient checkpointing;
- use `shared_expert_chunk_tokens` for the dense shared path;
- use chunked tied-head CE;
- use BF16 where supported;
- disable MoE telemetry for throughput measurements;
- avoid forcing cache emptying at short intervals unless diagnosing pressure.

## Compilation

`torch.compile` remains shape- and backend-sensitive but is not exposed by the
current LoRA-SFT CLI. Custom-kernel selection is controlled with
`--use-custom-kernels {auto,true,false}`. Exclude initialization and warm-up
from throughput measurements and verify numerical parity before comparing
paths.

## Serving

The repository provides checkpoint export and OpenAI-compatible clients.
Upstream vLLM/SGLang does not automatically understand a custom TR-MoE
checkpoint. A serving integration must implement the architecture and its
route tables.

No device-specific tokens/s value is treated as universal documentation.
Always report hardware, dtype, batch/concurrency, prompt length, generated
length, quantization, and runtime commit.
