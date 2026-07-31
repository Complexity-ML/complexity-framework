# TR-Hash Engine

`complexity.tr_hash` is the execution contract for deterministic token-ID
routed residual experts. It is general inside the TR-Hash family; it is not a
learned contextual Mixtral router.

## Supported model matrix

| Axis | Contract |
|---|---|
| Experts | 2, 4, 8, 16 |
| Active experts | top-1, top-2, top-4 |
| Widths | independent shared and per-expert widths |
| Attention input | GQA or MHA hidden states |
| Hashes | modulo-cyclic, balanced token-ID hash, affine token-ID hash |
| Phase | training and inference |
| Dense precision | FP32, BF16, FP16 |
| Parallelism | one device or replicated DDP |
| CUDA | hash-native fused top-2, general CGGR/Triton, or PyTorch |
| CUDA Graph | inference buckets with persistent static buffers |

FP8 and INT8 are named in the public precision contract but intentionally fail
with an explicit error until the phase-2 quantized grouped-GEMM kernel exists.
The engine must not silently dequantize and claim a quantized fast path.

## Computation

For contextual hidden state \(x\), token ID \(t\), shared branch \(S\), expert
\(E_e\), fixed layer route \(r_{\ell,j}\), and fixed route weights \(a_j\):

\[
y =
\alpha S(x)
+ \beta \sum_{j=1}^{k}
a_j E_{r_{\ell,j}(t)}(x).
\]

The hash selects parameters. Both the shared and selected residual branches
still transform the contextual hidden state produced by GQA or MHA.

## Example

```python
import torch

from complexity.tr_hash import (
    AttentionBackbone,
    TRHashEngine,
    TRHashEngineConfig,
)

engine = TRHashEngine(
    TRHashEngineConfig(
        hidden_size=512,
        vocab_size=200_019,
        num_experts=8,
        top_k=2,
        shared_width=2_496,
        expert_width=64,
        attention_backbone=AttentionBackbone.GQA,
    )
)

hidden = torch.randn(2, 128, 512)
token_ids = torch.randint(0, 200_019, (2, 128))
output = engine(hidden, token_ids)
```

The same module can be wrapped in PyTorch DDP. `world_size` is recorded in the
engine manifest; it does not change route identity.

## Backend selection

`backend="auto"` first selects the hash-native fused CUDA path for top-2 with
two to four experts. That path compiles each vocabulary route into one byte,
fuses hash decoding with the counting partition, reads the original hidden
states indirectly for both input projections, keeps the down projection
expert-contiguous, and fuses route unpermutation with the weighted reduction.
Training uses the autograd-aware grouped projections; inference additionally
uses the fused Triton SwiGLU activation.

Other supported expert/top-k shapes use general CGGR/Triton when available.
CPU, MPS, and CUDA installations without the custom kernels select the
universal PyTorch reference and report the reason in `capability_summary()`.

The universal implementation is the numerical reference for every expert and
top-k combination. Optimized kernels must match it before being enabled.

## CUDA Graph buckets

Inference graphs require:

- `phase="inference"`;
- `backend="cuda_graph"`;
- one or more `(batch, sequence)` buckets;
- `eval()` and `torch.no_grad()`.

The runner selects the smallest containing bucket, copies the request into
persistent buffers, replays the captured graph, and returns the unpadded slice.
This keeps graph shapes and memory addresses stable. Graph capture currently
uses the allocation-free fixed expert loop; capture of the hash-native
partition pipeline remains a separate optimization because its temporary
partition buffers require a persistent workspace.

```python
from complexity.tr_hash import GraphBucket, TRHashBackend, TRHashPhase

config = TRHashEngineConfig(
    hidden_size=512,
    vocab_size=200_019,
    phase=TRHashPhase.INFERENCE,
    backend=TRHashBackend.CUDA_GRAPH,
    graph_buckets=(
        GraphBucket(1, 128),
        GraphBucket(1, 512),
        GraphBucket(4, 512),
    ),
)
```

## Kernel milestones

1. **Implemented:** general reference, arbitrary supported E/top-k, autograd,
   route validation, CGGR selection and combined top-k grouped dispatch.
2. **Implemented:** hash-native fused CUDA path for top-2 and up to four
   experts, including compact routes, linear partition, indirect projections,
   fused inference SwiGLU, and fused weighted route reduction.
3. **Implemented:** static CUDA Graph bucket manager using persistent buffers.
4. **Next:** generalize the hash-native partition workspace to 8/16 experts
   and top-1/top-4, then capture it inside CUDA Graph buckets.
5. **Then:** FP8 and INT8 grouped-GEMM specializations.
6. **Then:** expert-parallel all-to-all; replicated multi-GPU already uses DDP.
