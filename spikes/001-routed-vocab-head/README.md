# Spike 001 — Block-routed o200k vocabulary head

## Question

Given a batch-1 hidden state `[1, 384]` and an o200k output vocabulary, does selecting a few lexical blocks before computing logits reduce M5 latency and weight traffic versus a dense LM head?

## Prototype

- Apple M5 / MPS
- FP16 weights
- Vocabulary: 200,019
- Hidden size: 384
- 256 contiguous blocks of at most 782 tokens
- Learned linear router `384 → 256`
- 200 synchronized latency samples after warm-up
- The router is random: this spike measures mechanics and latency, not target coverage or language-model quality.

## Results

| Mode | Selected tokens | Approx. head weights touched | Median latency | Speedup |
|---|---:|---:|---:|---:|
| Dense | 200,019 | 100% | 0.807 ms | 1.00× |
| 1 block | 782 | 0.52% | 0.314 ms | 2.57× |
| 2 blocks | 1,564 | 0.91% | 0.344 ms | 2.35× |
| 4 blocks | 3,128 | 1.69% | 0.406 ms | 1.99× |
| 8 blocks | 6,256 | 3.26% | 0.553 ms | 1.46× |
| 16 blocks | 12,512 | 6.38% | 0.887 ms | 0.91× |

Selected local logits match the corresponding dense-head logits numerically.

## Verdict: PARTIAL

### What worked

- A static block-routed head is straightforward with ONNX-friendly primitives: Linear, TopK, Gather, MatMul/einsum.
- Four blocks reduce candidate logits from 200,019 to 3,128 and approximate head weight traffic by 98.3%.
- The unoptimized PyTorch/MPS prototype nearly halves median head latency at four blocks.

### What did not yet work

- Reduced weight traffic does not translate proportionally to latency. Top-k, gather, dispatch, synchronization, and small matrix operations dominate on MPS.
- At 16 blocks the routed implementation is slower than the dense MPS matrix multiplication.
- A random router has only `top_k / 256` expected block coverage. Quality is completely unvalidated.

### Recommendation

- Treat four blocks / roughly 3k candidates as the first quality-performance operating point.
- Train the router with an auxiliary target-block recall objective and report Recall@1/2/4/8 before replacing the dense head.
- Partition vocabulary by learned lexical/embedding clusters rather than contiguous token IDs.
- During training, retain the exact dense loss or use sampled/clustered softmax with explicit probability correction; do not silently train a locally normalized objective.
- For RK1828/ONNX, export explicit static outputs `(block_ids, local_token_ids, local_logits)` and benchmark the target NPU. A fused route+gather+matmul kernel is likely required for gains beyond the observed M5 2×.
