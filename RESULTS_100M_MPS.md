# Matched GQA and MHA MPS pilots

> **Historical evidence.** These measurements use architectures and a
> pretraining driver that have since been removed. They remain reproducibility
> records, not claims about the current implementation.

This page consolidates the recent 99,487,680-parameter comparisons between
dense SwiGLU and deterministic token-routed residual FFNs. Lower NLL and
perplexity are better.

## Common protocol

- local 400-document FineWeb-Edu sample;
- 356,120 total o200k tokens;
- deterministic 95% training / 5% evaluation-tail split;
- routing frequencies computed from the training partition only;
- 250 steps × batch size 4 × sequence length 1,024;
- 1,024,000 training tokens;
- AdamW, learning rate 3e-4, weight decay 0.1;
- hidden size 384 and 10 layers;
- seed 42 for width selection, followed by paired seed-43 confirmations;
- Apple MPS with PyTorch fallback kernels;
- no intermediate checkpoints.

The source file SHA-256 is
`76f0b1a36b614c316c1d3624224c4431e0750feb7d387dc4b3b1f9263d58bdd6`.

## Final seed-42 comparison

| Attention | Feed-forward path | Shared width | Routed width | Eval NLL | Eval PPL | NLL delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GQA, 8Q/2KV | Dense SwiGLU | 1,648 | 0 | 7.596686 | 1991.58 | — |
| GQA, 8Q/2KV | **TR-MoE** | 1,392 | 256 | **7.536167** | **1874.63** | **-0.060519** |
| MHA, 8Q/8KV | Dense SwiGLU | 1,456 | 0 | 7.586145 | 1970.70 | — |
| MHA, 8Q/8KV | **TR-MoE** | 1,296 | 160 | **7.536471** | **1875.20** | **-0.049674** |

Relative to its attention-matched dense baseline, evaluation perplexity is
5.87% lower for TR-GQA and 4.85% lower for TR-MHA.

## MPS throughput context

The median logged training throughput after step 10 was:

| Architecture | Median logged tokens/s | Difference from matched dense |
| --- | ---: | ---: |
| Dense GQA | 5,418.5 | — |
| TR-GQA | 4,708.0 | -13.11% |
| Dense MHA | 5,590.5 | — |
| TR-MHA | 4,398.5 | -21.32% |

These are MPS fallback measurements, not CUDA deployment benchmarks. They show
that the short-budget NLL gain is not a free throughput gain in the current
local implementation.

## Seed-43 confirmations

The routed widths selected with seed 42 were frozen and rerun against their
dense baselines with seed 43:

| Attention | Architecture | Eval NLL | Eval PPL | NLL delta |
| --- | --- | ---: | ---: | ---: |
| GQA | Dense | 7.530082 | 1863.26 | — |
| GQA | **TR-MoE** | **7.492290** | **1794.16** | **-0.037792** |
| MHA | Dense | 7.726971 | 2268.72 | — |
| MHA | **TR-MoE** | **7.541488** | **1884.63** | **-0.185483** |

The routed direction repeats for both attention families on a second
initialization. The mean paired NLL difference across the two seeds is
-0.049156 for GQA and -0.117579 for MHA. With only two seeds, the large
variation in the MHA differences is itself a reason not to claim statistical
significance.

## Learned contextual-router controls

The fixed TR-GQA and TR-MHA configurations were also compared with a learned
contextual top-2 router. Within each attention family, the shared path,
experts, backbone, seed, data order, optimizer, and token budget are identical.
The learned router adds only its 384-by-4 projection in each layer: 15,360
parameters, or 0.0154%.

| Attention | Seed | Dense NLL | Learned contextual NLL | Fixed token-ID NLL | Fixed improvement vs learned |
| --- | ---: | ---: | ---: | ---: | ---: |
| GQA | 42 | 7.596686 | 7.602109 | **7.536167** | 0.065942 |
| GQA | 43 | 7.530082 | 7.548665 | **7.492290** | 0.056375 |
| MHA | 42 | 7.586145 | 7.592847 | **7.536471** | 0.056376 |
| MHA | 43 | 7.726971 | 7.666093 | **7.541488** | 0.124605 |

Fixed token-ID routing has lower NLL than the learned contextual control in all
four runs. The learned control is slightly behind Dense in three runs and
ahead of Dense by 0.060878 NLL for MHA seed 43. These controls isolate the
routing signal more directly than a fixed-versus-dense comparison, but remain
short two-seed pilots on one small data sample. Machine-readable values are in
[`results/matched_gqa_mha_mps_100m.csv`](results/matched_gqa_mha_mps_100m.csv).

## Interpretation limits

The evaluation tail is excluded from training and route construction, but it
comes from the same small source file. Widths were selected using seed 42.
These pilots support further falsification; they do not demonstrate
generalization, scaling, or statistical significance.

The next strongest tests are:

1. add a third seed for both attention families;
2. evaluate frozen checkpoints on a separate corpus;
3. compare equal wall-clock and equal-compute budgets;
4. repeat at a longer training-token budget.
