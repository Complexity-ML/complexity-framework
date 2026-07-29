# Matched TR-GQA MPS pilot

This note documents a bounded architecture-selection pilot and one independent
seed confirmation. It does not establish scaling, statistical significance, or
general superiority.

## Question

For a parameter-matched GQA decoder, does replacing part of a dense SwiGLU FFN
with a shared-plus-routed TR-MoE branch improve next-token NLL at the same
training-token budget?

## Shared protocol

- 99,487,680 trainable parameters;
- hidden size 384, 10 layers, 8 query heads, and 2 key/value heads;
- o200k tokenizer with vocabulary size 200,019;
- 250 steps × batch size 4 × sequence length 1,024 = 1,024,000 training
  tokens;
- AdamW, learning rate 3e-4, weight decay 0.1;
- Apple MPS, PyTorch fallback kernels;
- 4 evaluation batches every 50 steps;
- no intermediate checkpoint;
- local 400-document FineWeb-Edu sample containing 356,120 tokens;
- deterministic 95% training / 5% evaluation-tail split;
- source SHA-256
  `76f0b1a36b614c316c1d3624224c4431e0750feb7d387dc4b3b1f9263d58bdd6`.

The evaluation tail is excluded from gradient updates and from construction of
the routing-frequency table. It is still a small tail of the same source
sample, not a separate held-out corpus.

## Architectures

Dense GQA uses a 1,648-unit dense SwiGLU FFN. Each TR-GQA variant preserves the
same total stored FFN width:

```text
shared width + routed width = 1,648
```

TR-GQA uses four experts, fixed top-2 routes, fixed 0.5/0.5 route weights, and
`modulo_balanced_secondary`. There is no learned router and no learned
shared/routed gate.

## Seed-42 width sweep

| Routed width | Shared width | Evaluation NLL | Evaluation PPL |
| ---: | ---: | ---: | ---: |
| Dense GQA | 1,648 | 7.596686 | 1991.58 |
| 64 | 1,584 | 7.598139 | 1994.48 |
| 128 | 1,520 | 7.570862 | 1940.81 |
| 160 | 1,488 | 7.547887 | 1896.73 |
| **256** | **1,392** | **7.536167** | **1874.63** |

This sweep selected routed width 256. Its seed-42 difference from Dense GQA is
-0.060519 NLL. Because the same evaluation tail selected the width, this
comparison is exploratory.

## Seed-43 confirmation

The selected width-256 configuration and Dense GQA were rerun from seed 43
without changing the protocol.

| Architecture | Evaluation NLL | Evaluation PPL |
| --- | ---: | ---: |
| Dense GQA | 7.530082 | 1863.26 |
| **TR-GQA, routed width 256** | **7.492290** | **1794.16** |

The seed-43 difference is -0.037792 NLL in favor of TR-GQA. Across the two
reported seeds, the direction is consistent. The mean paired difference is
-0.049156 NLL, but two seeds are insufficient for a statistical claim.

## Required falsification work

The next useful controls are:

1. add at least one more predeclared seed without further width selection;
2. evaluate frozen checkpoints on a genuinely separate corpus;
3. compare at equal wall-clock or compute budget in addition to equal tokens;
4. test fixed modulo-adjacent, random, and dense-residual controls at width 256;
5. repeat at a longer token budget to determine whether the early advantage
   persists.

Machine-readable final values are in
[`results/tr_gqa_mps_100m.csv`](results/tr_gqa_mps_100m.csv).
