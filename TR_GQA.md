# Matched TR-GQA MPS pilot

> **Historical evidence.** This pilot predates the canonical
> `TRHashEngineMLP` implementation and current training entrypoints. Its values
> are preserved as bounded evidence and are not current performance claims.

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

## Learned contextual-router control

A matched local control replaces only the fixed token-ID lookup with a learned
contextual top-2 router. The GQA backbone, shared path, expert tensors, data
order, seed, optimizer, token budget, and evaluation protocol remain fixed.
Common expert and shared-path tensors are initialized identically under each
paired seed. The learned router adds a 384-by-4 projection: 15,360 parameters
over 10 layers, or 0.0154% above the 99,487,680-parameter reference. Its
differentiable load-balancing loss uses coefficient 0.01.

| Seed | Dense GQA | Learned contextual top-2 | Fixed token-ID top-2 | Fixed vs dense | Fixed vs learned |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 7.596686 | 7.602109 | **7.536167** | -0.060519 | -0.065942 |
| 43 | 7.530082 | 7.548665 | **7.492290** | -0.037792 | -0.056375 |

At this short budget, fixed token-ID routing is lower in NLL than the learned
contextual control for both seeds. The mean fixed-versus-learned difference is
-0.061159 NLL. This is a two-seed local pilot, not evidence that fixed routing
is generally superior to learned routing.

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
