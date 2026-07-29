# TR-MHA

TR-MHA is the GQA-free continuation of the token-routing experiments. It keeps
standard causal multi-head attention and changes only a small residual
projection path.

## Architecture

- Full causal MHA: 8 query heads, 8 key heads and 8 value heads.
- Shared dense Q, K, V and output projections remain active for every token.
- Token identity supplies two fixed, layer-specific adapter candidates.
- The current hidden state reweights those candidates without changing their
  identity.
- The selected top-2 route combines a shared low-rank Q/V residual.
- K remains shared in this first experiment.
- The MLP is a dense SwiGLU, so the pilot isolates attention-side routing.

TR-MHA v2 preserves the full dense MHA baseline exactly at initialization by
zero-initializing the routed up-projection. It computes only selected
expert/token pairs. That selected-only dispatch improves the pilot throughput,
but it must be replaced with a fused implementation before claiming static
graph capture.

## Bounded MPS pilots

Each approximately 100M-parameter pilot uses the same tokenizer, text stream,
seed, 1,024,000-token training budget and evaluation schedule. Only the final
checkpoint is saved.

```bash
cd /Users/boris/Dev/tr-mha
python3 -m complexity.training.o200k_pretrain \
  --config configs/run_configs/experiments_100m/100m_params_tr_mha_v2_mps.yaml
```

### Pilot result

| Architecture | Parameters | Final eval NLL | Eval PPL | Median logged tok/s |
| --- | ---: | ---: | ---: | ---: |
| GQA dense (8Q/2KV) | 99,487,680 | 7.359221 | 1570.61 | 5,647.5 |
| MHA dense (8Q/8KV) | 99,487,680 | 7.369812 | 1587.34 | 5,712.5 |
| TR-MHA v1 | 99,445,460 | 7.477369 | 1767.58 | 4,204.0 |
| TR-MHA v2 Q/V | 99,506,900 | 7.408230 | 1649.50 | 5,037.0 |
| TR-MHA v2 V-only | 99,503,060 | 7.410834 | 1653.81 | 5,163.5 |
| MHA + shared FFN + modulo/adjacent experts | 99,487,680 | 7.367669 | 1583.94 | 4,962.5 |
| MHA + shared-1,360 FFN + modulo/balanced experts | 99,487,680 | 7.383080 | 1608.54 | 4,177.5 |
| MHA + shared-1,328 FFN + modulo/balanced experts | 99,487,680 | 7.366338 | 1581.83 | 4,954.0 |
| MHA + shared-1,296 FFN + modulo/balanced experts | 99,487,680 | **7.321415** | **1512.34** | 4,776.5 |

The plain MHA control is only 0.010591 NLL behind GQA at this budget. TR-MHA v2
recovers 0.069139 NLL and about 19.8% median logged throughput relative to v1,
but remains 0.038418 NLL behind the matched MHA control and about 11.8% slower.
The V-only ablation is 0.002604 NLL behind Q/V. Removing the routed Q residual
therefore does not recover the dense MHA result, although it improves median
throughput by about 2.5%. The v2 verifier use stays near 0.09 and the four
aggregate route shares remain near 25%, so these runs do not yet demonstrate a
useful contextual routing effect.

### Shared path with deterministic token-ID experts

The strongest MHA pilot moves token routing from attention into the SwiGLU
feed-forward block:

- every token uses a shared width-1,296 SwiGLU path;
- the token ID selects a primary expert by modulo;
- a small offline table assigns a frequency-balanced secondary expert;
- the two selected width-40 experts are mixed 0.5/0.5;
- the full MHA path remains dense and unchanged.

The shared path plus all four experts has the same 99,487,680 parameters as the
dense MHA and GQA controls. Only the shared path and two experts are active for
a token, for a theoretical active FFN width of 1,376 instead of 1,456. The
balanced-secondary result improves on dense MHA by 0.048397 NLL and 75.00
perplexity points. It also improves on matched GQA by 0.037806 NLL and 58.27
perplexity points in this single-seed short pilot.

The current MPS fallback evaluates all four experts with masks, so its measured
throughput does not realize the 5.5% reduction in active FFN width. A selected
expert dispatch or fused grouped GEMM is required before making a runtime speed
claim.

These short pilots validate implementation, causal cache equivalence, route
telemetry and the direction of the next ablations. They are not evidence that
one architecture is generally better. A full-vocabulary and a hashed lexical
transition residual were also prototyped, but removed: they add deployment
state and did not improve the matched control in this short budget.
