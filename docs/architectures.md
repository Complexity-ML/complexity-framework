# Architecture Overview

## Complexity-Deep

![Architecture](../figures/architecture_complexity_deep.png)

### Decoder Layer

Each of the 18 decoder layers:

1. **RMSNorm + GQA Attention** (12 Q heads, 4 KV heads, head_dim=64)
   - Mu-Guided Q/K/V bias from previous layer
   - QK RMSNorm + RoPE (theta=10000)
   - Residual connection

2. **RMSNorm + Token-Routed MLP** (4 experts SwiGLU, 512d each)
   - Sort-and-split dispatch (bmm, fullgraph safe)
   - Zipf-balanced deterministic routing
   - Shared Lexical Expert (dense SwiGLU, all tokens)
   - Residual connection

3. **Mu-Guidance** (after MLP)
   - mu = clamp(mu_param + mu_proj(h), -2, 2)
   - Flows to next layer's attention

### Specs (187M)

| Component | Value |
|-----------|-------|
| Hidden size | 768 |
| Layers | 18 |
| Attention | GQA (12h / 4kv) |
| MLP | Token-Routed (4 experts) |
| Expert size | 512 |
| Shared expert | Yes |
| Routing | Zipf bin-packing |
| Mu-Guidance | Yes |
| Vocab | 32k BPE |

## Supported Architectures

The framework also supports:

| Architecture | Type | Use Case |
|-------------|------|----------|
| Dense SwiGLU | Standard MLP | Baseline comparison |
| Mixtral MoE | Learned router | MoE baseline |
| Mamba | SSM (O(N)) | Long sequences |
| RetNet | Retention | Efficient inference |
| RWKV | Linear attention | Low memory |

## See Also

- [Token-Routed MLP](token-routed.md)
- [Mu-Guidance](dynamics.md)
- [Training](training.md)
