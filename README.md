# Complexity Framework v0.4.0

**Modular Python framework for building LLMs with Token-Routed MoE, Mu-Guidance, and Zipf-balanced routing.**

```bash
pip install complexity-framework
```

## What's New in v0.4.0

- **Zipf-balanced routing**: frequency-aware expert assignment (no more load imbalance)
- **GPT-style residual init**: `1/√(2N)` scaling on output projections
- **3D cluster parallelism**: TP × PP × DP for multi-node training
- **Dynamic warmup**: auto 5% of total steps
- **LR auto-scaling**: `√(effective_batch/64)` prevents explosion with large batches
- **WSD scheduler**: Warmup-Stable-Decay (LLaMA 3 style)
- **Muon optimizer**: Newton-Schulz orthogonalization for 2x convergence
- **OmniModel**: any-to-any multimodal (text + image + audio + video)

## Quick Start

```python
from complexity.models import ComplexityModel
from complexity.config import ModelConfig

# 150M Token-Routed model with Mu-Guidance
config = ModelConfig(
    hidden_size=768,
    num_hidden_layers=18,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2048,
    vocab_size=32000,
    mlp_type="token_routed",
    num_experts=4,
    use_mu_guidance=True,
)
model = ComplexityModel(config)
# 170M params, deterministic routing, no auxiliary losses
```

## Architecture

```
Input
  │
  ▼
[Embed] ──► mu_init (learnable, layer 0 guidance)
  │              │
  ▼              ▼
[RMSNorm] ─► [Mu-Guided GQA] ─► [MuGuidance] ─► [RMSNorm] ─► [Token-Routed MLP]
  │              ▲                    │                            ▲
  │              │                    │                            │
  │         mu_prev              mu_contextual                Zipf-balanced
  │                                   │                       expert dispatch
  +────────── Residual ──────────────┼─────────── Residual ───────+
  │                                   │                            │
  ▼                                   ▼                            │
Output ◄────────────────────── mu_next (to next layer) ◄──────────┘
```

## Key Innovations

### 1. Token-Routed MLP (Deterministic MoE)

```python
from complexity.core.mlp import TokenRoutedMLP, MLPConfig

config = MLPConfig(
    hidden_size=768,
    intermediate_size=3072,
    num_experts=4,
    vocab_size=32000,
    token_frequencies=freqs,  # Zipf-balanced routing
)
mlp = TokenRoutedMLP(config)
```

| Aspect | Top-K MoE | Token-Routed |
|--------|-----------|--------------|
| Routing | Learned softmax | **Deterministic (Zipf-balanced)** |
| Load Balance | Auxiliary loss | **Guaranteed by design** |
| Expert Collapse | Risk | **Impossible** |
| Compute | 1/k of dense | **1/k of dense** |

### 2. Mu-Guidance (Cross-layer Information Flow)

```python
# mu flows between layers: layer N's mu guides layer N+1's attention
# mu_init: learnable parameter so layer 0 also gets guidance
q = q + mu_to_q(mu_prev)  # mu biases Q projection
k = k + mu_to_k(mu_prev)  # mu biases K projection
v = v + mu_to_v(mu_prev)  # mu biases V projection
```

### 3. Zipf-balanced Routing

```python
# Problem: token_id % num_experts concentrates frequent tokens
# Solution: sort by frequency, distribute round-robin
sorted_by_freq = vocab.argsort(by=frequency, descending=True)
expert_assignment[sorted_by_freq[i]] = i % num_experts
# Result: each expert gets equal mix of frequent/rare tokens
```

## Cluster Parallelism

```python
from complexity.parallel.cluster import ClusterConfig, ClusterModel

# Auto-configures TP × PP × DP based on model size and GPU count
config = ClusterConfig(tp_size=8, pp_size=1, dp_size=2)
model = ClusterModel(model, config)
```

| GPUs | Config | Effective Batch | Use Case |
|------|--------|-----------------|----------|
| 1 | DP=1 | 64 | Dev/test |
| 4 | DP=4 | 256 | Ablation |
| 8 | DP=8 | 512 | 400M training |
| 16 | DP=16 | 1,024 | 1B training |
| 64 | DP=64 | 4,096 | 7B training |

## Features

| Module | Description |
|--------|-------------|
| **Attention** | GQA/MHA/MQA with Mu-Guided KQV, QK Norm, RoPE |
| **MLP** | Token-Routed with Zipf-balanced routing, Fused gate+up |
| **Mu-Guidance** | Cross-layer contextual mu, learnable mu_init |
| **Optimizers** | AdamW, Muon (Newton-Schulz), muP scaling |
| **Schedulers** | Cosine, WSD (LLaMA 3), Linear, Constant |
| **Parallel** | FSDP v2, Tensor Parallel, Pipeline Parallel, 3D Cluster |
| **CUDA/Triton** | Flash Attention, CGGR expert dispatch |
| **Multimodal** | OmniModel (text + image + audio + video) |

## Links

- [HuggingFace](https://huggingface.co/Complexity-ML)
- [PyPI](https://pypi.org/project/complexity-framework/)
- [GitHub](https://github.com/Complexity-ML/complexity-framework)

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
