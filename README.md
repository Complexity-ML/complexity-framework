# Complexity Framework v0.3.0

**Framework Python modulaire pour construire des LLMs avec Mu-Guided Architecture et stabilité INL.**

```bash
pip install complexity-framework
```

## What's New in v0.3.0

- **Mu-Guided Attention (KQV)**: μ biases K, Q, AND V projections
- **Mu-Guided Expert Routing**: μ influences MLP expert selection
- **Contextual Mu**: mu_proj adapts μ based on hidden state
- **Fused Mu-KQV**: concat+cuBLAS for 2x speedup
- **Fused gate+up projection**: 1.3x MLP speedup
- **KQV Order**: Industry standard (Llama, Qwen, GPT)

## Quick Start

```python
from complexity.core.attention import GroupedQueryAttention
from complexity.core.mlp import TokenRoutedMLP
from complexity.core.dynamics import INLDynamics

# Mu-Guided GQA (v0.3.0)
attn = GroupedQueryAttention(
    hidden_size=2048,
    num_heads=16,
    num_kv_heads=8,
)
# Forward with mu guidance
attn_out, _, mu_attn = attn(hidden_states, mu_prev=mu_prev)

# Token-Routed MLP with Mu Routing (v0.3.0)
mlp = TokenRoutedMLP(
    hidden_size=2048,
    intermediate_size=5632,
    num_experts=4,
)
mlp_out = mlp(hidden_states, input_ids=input_ids, mu_prev=mu_prev)

# INL Dynamics with Contextual Mu (v0.3.0)
dynamics = INLDynamics(hidden_size=2048, beta_max=2.0)
h_next, v_next, mu_contextual = dynamics(h, v, return_mu=True)
```

## Architecture (v0.3.0)

```
Input
  │
  ▼
[RMSNorm] ─► [Mu-Guided GQA (KQV)] ─► [INL Dynamics] ─► [RMSNorm] ─► [Token-Routed MLP]
  │              ▲                         │                              ▲
  │              │                         │                              │
  │         mu_prev                   mu_contextual ──────────────────────┘
  │                                        │
  +─────────────────── Residual ───────────┼──────────────────────────────+
  │                                        │                              │
  ▼                                        ▼                              │
Output ◄───────────────────────────── mu_next (to next layer) ◄──────────┘
```

## Features

| Module | Description | Version |
|--------|-------------|---------|
| **Attention** | GQA/MHA/MQA with Mu-Guided KQV, QK Norm, RoPE | v0.3.0 |
| **MLP** | Token-Routed with Mu Routing, Fused gate+up | v0.3.0 |
| **INL Dynamics** | Contextual Mu, velocity tracking, beta clamping | v0.3.0 |
| **CUDA/Triton** | Flash Attention, Fused Mu-KQV (concat+cuBLAS) | v0.3.0 |
| **Linear (O(N))** | Mamba, RWKV, RetNet | v0.2.x |
| **Multimodal** | Vision, Audio, Fusion | v0.2.x |

## Mu-Guided Architecture

The key innovation: **μ (mu)** from previous layers guides ALL components:

```python
# Attention: Fused Mu-KQV (2x faster)
x_mu = torch.cat([x, mu_prev], dim=-1)
k = F.linear(x_mu, torch.cat([W_k, W_mu_k], dim=1))
q = F.linear(x_mu, torch.cat([W_q, W_mu_q], dim=1))
v = F.linear(x_mu, torch.cat([W_v, W_mu_v], dim=1))

# MLP: Mu-Guided Expert Routing
router_logits = base_router(x) + mu_router(mu_prev)

# Dynamics: Contextual Mu
mu_contextual = mu + mu_proj(h)
```

## INL Dynamics

Velocity tracking to prevent explosion after 400k+ steps:

```python
# CRITICAL: beta in [0, 2], NOT [0, inf)!
dynamics = INLDynamics(
    hidden_size=768,
    beta_max=2.0,       # Clamp beta for stability
    velocity_max=10.0,  # Limit velocity
)

# v0.3.0: Get contextual mu for next layer
h_next, v_next, mu_contextual = dynamics(h, v, return_mu=True)
```

## Token-Routed MLP + Mu Override

```python
# Deterministic base + learned override
mlp = TokenRoutedMLP(
    hidden_size=2048,
    intermediate_size=5632,
    num_experts=4,
)

# Base: token_id % num_experts (perfectly balanced)
# Override: mu_router shifts expert selection based on context
out = mlp(x, input_ids=input_ids, mu_prev=mu_prev)
```

| Aspect | Top-K MoE | Token-Routed + Mu |
|--------|-----------|-------------------|
| Base Routing | Learned | **Deterministic** |
| Context-Aware | Router | **Mu (lightweight)** |
| Expert Collapse | Risk | **None** |
| Auxiliary Loss | Required | **Not needed** |

## Links

- [HuggingFace Models](https://huggingface.co/Pacific-Prime)
- [PyPI](https://pypi.org/project/complexity-framework/)
- [GitHub](https://github.com/Complexity-ML/complexity-framework)
- [complexity-deep](https://github.com/Complexity-ML/complexity-deep)

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
