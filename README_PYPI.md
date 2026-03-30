# Complexity Framework

**Modular Python framework for building LLMs with INL Dynamics stability.**

```bash
pip install complexity-framework
```

## Quick Start

```python
from complexity.api import (
    # Building blocks
    Attention, MLP, RMSNorm, RoPE, INLDynamics,
    # Optimizations
    CUDA, Efficient,
    # Linear architectures O(N)
    Architecture, Mamba, RWKV,
)

# Flash Attention
attn = CUDA.flash(hidden_size=4096, num_heads=32)

# INL Dynamics (training stability)
dynamics = INLDynamics(hidden_size=768, beta_max=2.0)
h, velocity = dynamics(hidden_states, velocity)

# Small budget model
model = Efficient.tiny_llm(vocab_size=32000)  # ~125M params
```

## Features

| Module | Description |
|--------|-------------|
| **Core (O(N²))** | Attention (GQA/MHA/MQA), MLP (SwiGLU/GeGLU/MoE), Position (RoPE/YaRN/ALiBi) |
| **INL Dynamics** | Velocity tracking for training stability |
| **CUDA/Triton** | Flash Attention, Sliding Window, Sparse, Linear |
| **Efficient** | Quantization, Mixed Precision, Small Models |
| **Linear (O(N))** | Mamba, RWKV, RetNet |
| **Multimodal** | Vision, Audio, Fusion |

## INL Dynamics

Velocity tracking to prevent explosion after 400k+ steps:

```python
# CRITICAL: beta in [0, 2], NOT [0, inf)!
dynamics = INLDynamics(
    hidden_size=768,
    beta_max=2.0,       # Clamp beta for stability
    velocity_max=10.0,  # Limit velocity
)
```

## Mixture of Experts (MoE)

```python
from complexity.api import MLP

# Token-Routed MoE (our innovation - deterministic, no aux_loss)
moe = MLP.moe(hidden_size=4096, num_experts=4)
output = moe(hidden_states, token_ids)  # No aux_loss!

# Sparse MoE (standard - learned routing)
moe = MLP.sparse_moe(hidden_size=4096, num_experts=8, top_k=2)
output, aux_loss = moe(hidden_states)
```

## Small Budget Training

```python
from complexity.api import Efficient

# Pre-configured models
model = Efficient.nano_llm(vocab_size=32000)   # ~10M params
model = Efficient.tiny_llm(vocab_size=32000)   # ~125M params
model = Efficient.small_llm(vocab_size=32000)  # ~350M params

# Memory optimizations
Efficient.enable_checkpointing(model)
```

## Linear Architectures (O(N))

```python
from complexity.api import Architecture

model = Architecture.mamba(hidden_size=768, num_layers=12)
model = Architecture.rwkv(hidden_size=768, num_layers=12)
model = Architecture.retnet(hidden_size=768, num_layers=12)
```

## Documentation

Full documentation with training curves and examples:
**[GitHub](https://github.com/Complexity-ML/complexity-framework)**

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
