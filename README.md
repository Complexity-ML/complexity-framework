# Complexity Framework

**Framework Python modulaire pour construire des LLMs avec stabilité INL.**

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
    # Architectures O(N)
    Architecture, Mamba, RWKV,
)

# Flash Attention
attn = CUDA.flash(hidden_size=4096, num_heads=32)

# INL Dynamics (stabilité training)
dynamics = INLDynamics(hidden_size=768, beta_max=2.0)
h, velocity = dynamics(hidden_states, velocity)

# Small budget model
model = Efficient.tiny_llm(vocab_size=32000)  # ~125M params
```

## Features

| Module | Description |
|--------|-------------|
| **Core** | Attention (GQA/MHA/MQA), MLP (SwiGLU/GeGLU), Position (RoPE/YaRN/ALiBi) |
| **INL Dynamics** | Velocity tracking pour stabilité training - [docs](docs/dynamics.md) |
| **CUDA/Triton** | Flash Attention, Sliding Window, Sparse, Linear - [docs](docs/cuda.md) |
| **Efficient** | Quantization, Mixed Precision, Small Models - [docs](docs/efficient.md) |
| **O(N) Architectures** | Mamba, RWKV, RetNet - [docs](docs/architectures.md) |
| **Multimodal** | Vision, Audio, Fusion - [docs](docs/multimodal.md) |

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

### Training Results

![Training Curves](docs/training-curves.png)

*Stable training with INL Dynamics: loss converges smoothly, perplexity drops rapidly.*

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api.md)
- [Building Custom Models](docs/custom-models.md)
- [INL Dynamics](docs/dynamics.md)
- [CUDA Optimizations](docs/cuda.md)
- [Training Guide](docs/training.md)

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
