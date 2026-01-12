---
title: Complexity Framework
emoji: 🐢
colorFrom: purple
colorTo: blue
sdk: static
pinned: true
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/643222d9f76c34519e96a299/8j1GHX24MV3-sv-4zl7ZB.png
---

# Complexity Framework

**Modular Python framework for building LLMs with INL Dynamics stability**

## What is Complexity Framework?

Complexity Framework is a complete toolkit for building transformer architectures with built-in training stability. It provides:

- **INL Dynamics** - Second-order dynamical system for training stability
- **Token-Routed MLP (MoE)** - Efficient sparse activation
- **CUDA/Triton Optimizations** - Flash Attention, Sliding Window, Sparse, Linear
- **O(N) Architectures** - Mamba, RWKV, RetNet
- **Small Budget Training** - Quantization, Mixed Precision, Gradient Checkpointing

## Key Innovation: INL Dynamics

Velocity tracking to prevent training explosion after 400k+ steps:

```python
from complexity.api import INLDynamics

# CRITICAL: beta in [0, 2], NOT [0, inf)!
dynamics = INLDynamics(
    hidden_size=768,
    beta_max=2.0,       # Clamp beta for stability
    velocity_max=10.0,  # Limit velocity
)

h_next, v_next = dynamics(hidden_states, velocity)
```

**The bug we fixed**: `softplus` without clamp goes to infinity, causing NaN after 400k steps. Clamping beta to [0, 2] keeps training stable.

## Loss Spike Recovery

![Loss Spike Recovery](https://raw.githubusercontent.com/Complexity-ML/complexity-framework/main/docs/loss-spike-recovery.png)

*INL Dynamics recovers from loss spikes thanks to velocity damping.*

## Stability at 400k+ Steps

![Training at 400k steps](https://raw.githubusercontent.com/Complexity-ML/complexity-framework/main/docs/training-400k-stable.png)

*After beta clamping fix: training remains stable past 400k steps where it previously exploded.*

## Quick Start

```bash
pip install complexity-framework
```

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

# INL Dynamics (training stability)
dynamics = INLDynamics(hidden_size=768, beta_max=2.0)
h, velocity = dynamics(hidden_states, velocity)

# Small budget model
model = Efficient.tiny_llm(vocab_size=32000)  # ~125M params
```

## Features

| Module | Description |
|--------|-------------|
| **Core** | Attention (GQA/MHA/MQA), MLP (SwiGLU/GeGLU/MoE), Position (RoPE/YaRN/ALiBi) |
| **INL Dynamics** | Velocity tracking for training stability |
| **CUDA/Triton** | Flash Attention, Sliding Window, Sparse, Linear |
| **Efficient** | Quantization, Mixed Precision, Small Models |
| **O(N) Architectures** | Mamba, RWKV, RetNet |
| **Multimodal** | Vision, Audio, Fusion |

## Token-Routed MLP (MoE)

```python
from complexity.api import MLP, TokenRoutedMLP

# Via factory
moe = MLP.moe(hidden_size=4096, num_experts=8, top_k=2)

# Direct
moe = TokenRoutedMLP(
    hidden_size=4096,
    num_experts=8,
    top_k=2,
)

output, aux_loss = moe(hidden_states)
```

## Small Budget Training

```python
from complexity.api import Efficient

# Pre-configured models
model = Efficient.nano_llm(vocab_size=32000)   # ~10M params
model = Efficient.micro_llm(vocab_size=32000)  # ~30M params
model = Efficient.tiny_llm(vocab_size=32000)   # ~125M params
model = Efficient.small_llm(vocab_size=32000)  # ~350M params

# Memory optimizations
Efficient.enable_checkpointing(model)
model, optimizer, scaler = Efficient.mixed_precision(model, optimizer)
```

## O(N) Architectures

For very long sequences:

```python
from complexity.api import Architecture

model = Architecture.mamba(hidden_size=768, num_layers=12)
model = Architecture.rwkv(hidden_size=768, num_layers=12)
model = Architecture.retnet(hidden_size=768, num_layers=12)
```

## Documentation

- [Getting Started](https://github.com/Complexity-ML/complexity-framework/blob/main/docs/getting-started.md)
- [API Reference](https://github.com/Complexity-ML/complexity-framework/blob/main/docs/api.md)
- [INL Dynamics](https://github.com/Complexity-ML/complexity-framework/blob/main/docs/dynamics.md)
- [MoE / Token-Routed MLP](https://github.com/Complexity-ML/complexity-framework/blob/main/docs/moe.md)
- [CUDA Optimizations](https://github.com/Complexity-ML/complexity-framework/blob/main/docs/cuda.md)
- [Efficient Training](https://github.com/Complexity-ML/complexity-framework/blob/main/docs/efficient.md)
- [O(N) Architectures](https://github.com/Complexity-ML/complexity-framework/blob/main/docs/architectures.md)

## Links

- [GitHub](https://github.com/Complexity-ML/complexity-framework)
- [PyPI](https://pypi.org/project/complexity-framework/) (coming soon)

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)

## Citation

```bibtex
@software{complexity_framework_2024,
  title={Complexity Framework: Modular LLM Building Blocks with INL Dynamics},
  author={Complexity-ML},
  year={2024},
  url={https://github.com/Complexity-ML/complexity-framework}
}
```

---

**Build stable LLMs. Train with confidence.**
