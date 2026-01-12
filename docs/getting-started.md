# Getting Started

## Installation

```bash
pip install complexity-framework

# Avec support CUDA/Triton
pip install complexity-framework[cuda]

# Développement
git clone https://github.com/Web3-League/complexity-framework.git
cd complexity-framework
pip install -e .
```

## Premier modèle

### Usage simple

```python
from complexity.api import Model, Tokenizer, Generate

# Charger un modèle pré-entraîné
tokenizer = Tokenizer.load("llama-7b")
model = Model.load("llama-7b", device="cuda")

# Générer
output = model.generate("Hello, world!", max_tokens=100)
print(output)
```

### Construire un modèle custom

```python
from complexity.api import (
    Attention, MLP, Norm, Position,
    GQA, SwiGLU, RMSNorm, RoPE,
    INLDynamics,
    CUDA,
)
import torch.nn as nn

class MyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Embedding
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Layers
        self.layers = nn.ModuleList([
            MyDecoderLayer(config) for _ in range(config.num_layers)
        ])

        # Output
        self.norm = RMSNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, input_ids):
        h = self.embed(input_ids)

        velocity = None
        for layer in self.layers:
            h, velocity = layer(h, velocity)

        h = self.norm(h)
        return self.lm_head(h)


class MyDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Attention avec Flash Attention
        self.attn = CUDA.flash(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
        )

        # INL Dynamics pour stabilité
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            beta_max=2.0,
        )

        # MLP
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
        )

        self.norm1 = RMSNorm(config.hidden_size)
        self.norm2 = RMSNorm(config.hidden_size)

    def forward(self, h, velocity=None):
        # Attention
        residual = h
        h = self.norm1(h)
        h, _ = self.attn(h)

        # Dynamics
        h, velocity = self.dynamics(h, velocity)
        h = residual + h

        # MLP
        residual = h
        h = self.norm2(h)
        h = self.mlp(h)
        h = residual + h

        return h, velocity
```

## Avec petit budget GPU

```python
from complexity.api import Efficient

# Vérifier le VRAM disponible
vram = Efficient.get_vram()
print(f"VRAM: {vram:.1f} GB")

# Config recommandée
config = Efficient.recommend_config(vram_gb=vram, training=True)
print(config)

# Créer un petit modèle
model = Efficient.tiny_llm(vocab_size=32000)  # ~125M params

# Optimisations
Efficient.enable_checkpointing(model)
model, optimizer, scaler = Efficient.mixed_precision(model, optimizer)
```

## Architectures O(N)

Pour des séquences très longues:

```python
from complexity.api import Architecture

# Mamba (State Space Model)
model = Architecture.mamba(hidden_size=768, num_layers=12)

# RWKV (Linear RNN)
model = Architecture.rwkv(hidden_size=768, num_layers=12)

# RetNet
model = Architecture.retnet(hidden_size=768, num_layers=12)
```

## Prochaines étapes

- [API Reference](api.md) - Documentation complète de l'API
- [INL Dynamics](dynamics.md) - Comprendre le velocity tracking
- [CUDA Optimizations](cuda.md) - Flash Attention et plus
- [Training Guide](training.md) - Entraîner votre modèle
