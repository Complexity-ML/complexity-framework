# O(N) Architectures

**Alternatives aux Transformers pour séquences longues.**

## Comparaison

| Architecture | Complexité | Mémoire | Parallélisable | Inference |
|--------------|------------|---------|----------------|-----------|
| Transformer | O(N²) | O(N²) | Oui | O(N) par token |
| **Mamba** | O(N) | O(N) | Oui | O(1) par token |
| **RWKV** | O(N) | O(N) | Oui | O(1) par token |
| **RetNet** | O(N) | O(N) | Oui | O(1) par token |

## Mamba (State Space Model)

État de l'art pour SSM. Excellente qualité proche des Transformers.

```python
from complexity.api import Architecture

# Modèle complet
model = Architecture.mamba(
    hidden_size=768,
    num_layers=12,
    vocab_size=32000,
)

# Block individuel
block = Architecture.mamba_block(hidden_size=768)
```

### Caractéristiques
- Selective state spaces
- Hardware-efficient (scan linéaire)
- Bon pour audio, génomique, texte long

## RWKV (Linear Attention RNN)

Combine avantages RNN (inference rapide) et Transformer (training parallèle).

```python
model = Architecture.rwkv(
    hidden_size=768,
    num_layers=12,
    vocab_size=32000,
)

block = Architecture.rwkv_block(hidden_size=768)
```

### Caractéristiques
- Time-mixing + Channel-mixing
- Pas de KV cache nécessaire
- Inference très rapide

## RetNet (Retentive Networks)

Training parallèle, inference récurrente.

```python
model = Architecture.retnet(
    hidden_size=768,
    num_layers=12,
    vocab_size=32000,
)

block = Architecture.retnet_block(hidden_size=768)
```

### Caractéristiques
- 3 modes: parallèle, récurrent, chunk
- Retention mechanism (decay exponentiel)
- Multi-scale retention

## Mixture of Depths (MoD)

Sélectionne dynamiquement quels tokens passent par le bloc.

```python
block = Architecture.mod_block(
    hidden_size=768,
    capacity_factor=0.5,  # 50% des tokens
)
```

### Caractéristiques
- Réduit le compute pour tokens "faciles"
- Capacity factor contrôlable
- Compatible avec autres architectures

## Quand utiliser quoi

| Cas d'usage | Recommandation |
|-------------|----------------|
| Texte standard (<4k) | Transformer + Flash Attention |
| Texte long (4k-32k) | Mamba ou Sliding Window |
| Texte très long (>32k) | Mamba ou RWKV |
| Inference temps réel | RWKV ou RetNet |
| Audio | Mamba |
| Code | Transformer ou Mamba |

## Hybrid architectures

Combiner Transformer et SSM:

```python
import torch.nn as nn
from complexity.api import Architecture, CUDA

class HybridLayer(nn.Module):
    def __init__(self, hidden_size, use_mamba=False):
        super().__init__()
        if use_mamba:
            self.core = Architecture.mamba_block(hidden_size)
        else:
            self.core = CUDA.flash(hidden_size, num_heads=12)

# Alterner: Transformer aux layers critiques, Mamba ailleurs
layers = []
for i in range(24):
    use_mamba = i % 4 != 0  # Transformer toutes les 4 layers
    layers.append(HybridLayer(768, use_mamba=use_mamba))
```

## Configs

```python
from complexity.api import MambaConfig, RWKVConfig, RetNetConfig

# Mamba
config = MambaConfig(
    hidden_size=768,
    num_hidden_layers=12,
    state_size=16,
    expand=2,
)

# RWKV
config = RWKVConfig(
    hidden_size=768,
    num_hidden_layers=12,
    context_length=4096,
)

# RetNet
config = RetNetConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_heads=8,
)
```
