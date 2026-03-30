# CUDA / Triton Optimizations

**Attention layers optimisées pour GPU.**

## Vue d'ensemble

| Type | Complexité | Mémoire | Usage |
|------|------------|---------|-------|
| Flash Attention | O(N²) | O(N) | Standard, jusqu'à 8k tokens |
| Sliding Window | O(N×W) | O(W) | Mistral-style, longues séquences |
| Sparse | O(N×B) | O(B) | BigBird/Longformer |
| Linear | O(N) | O(1) | Très longues séquences |
| MultiScale | Variable | Variable | Mix local + global |

## Flash Attention

2-4x plus rapide, O(N) mémoire au lieu de O(N²).

```python
from complexity.api import CUDA

attn = CUDA.flash(
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,  # GQA
)

output, cache = attn(hidden_states)
```

**Requis:** PyTorch 2.0+

## Sliding Window Attention

Chaque token n'attend que les W tokens précédents.

```python
attn = CUDA.sliding_window(
    hidden_size=4096,
    num_heads=32,
    window_size=4096,  # Mistral-style
)
```

**Utilisé par:** Mistral, Longformer

## Sparse Attention

Combine attention locale + globale + aléatoire.

```python
attn = CUDA.sparse(
    hidden_size=4096,
    num_heads=32,
    block_size=64,
    num_global_tokens=1,  # [CLS] token
)
```

**Utilisé par:** BigBird, Longformer

## Linear Attention

O(N) complexité totale - pour séquences très longues.

```python
attn = CUDA.linear(
    hidden_size=4096,
    num_heads=32,
    feature_map="elu",  # "elu", "relu", "softmax"
)
```

**Référence:** "Transformers are RNNs" (Katharopoulos et al., 2020)

## Multi-Scale Attention

Différentes têtes utilisent différentes échelles.

```python
attn = CUDA.multiscale(
    hidden_size=4096,
    num_heads=32,
    local_heads=16,  # 16 têtes locales, 16 globales
    window_sizes=(256, 512, 1024),
)
```

## Factory générique

```python
attn = CUDA.create(
    attn_type="flash",  # "flash", "sliding_window", "sparse", "linear", "multiscale"
    hidden_size=4096,
    num_heads=32,
    **kwargs
)
```

## Comparaison mémoire

Pour seq_len=8192, hidden_size=4096, batch_size=1:

| Type | Mémoire activations |
|------|---------------------|
| Standard | ~2 GB |
| Flash | ~500 MB |
| Sliding (W=512) | ~64 MB |
| Linear | ~32 MB |

## Tips

1. **Flash Attention** pour la plupart des cas
2. **Sliding Window** si séquences > 8k tokens
3. **Linear Attention** si séquences > 32k tokens
4. **Sparse** pour documents avec structure (début important)

## Avec GQA

```python
# 32 query heads, 8 KV heads = 4x moins de mémoire KV
attn = CUDA.flash(
    hidden_size=4096,
    num_heads=32,
    num_kv_heads=8,
)
```
