# API Reference

## Import principal

```python
from complexity.api import (
    # === Base ===
    Tokenizer, Model, Dataset, Trainer,

    # === Building Blocks ===
    Attention, MLP, Position, Norm, Block,
    GQA, MHA, MQA,
    SwiGLU, GeGLU, TokenRoutedMLP,
    RoPE, YaRN, ALiBi,
    RMSNorm, LayerNorm,

    # === INL Dynamics ===
    INLDynamics, INLDynamicsLite, DynamicsConfig,

    # === CUDA/Triton ===
    CUDA, Triton,
    FlashAttention, SlidingWindowAttention, SparseAttention, LinearAttention,

    # === Efficient ===
    Efficient, Quantize, SmallModels,

    # === Architectures O(N) ===
    Architecture, Mamba, RWKV, RetNet,

    # === Helpers ===
    Helpers, Init, Mask, KVCache, Sampling, Debug,

    # === Multimodal ===
    Vision, Audio, Fusion,

    # === Registry ===
    register, Registry,
)
```

---

## Attention

### Factory

```python
attn = Attention.gqa(hidden_size=4096, num_heads=32, kv_heads=8)
attn = Attention.mha(hidden_size=4096, num_heads=32)
attn = Attention.mqa(hidden_size=4096, num_heads=32)
```

### Classes directes

```python
from complexity.api import GQA, MHA, MQA, AttentionConfig

config = AttentionConfig(
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=8,
)
attn = GQA(config)
```

---

## MLP

### Factory

```python
mlp = MLP.swiglu(hidden_size=4096, intermediate_size=11008)
mlp = MLP.geglu(hidden_size=4096, intermediate_size=11008)
mlp = MLP.standard(hidden_size=4096, intermediate_size=16384)
mlp = MLP.moe(hidden_size=4096, num_experts=8, top_k=2)
```

### Classes directes

```python
from complexity.api import SwiGLU, GeGLU, TokenRoutedMLP

mlp = SwiGLU(hidden_size=4096, intermediate_size=11008)
```

---

## Position Embeddings

### Factory

```python
pos = Position.rope(dim=128, max_seq_len=4096)
pos = Position.yarn(dim=128, max_seq_len=4096, scale=4)
pos = Position.alibi(num_heads=32)
```

### Classes directes

```python
from complexity.api import RoPE, YaRN, ALiBi

rope = RoPE(dim=128, max_seq_len=4096, base=10000)
```

---

## Normalization

### Factory

```python
norm = Norm.rms(hidden_size=4096)
norm = Norm.layer(hidden_size=4096)
```

### Classes directes

```python
from complexity.api import RMSNorm, LayerNorm

norm = RMSNorm(hidden_size=4096, eps=1e-6)
```

---

## INL Dynamics

```python
from complexity.api import INLDynamics, INLDynamicsLite

# Version complète (paramètres adaptatifs)
dynamics = INLDynamics(
    hidden_size=768,
    init_alpha=0.9,
    init_beta=0.1,
    init_gate=0.5,
    dt=0.1,
    beta_max=2.0,        # CRITICAL!
    velocity_max=10.0,
)

# Version lite (paramètres fixes)
dynamics = INLDynamicsLite(
    hidden_size=768,
    alpha=0.9,
    beta=0.1,
    gate=0.5,
)

# Forward
h_next, v_next = dynamics(hidden_states, velocity_states)
```

---

## CUDA / Triton

```python
from complexity.api import CUDA

attn = CUDA.flash(hidden_size=4096, num_heads=32)
attn = CUDA.sliding_window(hidden_size=4096, num_heads=32, window_size=4096)
attn = CUDA.sparse(hidden_size=4096, num_heads=32, block_size=64)
attn = CUDA.linear(hidden_size=4096, num_heads=32)
attn = CUDA.multiscale(hidden_size=4096, num_heads=32)
```

---

## Efficient

```python
from complexity.api import Efficient

# Modèles
model = Efficient.nano_llm(vocab_size=32000)   # ~10M
model = Efficient.micro_llm(vocab_size=32000)  # ~30M
model = Efficient.tiny_llm(vocab_size=32000)   # ~125M
model = Efficient.small_llm(vocab_size=32000)  # ~350M

# Optimisations
Efficient.enable_checkpointing(model)
model, opt, scaler = Efficient.mixed_precision(model, optimizer)
model = Efficient.quantize_model(model, bits=4)

# Estimation
mem = Efficient.estimate_memory(model, batch_size=4, seq_len=2048)
config = Efficient.recommend_config(vram_gb=12, training=True)
```

---

## Architectures O(N)

```python
from complexity.api import Architecture

# Modèles complets
model = Architecture.mamba(hidden_size=768, num_layers=12)
model = Architecture.rwkv(hidden_size=768, num_layers=12)
model = Architecture.retnet(hidden_size=768, num_layers=12)

# Blocks individuels
block = Architecture.mamba_block(hidden_size=768)
block = Architecture.rwkv_block(hidden_size=768)
block = Architecture.mod_block(hidden_size=768, capacity_factor=0.5)
```

---

## Helpers

```python
from complexity.api import Helpers, Init, Mask, KVCache, Sampling, Debug

# Initialisation
Init.xavier(model)
Init.kaiming(model, mode="fan_out")
Helpers.init_weights(model, init_type="normal", std=0.02)

# Masking
mask = Mask.causal(seq_len=2048, device="cuda")
mask = Mask.sliding_window(seq_len=2048, window_size=512)
mask = Mask.padding(lengths=[10, 15, 8], max_len=20)

# KV Cache
cache = KVCache.create(num_layers=32, num_heads=32, head_dim=128)
k, v = cache.update(layer_idx, new_k, new_v)

# Sampling
token = Sampling.sample(logits, temperature=0.7, top_k=50, top_p=0.9)
token = Sampling.greedy(logits)

# Debug
print(Debug.count_params(model))  # "7.2B"
Debug.print_summary(model)
mem = Debug.memory_usage()
```

---

## Registry (Custom Components)

```python
from complexity.api import register, AttentionBase

@register("attention", "my_attention")
class MyAttention(AttentionBase):
    def forward(self, x, **kwargs):
        # Custom implementation
        return x, None

# Utilisation
attn = Attention.create("my_attention", hidden_size=768)
```

---

## Multimodal

```python
from complexity.api import Vision, Audio, Fusion

# Vision
encoder = Vision.clip(image_size=224, patch_size=16)
encoder = Vision.siglip(image_size=384)

# Audio
encoder = Audio.whisper(sample_rate=16000)
encoder = Audio.mel_spectrogram(n_mels=80)

# Fusion
fusion = Fusion.cross_attention(hidden_size=768)
fusion = Fusion.gated(hidden_size=768)
fusion = Fusion.perceiver(hidden_size=768, num_latents=64)
```
