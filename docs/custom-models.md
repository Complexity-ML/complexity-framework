# Custom Models

Build your own architectures using Complexity building blocks.

## Basic Custom Model

```python
import torch.nn as nn
from complexity.api import (
    GQA, SwiGLU, RoPE, RMSNorm,
    AttentionConfig, register
)

class MyTransformerBlock(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12, kv_heads=4):
        super().__init__()

        # Attention with GQA
        self.attn = GQA(
            AttentionConfig(
                hidden_size=hidden_size,
                num_attention_heads=num_heads,
                num_key_value_heads=kv_heads,
            )
        )

        # MLP with SwiGLU
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=hidden_size * 4,
        )

        # Normalization
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)

        # Position embeddings
        self.rope = RoPE(dim=hidden_size // num_heads, max_seq_len=4096)

    def forward(self, x, position_ids=None):
        # Pre-norm architecture
        h = self.norm1(x)

        # Apply RoPE
        cos, sin = self.rope(h, position_ids)

        # Attention
        attn_out, _ = self.attn(h, cos=cos, sin=sin)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))

        return x
```

## Full Custom Model

```python
from complexity.api import (
    Model, ModelConfig, TransformerBlock,
    INLDynamics, Efficient
)

class MyLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # INL Dynamics for stability
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            beta_max=2.0,  # CRITICAL!
        )

        # Output
        self.norm = RMSNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, velocity=None):
        # Embed
        h = self.embed(input_ids)

        # Initialize velocity if needed
        if velocity is None:
            velocity = torch.zeros_like(h)

        # Process through blocks
        for block in self.blocks:
            h = block(h)

            # Apply dynamics every N layers
            h, velocity = self.dynamics(h, velocity)

        # Output
        h = self.norm(h)
        logits = self.head(h)

        return logits, velocity
```

## Using Factories

```python
from complexity.api import Attention, MLP, Position, Norm

# Quick construction via factories
attn = Attention.gqa(hidden_size=768, num_heads=12, kv_heads=4)
mlp = MLP.swiglu(hidden_size=768, intermediate_size=3072)
rope = Position.rope(dim=64, max_seq_len=4096)
norm = Norm.rms(hidden_size=768)
```

## Registering Custom Components

```python
from complexity.api import register, AttentionBase

@register("attention", "my_custom_attention")
class MyCustomAttention(AttentionBase):
    """Custom attention with special features."""

    def __init__(self, config):
        super().__init__(config)
        # Custom initialization

    def forward(self, x, **kwargs):
        # Custom implementation
        return output, attention_weights

# Use via registry
from complexity.api import Attention
attn = Attention.create("my_custom_attention", hidden_size=768)
```

## Hybrid Architectures

Mix Transformer with O(N) components:

```python
from complexity.api import (
    TransformerBlock, MambaBlock, Architecture
)

class HybridModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            if i % 4 == 0:
                # Every 4th layer is Mamba (O(N))
                self.layers.append(MambaBlock(config.hidden_size))
            else:
                # Regular Transformer
                self.layers.append(TransformerBlock(config))
```

## See Also

- [API Reference](api.md) - Full API documentation
- [Architectures](architectures.md) - O(N) architectures
- [INL Dynamics](dynamics.md) - Stability system
