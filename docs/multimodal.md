# Multimodal

Vision, Audio, and Fusion components for multimodal models.

## Vision

### CLIP Encoder

```python
from complexity.api import Vision

# CLIP-style vision encoder
encoder = Vision.clip(
    image_size=224,
    patch_size=16,
    hidden_size=768,
    num_layers=12,
)

# Forward
image_features = encoder(images)  # [B, num_patches, hidden]
```

### SigLIP Encoder

```python
# SigLIP (better for retrieval)
encoder = Vision.siglip(
    image_size=384,
    patch_size=14,
)
```

### Custom Vision Transformer

```python
from complexity.api import VisionTransformer, VisionConfig, PatchEmbedding

config = VisionConfig(
    image_size=224,
    patch_size=16,
    hidden_size=768,
    num_attention_heads=12,
    num_layers=12,
)

encoder = VisionTransformer(config)
```

## Audio

### Whisper Encoder

```python
from complexity.api import Audio

# Whisper-style audio encoder
encoder = Audio.whisper(
    sample_rate=16000,
    n_mels=80,
    hidden_size=768,
)

# Forward
audio_features = encoder(mel_spectrogram)
```

### Mel Spectrogram

```python
# Just the mel spectrogram encoder
mel_encoder = Audio.mel_spectrogram(
    sample_rate=16000,
    n_mels=80,
    n_fft=400,
)

features = mel_encoder(waveform)
```

### Custom Audio Stack

```python
from complexity.api import AudioConvStack, AudioConfig

config = AudioConfig(
    sample_rate=16000,
    n_mels=80,
    hidden_size=768,
)

encoder = AudioConvStack(config)
```

## Fusion

### Cross-Attention Fusion

```python
from complexity.api import Fusion

# Cross-attention between modalities
fusion = Fusion.cross_attention(
    hidden_size=768,
    num_heads=12,
)

# text attends to image
fused = fusion(
    query=text_features,
    key_value=image_features,
)
```

### Gated Fusion

```python
# Learned gating between modalities
fusion = Fusion.gated(hidden_size=768)

# Combines with learned weights
fused = fusion(text_features, image_features)
```

### Perceiver Resampler

```python
# Compress variable-length inputs to fixed latents
fusion = Fusion.perceiver(
    hidden_size=768,
    num_latents=64,  # Output size
    num_layers=2,
)

# [B, N, D] -> [B, 64, D]
compressed = fusion(image_features)
```

## Complete Multimodal Model

```python
import torch.nn as nn
from complexity.api import (
    Vision, Audio, Fusion,
    Model, INLDynamics
)

class MultimodalLLM(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Encoders
        self.vision = Vision.clip(image_size=224, patch_size=16)
        self.audio = Audio.whisper(sample_rate=16000)

        # Compress to fixed size
        self.vision_resampler = Fusion.perceiver(
            hidden_size=config.hidden_size,
            num_latents=64,
        )
        self.audio_resampler = Fusion.perceiver(
            hidden_size=config.hidden_size,
            num_latents=32,
        )

        # LLM backbone
        self.llm = Model.from_config(config)

        # Dynamics for stability
        self.dynamics = INLDynamics(
            hidden_size=config.hidden_size,
            beta_max=2.0,
        )

    def forward(self, text_ids, images=None, audio=None):
        # Encode text
        text_embeds = self.llm.embed(text_ids)

        # Encode and resample vision
        if images is not None:
            vis_features = self.vision(images)
            vis_features = self.vision_resampler(vis_features)

        # Encode and resample audio
        if audio is not None:
            aud_features = self.audio(audio)
            aud_features = self.audio_resampler(aud_features)

        # Concatenate all modalities
        inputs = [text_embeds]
        if images is not None:
            inputs.insert(0, vis_features)  # Vision first
        if audio is not None:
            inputs.insert(0, aud_features)  # Audio first

        hidden = torch.cat(inputs, dim=1)

        # Process through LLM
        output = self.llm.forward_hidden(hidden)

        return output
```

## API Reference

### Vision Factory

```python
Vision.clip(image_size, patch_size, **kwargs)
Vision.siglip(image_size, **kwargs)
Vision.vit(image_size, patch_size, **kwargs)
```

### Audio Factory

```python
Audio.whisper(sample_rate, n_mels, **kwargs)
Audio.mel_spectrogram(sample_rate, n_mels, **kwargs)
Audio.wav2vec(sample_rate, **kwargs)
```

### Fusion Factory

```python
Fusion.cross_attention(hidden_size, num_heads, **kwargs)
Fusion.gated(hidden_size, **kwargs)
Fusion.perceiver(hidden_size, num_latents, **kwargs)
Fusion.concat(hidden_size, **kwargs)
```

## See Also

- [API Reference](api.md) - Full API documentation
- [Custom Models](custom-models.md) - Building custom architectures
