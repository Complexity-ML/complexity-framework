"""
Multi-modal module for framework-complexity.

Supports:
- Vision encoding (ViT, CLIP)
- Audio encoding (Whisper-style)
- Multi-modal fusion
- Vision-Language models

Usage:
    from complexity.multimodal import VisionEncoder, AudioEncoder, MultimodalFusion

    # Vision encoder
    vision = VisionEncoder(
        image_size=224,
        patch_size=16,
        hidden_size=768,
    )
    image_features = vision(images)

    # Audio encoder
    audio = AudioEncoder(
        n_mels=80,
        hidden_size=768,
    )
    audio_features = audio(mel_spectrograms)

    # Fusion
    fusion = MultimodalFusion(hidden_size=768)
    combined = fusion(text_features, image_features)
"""

from .vision import (
    VisionEncoder,
    VisionConfig,
    PatchEmbedding,
    VisionTransformer,
    CLIPVisionEncoder,
    SigLIPEncoder,
)

from .audio import (
    AudioEncoder,
    AudioConfig,
    MelSpectrogramEncoder,
    WhisperEncoder,
    AudioConvStack,
)

from .fusion import (
    MultimodalFusion,
    FusionConfig,
    CrossAttentionFusion,
    GatedFusion,
    ConcatProjection,
    PerceiverResampler,
)

__all__ = [
    # Vision
    "VisionEncoder",
    "VisionConfig",
    "PatchEmbedding",
    "VisionTransformer",
    "CLIPVisionEncoder",
    "SigLIPEncoder",
    # Audio
    "AudioEncoder",
    "AudioConfig",
    "MelSpectrogramEncoder",
    "WhisperEncoder",
    "AudioConvStack",
    # Fusion
    "MultimodalFusion",
    "FusionConfig",
    "CrossAttentionFusion",
    "GatedFusion",
    "ConcatProjection",
    "PerceiverResampler",
]
