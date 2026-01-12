"""
Multimodal API - Vision et Audio faciles.
=========================================

Usage:
    from complexity.api import Vision, Audio, Fusion

    # Vision simple
    encoder = Vision.encoder(image_size=224, hidden_size=768)
    features = encoder(images)

    # CLIP style
    encoder = Vision.clip(hidden_size=768)

    # Audio
    encoder = Audio.encoder(n_mels=80, hidden_size=768)
    features = encoder(mel_spectrograms)

    # Whisper style
    encoder = Audio.whisper(hidden_size=768)

    # Fusion
    fusion = Fusion.cross_attention(hidden_size=768)
    combined = fusion(text_features, image_features)
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Union, List
import torch
import torch.nn as nn

# Import depuis multimodal module
from complexity.multimodal import (
    # Vision
    VisionEncoder,
    VisionConfig,
    PatchEmbedding,
    VisionTransformer,
    CLIPVisionEncoder,
    SigLIPEncoder,
    # Audio
    AudioEncoder,
    AudioConfig,
    MelSpectrogramEncoder,
    WhisperEncoder,
    AudioConvStack,
    # Fusion
    MultimodalFusion,
    FusionConfig,
    CrossAttentionFusion,
    GatedFusion,
    ConcatProjection,
    PerceiverResampler,
)


class Vision:
    """
    Factory pour créer des encodeurs vision.

    Usage:
        # Encoder basique
        encoder = Vision.encoder(image_size=224, hidden_size=768)

        # CLIP
        encoder = Vision.clip(hidden_size=768)

        # SigLIP
        encoder = Vision.siglip(hidden_size=768)

        # Avec config
        encoder = Vision.create("vit", image_size=384, patch_size=14)
    """

    TYPES = {
        "vit": VisionEncoder,
        "clip": CLIPVisionEncoder,
        "siglip": SigLIPEncoder,
        "transformer": VisionTransformer,
    }

    @classmethod
    def create(cls, vision_type: str = "vit", **kwargs) -> nn.Module:
        """
        Crée un encodeur vision.

        Args:
            vision_type: "vit", "clip", "siglip"
            **kwargs: image_size, patch_size, hidden_size, num_layers, ...
        """
        if vision_type not in cls.TYPES:
            raise ValueError(f"Unknown vision type: {vision_type}. Use: {list(cls.TYPES.keys())}")

        vision_cls = cls.TYPES[vision_type]

        # Build config si nécessaire
        if vision_type in ["vit", "transformer"]:
            config = VisionConfig(**kwargs)
            return vision_cls(config)
        else:
            return vision_cls(**kwargs)

    @classmethod
    def encoder(
        cls,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        **kwargs
    ) -> nn.Module:
        """
        Vision Transformer encoder standard.

        Args:
            image_size: Taille image (224, 384, etc.)
            patch_size: Taille patch (16, 14, etc.)
            hidden_size: Dimension hidden
            num_layers: Nombre de layers
            num_heads: Nombre de heads attention
        """
        return cls.create(
            "vit",
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            **kwargs
        )

    @classmethod
    def clip(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """CLIP vision encoder."""
        return cls.create("clip", hidden_size=hidden_size, **kwargs)

    @classmethod
    def siglip(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """SigLIP vision encoder."""
        return cls.create("siglip", hidden_size=hidden_size, **kwargs)

    @classmethod
    def patches(cls, image_size: int = 224, patch_size: int = 16, hidden_size: int = 768) -> nn.Module:
        """Patch embedding layer seul."""
        return PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )


class Audio:
    """
    Factory pour créer des encodeurs audio.

    Usage:
        # Encoder basique
        encoder = Audio.encoder(n_mels=80, hidden_size=768)

        # Whisper style
        encoder = Audio.whisper(hidden_size=768)

        # Mel spectrogram
        encoder = Audio.mel(n_mels=80, hidden_size=768)
    """

    TYPES = {
        "standard": AudioEncoder,
        "whisper": WhisperEncoder,
        "mel": MelSpectrogramEncoder,
    }

    @classmethod
    def create(cls, audio_type: str = "standard", **kwargs) -> nn.Module:
        """
        Crée un encodeur audio.

        Args:
            audio_type: "standard", "whisper", "mel"
            **kwargs: n_mels, hidden_size, num_layers, ...
        """
        if audio_type not in cls.TYPES:
            raise ValueError(f"Unknown audio type: {audio_type}. Use: {list(cls.TYPES.keys())}")

        audio_cls = cls.TYPES[audio_type]

        # Build config si nécessaire
        if audio_type in ["standard", "whisper"]:
            config = AudioConfig(**kwargs)
            return audio_cls(config)
        else:
            return audio_cls(**kwargs)

    @classmethod
    def encoder(
        cls,
        n_mels: int = 80,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        **kwargs
    ) -> nn.Module:
        """
        Audio encoder standard.

        Args:
            n_mels: Nombre de mel bins
            hidden_size: Dimension hidden
            num_layers: Nombre de layers
            num_heads: Nombre de heads
        """
        return cls.create(
            "standard",
            n_mels=n_mels,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            **kwargs
        )

    @classmethod
    def whisper(cls, hidden_size: int = 768, n_mels: int = 80, **kwargs) -> nn.Module:
        """Whisper-style audio encoder."""
        return cls.create("whisper", hidden_size=hidden_size, n_mels=n_mels, **kwargs)

    @classmethod
    def mel(cls, n_mels: int = 80, hidden_size: int = 768, **kwargs) -> nn.Module:
        """Mel spectrogram encoder."""
        return cls.create("mel", n_mels=n_mels, hidden_size=hidden_size, **kwargs)

    @classmethod
    def conv_stack(cls, n_mels: int = 80, hidden_size: int = 768) -> nn.Module:
        """Conv stack pour audio (comme Whisper)."""
        return AudioConvStack(n_mels=n_mels, hidden_size=hidden_size)


class Fusion:
    """
    Factory pour créer des modules de fusion multimodal.

    Usage:
        # Cross-attention
        fusion = Fusion.cross_attention(hidden_size=768)
        combined = fusion(text_features, image_features)

        # Gated fusion
        fusion = Fusion.gated(hidden_size=768)

        # Concat + projection
        fusion = Fusion.concat(hidden_sizes=[768, 768], output_size=768)

        # Perceiver resampler (comme Flamingo)
        fusion = Fusion.perceiver(hidden_size=768, num_latents=64)
    """

    TYPES = {
        "cross_attention": CrossAttentionFusion,
        "gated": GatedFusion,
        "concat": ConcatProjection,
        "perceiver": PerceiverResampler,
        "multimodal": MultimodalFusion,
    }

    @classmethod
    def create(cls, fusion_type: str = "cross_attention", **kwargs) -> nn.Module:
        """
        Crée un module de fusion.

        Args:
            fusion_type: "cross_attention", "gated", "concat", "perceiver", "multimodal"
            **kwargs: hidden_size, num_heads, ...
        """
        if fusion_type not in cls.TYPES:
            raise ValueError(f"Unknown fusion type: {fusion_type}. Use: {list(cls.TYPES.keys())}")

        fusion_cls = cls.TYPES[fusion_type]

        # Build config si nécessaire
        if fusion_type in ["cross_attention", "multimodal"]:
            config = FusionConfig(**kwargs)
            return fusion_cls(config)
        else:
            return fusion_cls(**kwargs)

    @classmethod
    def cross_attention(
        cls,
        hidden_size: int = 768,
        num_heads: int = 12,
        num_layers: int = 2,
        **kwargs
    ) -> nn.Module:
        """
        Cross-attention fusion (texte attend sur vision).

        Args:
            hidden_size: Dimension hidden
            num_heads: Nombre de heads
            num_layers: Nombre de layers cross-attention
        """
        return cls.create(
            "cross_attention",
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            num_hidden_layers=num_layers,
            **kwargs
        )

    @classmethod
    def gated(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """Gated fusion (apprentissage du ratio)."""
        return cls.create("gated", hidden_size=hidden_size, **kwargs)

    @classmethod
    def concat(cls, hidden_sizes: List[int], output_size: int, **kwargs) -> nn.Module:
        """Concat + projection."""
        return cls.create("concat", hidden_sizes=hidden_sizes, output_size=output_size, **kwargs)

    @classmethod
    def perceiver(
        cls,
        hidden_size: int = 768,
        num_latents: int = 64,
        num_layers: int = 2,
        **kwargs
    ) -> nn.Module:
        """
        Perceiver resampler (comme Flamingo).

        Réduit les tokens vision à un nombre fixe de latents.
        """
        return cls.create(
            "perceiver",
            hidden_size=hidden_size,
            num_latents=num_latents,
            num_layers=num_layers,
            **kwargs
        )

    @classmethod
    def multimodal(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """Fusion multimodale générique."""
        return cls.create("multimodal", hidden_size=hidden_size, **kwargs)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Factories
    "Vision",
    "Audio",
    "Fusion",
    # Direct classes - Vision
    "VisionEncoder",
    "VisionConfig",
    "PatchEmbedding",
    "VisionTransformer",
    "CLIPVisionEncoder",
    "SigLIPEncoder",
    # Direct classes - Audio
    "AudioEncoder",
    "AudioConfig",
    "MelSpectrogramEncoder",
    "WhisperEncoder",
    "AudioConvStack",
    # Direct classes - Fusion
    "MultimodalFusion",
    "FusionConfig",
    "CrossAttentionFusion",
    "GatedFusion",
    "ConcatProjection",
    "PerceiverResampler",
]
