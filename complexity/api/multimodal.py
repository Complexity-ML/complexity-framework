"""
Multimodal API - Vision, Audio, Video, Omni faciles.
=====================================================

Usage:
    from complexity.api import Vision, Audio, Video, Fusion, Omni

    # Vision (avec token-routed MLP)
    encoder = Vision.encoder(image_size=224, hidden_size=768, num_experts=4)
    features = encoder(images)

    # Audio
    encoder = Audio.whisper(hidden_size=768, num_experts=4)

    # Video (ViViT + token-routed MLP)
    encoder = Video.encoder(num_frames=16, hidden_size=768, num_experts=4)
    features = encoder(video)   # [B, C, T, H, W]

    # Omni — any-to-any (text + image + audio + video)
    model = Omni.model(hidden_size=1024, vocab_size=32000)
    out = model(text_ids=text, pixel_values=images)
    logits = out["logits"]
"""

from __future__ import annotations

from typing import Optional, Dict, Any, Union, List
import torch
import torch.nn as nn

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
    # Video
    VideoEncoder,
    VideoConfig,
    TubeletEmbedding,
    VideoTransformer,
    # Fusion
    MultimodalFusion,
    FusionConfig,
    CrossAttentionFusion,
    GatedFusion,
    ConcatProjection,
    PerceiverResampler,
    # Omni
    OmniModel,
    OmniConfig,
    PositionRoutedMLP,
    Modality,
    ModalityMLPConfig,
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


class Video:
    """
    Factory pour créer des encodeurs vidéo (ViViT + token-routed MLP).

    Usage:
        # Encoder basique
        encoder = Video.encoder(num_frames=16, hidden_size=768)
        features = encoder(video)   # [B, C, T, H, W]

        # Avec config complète
        encoder = Video.create(image_size=224, patch_size=16, num_frames=32)
    """

    @classmethod
    def encoder(
        cls,
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        temporal_patch_size: int = 2,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        num_experts: int = 4,
        output_dim: Optional[int] = None,
    ) -> nn.Module:
        """
        Video encoder (ViViT Factorised + token-routed MLP).

        Args:
            image_size: Taille des frames
            patch_size: Taille patch spatial
            num_frames: Nombre de frames
            temporal_patch_size: Taille patch temporel
            hidden_size: Dimension hidden
            num_layers: Nombre de layers
            num_heads: Nombre de heads
            num_experts: Experts par bloc MLP (routing par position spatiale)
            output_dim: Projection de sortie optionnelle
        """
        return VideoEncoder(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_experts=num_experts,
            output_dim=output_dim,
        )

    @classmethod
    def create(cls, **kwargs) -> nn.Module:
        """VideoEncoder avec kwargs libres (passe à VideoConfig)."""
        config = VideoConfig(**kwargs)
        return VideoTransformer(config)

    @classmethod
    def tubelets(
        cls,
        image_size: int = 224,
        patch_size: int = 16,
        num_frames: int = 16,
        temporal_patch_size: int = 2,
        hidden_size: int = 768,
    ) -> nn.Module:
        """Tubelet embedding seul (Conv3d)."""
        config = VideoConfig(
            image_size=image_size,
            patch_size=patch_size,
            num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
            hidden_size=hidden_size,
        )
        return TubeletEmbedding(config)


class Omni:
    """
    Factory pour créer des modèles any-to-any (text + image + audio + video).

    Usage:
        model = Omni.model(hidden_size=1024, vocab_size=32000)
        out = model(
            text_ids=text,
            pixel_values=images,
            audio_features=mel,
            video_frames=video,
        )
        logits = out["logits"]

        # Avec config complète
        config = Omni.config(hidden_size=2048, text_num_experts=16)
        model = Omni.from_config(config)
    """

    @classmethod
    def model(cls, **kwargs) -> nn.Module:
        """
        Modèle OmniModel (any-to-any).

        Chaque OmniBlock a 2 couches MLP en cascade :
          1. General MLP  — partagé par tous les tokens (general_num_experts)
          2. Specialised  — un par modalité, derivé de Modality enum

        Accepts flat kwargs pour les experts par modalité (traduits en dict):
            text_num_experts, text_intermediate_size
            image_num_experts, image_intermediate_size
            audio_num_experts, audio_intermediate_size
            video_num_experts, video_intermediate_size

        Ou passe directement un modality_mlp dict:
            modality_mlp={Modality.VIDEO: ModalityMLPConfig(8, 4096)}

        Exemples:
            model = Omni.model(hidden_size=1024, vocab_size=32000)

            model = Omni.model(
                hidden_size=2048,
                general_num_experts=8,
                video_num_experts=8,
                audio_num_experts=8,
            )
        """
        # Build modality_mlp dict from flat kwargs (e.g. text_num_experts=8)
        modal_cfg = {m: ModalityMLPConfig() for m in Modality}
        omni_kwargs = {}
        for key, val in kwargs.items():
            matched = False
            for m in Modality:
                prefix = m.name.lower()
                if key == f"{prefix}_num_experts":
                    modal_cfg[m] = ModalityMLPConfig(val, modal_cfg[m].intermediate_size)
                    matched = True
                    break
                elif key == f"{prefix}_intermediate_size":
                    modal_cfg[m] = ModalityMLPConfig(modal_cfg[m].num_experts, val)
                    matched = True
                    break
            if not matched:
                omni_kwargs[key] = val

        # Allow explicit modality_mlp dict to override
        if "modality_mlp" not in omni_kwargs:
            omni_kwargs["modality_mlp"] = modal_cfg

        return OmniModel(OmniConfig(**omni_kwargs))

    @classmethod
    def config(cls, **kwargs) -> OmniConfig:
        """Crée un OmniConfig."""
        return OmniConfig(**kwargs)

    @classmethod
    def from_config(cls, config: OmniConfig) -> nn.Module:
        """OmniModel depuis un OmniConfig existant."""
        return OmniModel(config)

    @classmethod
    def position_routed_mlp(
        cls,
        hidden_size: int = 768,
        intermediate_size: int = 3072,
        num_experts: int = 4,
    ) -> nn.Module:
        """PositionRoutedMLP standalone (réutilisable dans n'importe quel modèle)."""
        return PositionRoutedMLP(hidden_size, intermediate_size, num_experts)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Factories
    "Vision",
    "Audio",
    "Video",
    "Fusion",
    "Omni",
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
    # Direct classes - Video
    "VideoEncoder",
    "VideoConfig",
    "TubeletEmbedding",
    "VideoTransformer",
    # Direct classes - Fusion
    "MultimodalFusion",
    "FusionConfig",
    "CrossAttentionFusion",
    "GatedFusion",
    "ConcatProjection",
    "PerceiverResampler",
    # Direct classes - Omni
    "OmniModel",
    "OmniConfig",
    "PositionRoutedMLP",
    "Modality",
    "ModalityMLPConfig",
]
