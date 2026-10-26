"""
Multimodal API - Vision, Audio, Video, Robot, Omni faciles.
=====================================================

Usage:
    from complexity.api import Vision, Audio, Video, Fusion, Robot, Omni

    # Vision (avec token-routed MLP)
    encoder = Vision.encoder(image_size=224, hidden_size=768, num_experts=4)
    features = encoder(images)

    # Audio
    encoder = Audio.whisper(hidden_size=768, num_experts=4)

    # Video (ViViT + token-routed MLP)
    encoder = Video.encoder(num_frames=16, hidden_size=768, num_experts=4)
    features = encoder(video)   # [B, C, T, H, W]

    # Robot — perception multimodale par fusion de capteurs
    model = Robot.model(num_classes=40, num_experts=4, top_k=2)

    # Omni — image + text -> text
    model = Omni.model(hidden_size=1024, vocab_size=32000)
    out = model(pixel_values=images, input_ids=text_ids)
    logits = out["logits"]
"""

from __future__ import annotations

from typing import List, Optional

import torch.nn as nn

from complexity.generative.sensor_fusion import (
    SENSOR_MODALITIES,
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)
from complexity.generative.vision_language import (
    TRHashImageTextToText,
    TRHashVisionLanguageConfig,
)
from complexity.multimodal import (
    AudioConfig,
    AudioConvStack,
    # Audio
    AudioEncoder,
    CLIPVisionEncoder,
    ConcatProjection,
    CrossAttentionFusion,
    FusionConfig,
    GatedFusion,
    MelSpectrogramEncoder,
    # Fusion
    MultimodalFusion,
    PatchEmbedding,
    PerceiverResampler,
    SigLIPEncoder,
    TubeletEmbedding,
    VideoConfig,
    # Video
    VideoEncoder,
    VideoTransformer,
    VisionConfig,
    # Vision
    VisionEncoder,
    VisionTransformer,
    WhisperEncoder,
)


def _rename_factory_kwargs(kwargs, aliases):
    """Accept config or wrapper dimension names without silent conflicts."""
    values = dict(kwargs)
    for source, target in aliases.items():
        if source in values:
            if target in values and values[target] != values[source]:
                raise ValueError(f"conflicting factory arguments: {source} and {target}")
            values[target] = values.pop(source)
    return values


class Vision:
    """
    Factory pour créer des encodeurs vision.

    Les nouveaux encodeurs utilisent Source67 : IDs de patches originaux,
    MLP contextuel à deux lags et attention GQA avec têtes routées et V0.
    ``architecture_profile="legacy"`` conserve le précédent contrat top1/MHA.
    Ces extensions vision n'héritent pas du score mesuré sur texte.

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

        if vision_type == "transformer":
            values = _rename_factory_kwargs(
                kwargs, {"num_layers": "num_hidden_layers", "num_heads": "num_attention_heads"}
            )
            return vision_cls(VisionConfig(**values))
        values = _rename_factory_kwargs(
            kwargs, {"num_hidden_layers": "num_layers", "num_attention_heads": "num_heads"}
        )
        return vision_cls(**values)

    @classmethod
    def encoder(
        cls,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        **kwargs,
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
            **kwargs,
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
    def patches(
        cls, image_size: int = 224, patch_size: int = 16, hidden_size: int = 768
    ) -> nn.Module:
        """Patch embedding layer seul."""
        return PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
        )


class Audio:
    """
    Factory pour créer des encodeurs audio.

    Les nouveaux encodeurs utilisent Source67 avec les positions de frames
    originales ; le profil legacy reste explicite. Ils attendent un mel déjà
    calculé ``[B, n_mels, frames]``. L'encodeur
    ``complexity.generative.audio.encoder.AudioEncoder`` accepte au contraire
    une waveform ``[B, samples]`` et calcule lui-même le mel.

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

        if audio_type == "mel":
            values = _rename_factory_kwargs(
                kwargs, {"num_layers": "num_hidden_layers", "num_heads": "num_attention_heads"}
            )
            return audio_cls(AudioConfig(**values))
        values = _rename_factory_kwargs(
            kwargs, {"num_hidden_layers": "num_layers", "num_attention_heads": "num_heads"}
        )
        return audio_cls(**values)

    @classmethod
    def encoder(
        cls,
        n_mels: int = 80,
        hidden_size: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        **kwargs,
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
            **kwargs,
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

    Cross-attention et Perceiver utilisent le MLP Source67 avec les IDs de
    requêtes originaux. Leurs cross-attentions restent spécifiques à la tâche.
    Les modules gated et concat gardent leurs projections dédiées.

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

        values = _rename_factory_kwargs(
            kwargs, {"num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
        )
        return fusion_cls(**values)

    @classmethod
    def cross_attention(
        cls, hidden_size: int = 768, num_heads: int = 12, num_layers: int = 2, **kwargs
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
            **kwargs,
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
        cls, hidden_size: int = 768, num_latents: int = 64, num_layers: int = 2, **kwargs
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
            **kwargs,
        )

    @classmethod
    def multimodal(cls, hidden_size: int = 768, **kwargs) -> nn.Module:
        """Fusion multimodale générique."""
        return cls.create("multimodal", hidden_size=hidden_size, **kwargs)


class Video:
    """
    Factory pour créer des encodeurs vidéo (ViViT + token-routed MLP).

    Le profil Source67 utilise les positions de tubelets originales pour le
    MLP contextuel et les attentions GQA spatiale/temporelle. Le profil legacy
    conserve l'ancien contrat. Aucun score vidéo n'est déduit du score texte.

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
        *,
        architecture_profile: str = "source67",
        num_key_value_heads: Optional[int] = None,
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
            num_experts: Experts par bloc MLP (4 pour Source67).
            architecture_profile: Source67 par défaut, legacy pour les anciens modèles.
            num_key_value_heads: Têtes KV explicites ; défaut 1 en Source67.
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
            architecture_profile=architecture_profile,
            num_key_value_heads=num_key_value_heads,
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


class Robot:
    """Factory TR-Hash Robotics pour la perception multimodale.

    Le modèle fusionne profondeur, infrarouge, thermique, IMU, radar et
    squelette. Le routage hash reste sparse tandis que la voie SwiGLU
    partagée garantit une capacité dense pour chaque token.

    Cette API décrit un modèle de perception/classification, pas encore une
    politique de contrôle robotique ou un agent autonome.
    """

    MODALITIES = SENSOR_MODALITIES

    @classmethod
    def model(cls, **kwargs) -> nn.Module:
        """Construit un modèle de perception TR-Hash Robotics."""
        return TRHashSensorFusionClassifier(TRHashSensorFusionConfig(**kwargs))

    @classmethod
    def config(cls, **kwargs) -> TRHashSensorFusionConfig:
        """Construit la configuration du modèle robotique."""
        return TRHashSensorFusionConfig(**kwargs)

    @classmethod
    def from_config(cls, config: TRHashSensorFusionConfig) -> nn.Module:
        """Construit le modèle depuis une configuration validée."""
        return TRHashSensorFusionClassifier(config)


class Omni:
    """
    Factory pour créer des modèles multimodaux (image + texte -> texte).

    Le modèle complet Source67 assigne des IDs aux préfixes visuels et aux
    tokens texte dans le même décodeur. La tour vision utilise également le
    profil commun ; le connecteur croisé garde sa capacité dédiée.

    Portée actuelle : image + texte -> texte uniquement (pas encore audio ni
    vidéo).

    Usage:
        model = Omni.model(hidden_size=768, vocab_size=32000)
        out = model(pixel_values=images, input_ids=text_ids, labels=labels)
        logits = out["logits"]

        # Avec config complète
        config = Omni.config(hidden_size=768, num_experts=4)
        model = Omni.from_config(config)
    """

    @classmethod
    def model(cls, **kwargs) -> nn.Module:
        """Modèle TRHashImageTextToText (image + texte -> texte).

        Accepte les kwargs de ``TRHashVisionLanguageConfig`` (image_size,
        patch_size, vision_hidden_size, num_visual_tokens, hidden_size,
        num_experts, top_k, ...).

        Exemples:
            model = Omni.model(hidden_size=768, vocab_size=32000)
            model = Omni.model(hidden_size=768, num_experts=4, top_k=2)
        """
        return TRHashImageTextToText(TRHashVisionLanguageConfig(**kwargs))

    @classmethod
    def config(cls, **kwargs) -> TRHashVisionLanguageConfig:
        """Crée un TRHashVisionLanguageConfig."""
        return TRHashVisionLanguageConfig(**kwargs)

    @classmethod
    def from_config(cls, config: TRHashVisionLanguageConfig) -> nn.Module:
        """TRHashImageTextToText depuis un TRHashVisionLanguageConfig existant."""
        return TRHashImageTextToText(config)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Factories
    "Vision",
    "Audio",
    "Video",
    "Fusion",
    "Robot",
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
    # Direct classes - Robot
    "TRHashSensorFusionClassifier",
    "TRHashSensorFusionConfig",
    # Direct classes - Omni
    "TRHashImageTextToText",
    "TRHashVisionLanguageConfig",
]
