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
    model = Robot.model(num_classes=40, num_experts=8, top_k=2)

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


class Vision:
    """
    Factory pour créer des encodeurs vision.

    ATTENTION : ces encodeurs routent par ``position % num_experts`` (voir
    ``complexity.multimodal``), pas par table de hachage déterministe par ID
    -- ils ne respectent pas le contrat de routing TR-Hash du reste du
    framework. Pour un encodeur vision conforme, utiliser
    ``complexity.generative.vision_language.vision_tower.TRHashVisionTower``
    (ou ``Omni.model(...)`` pour un modèle image+texte->texte complet) --
    attention, son contrat de sortie diffère (features par patch, pas de
    pooling CLS).

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

    ATTENTION : ces encodeurs routent par ``position % num_experts``, pas par
    table de hachage déterministe -- non conformes au routing TR-Hash du
    reste du framework. Ils attendent aussi un spectrogramme mel déjà
    calculé ``[B, n_mels, frames]``. Pour un encodeur conforme qui calcule le
    mel lui-même à partir d'une waveform brute ``[B, samples]``, utiliser
    ``complexity.generative.audio.encoder.AudioEncoder`` (ou
    ``complexity.generative.audio.TRHashSpeechToText`` pour un pipeline
    complet) -- ce n'est pas un remplacement direct, le format d'entrée
    diffère.

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

    ATTENTION : le MLP interne de ces modules de fusion route par
    ``query_position % num_experts``, pas par table de hachage déterministe
    -- non conforme au routing TR-Hash du reste du framework. Il n'existe
    pas de remplacement direct : dans le chemin conforme, la fusion se fait
    en assignant des routes déterministes aux tokens image/audio puis en les
    faisant passer par le même décodeur TR-Hash MoE que le texte -- voir
    ``Omni`` (``complexity.generative.vision_language.TRHashImageTextToText``)
    plutôt qu'un module de fusion séparé.

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

    ATTENTION : route par ``position % num_experts``, pas par table de
    hachage déterministe -- non conforme au routing TR-Hash du reste du
    framework. Pour un encodeur vidéo conforme, utiliser
    ``complexity.generative.video.TRHashVideoTower`` -- son contrat de
    sortie diffère (pas de pooling intégré).

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

    Remplace l'ancien ``OmniModel`` (routing par position de séquence,
    ``pos % num_experts`` — non conforme au routing TR-Hash par ID de
    token / table de hachage). ``TRHashImageTextToText`` assigne des routes
    déterministes fixes par patch d'image et les fait passer par le même
    décodeur TR-Hash MoE que les tokens de texte — voir
    ``complexity.generative.vision_language``.

    Portée actuelle : image + texte -> texte uniquement (pas encore audio ni
    vidéo).

    Usage:
        model = Omni.model(hidden_size=768, vocab_size=32000)
        out = model(pixel_values=images, input_ids=text_ids, labels=labels)
        logits = out["logits"]

        # Avec config complète
        config = Omni.config(hidden_size=2048, num_experts=8)
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
            model = Omni.model(hidden_size=1024, num_experts=8, top_k=2)
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
