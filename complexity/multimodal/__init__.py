"""Multimodal encoders and fusion modules.

New towers and routed feedforwards use the common Source67 TRHashEngine
with raw position IDs. Self-attention uses GQA, routed heads and V0; task
cross-attention remains specific to its modality. Explicit legacy configs
preserve the former position-modulo top1 BMM and dense fallback layouts.
These modality extensions do not inherit the measured text-only score.

Full image/text and speech models are available under ``complexity.generative``
and ``complexity.generative.audio`` respectively.
"""

from .vision import (
    VisionEncoder,
    VisionConfig,
    PatchEmbedding,
    VisionTransformer,
    VisionTokenRoutedMLP,
    CLIPVisionEncoder,
    SigLIPEncoder,
)

from .audio import (
    AudioEncoder,
    AudioConfig,
    MelSpectrogramEncoder,
    WhisperEncoder,
    AudioConvStack,
    AudioTokenRoutedMLP,
)

from .video import (
    VideoEncoder,
    VideoConfig,
    TubeletEmbedding,
    VideoTransformer,
    VideoTokenRoutedMLP,
    SpatioTemporalBlock,
)

from .fusion import (
    MultimodalFusion,
    FusionConfig,
    CrossAttentionFusion,
    GatedFusion,
    ConcatProjection,
    PerceiverResampler,
    FusionTokenRoutedMLP,
    VisionLanguageConnector,
)

__all__ = [
    # Vision
    "VisionEncoder",
    "VisionConfig",
    "PatchEmbedding",
    "VisionTransformer",
    "VisionTokenRoutedMLP",
    "CLIPVisionEncoder",
    "SigLIPEncoder",
    # Audio
    "AudioEncoder",
    "AudioConfig",
    "MelSpectrogramEncoder",
    "WhisperEncoder",
    "AudioConvStack",
    "AudioTokenRoutedMLP",
    # Video
    "VideoEncoder",
    "VideoConfig",
    "TubeletEmbedding",
    "VideoTransformer",
    "VideoTokenRoutedMLP",
    "SpatioTemporalBlock",
    # Fusion
    "MultimodalFusion",
    "FusionConfig",
    "CrossAttentionFusion",
    "GatedFusion",
    "ConcatProjection",
    "PerceiverResampler",
    "FusionTokenRoutedMLP",
    "VisionLanguageConnector",
]
