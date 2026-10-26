"""Multimodal encoders and fusion modules.

TR-HASH feedforwards use shared experts and contextual top-2 assignments.
Raw position IDs feed four-query/one-KV self-attention with head and value
routing. Cross-attention and resampling remain task-specific connectors.
These modality extensions have no validated quality score in this project.
Visual models live under complexity.vision; speech under complexity.generative.audio.
"""

from .audio import (
    AudioConfig,
    AudioConvStack,
    AudioEncoder,
    AudioTokenRoutedMLP,
    MelSpectrogramEncoder,
    WhisperEncoder,
)
from .fusion import (
    ConcatProjection,
    CrossAttentionFusion,
    FusionConfig,
    FusionTokenRoutedMLP,
    GatedFusion,
    MultimodalFusion,
    PerceiverResampler,
    VisionLanguageConnector,
)
from .video import (
    SpatioTemporalBlock,
    TubeletEmbedding,
    VideoConfig,
    VideoEncoder,
    VideoTokenRoutedMLP,
    VideoTransformer,
)
from .vision import (
    CLIPVisionEncoder,
    PatchEmbedding,
    SigLIPEncoder,
    VisionConfig,
    VisionEncoder,
    VisionTokenRoutedMLP,
    VisionTransformer,
)

__all__ = [
    "VisionEncoder",
    "VisionConfig",
    "PatchEmbedding",
    "VisionTransformer",
    "VisionTokenRoutedMLP",
    "CLIPVisionEncoder",
    "SigLIPEncoder",
    "AudioEncoder",
    "AudioConfig",
    "MelSpectrogramEncoder",
    "WhisperEncoder",
    "AudioConvStack",
    "AudioTokenRoutedMLP",
    "VideoEncoder",
    "VideoConfig",
    "TubeletEmbedding",
    "VideoTransformer",
    "VideoTokenRoutedMLP",
    "SpatioTemporalBlock",
    "MultimodalFusion",
    "FusionConfig",
    "CrossAttentionFusion",
    "GatedFusion",
    "ConcatProjection",
    "PerceiverResampler",
    "FusionTokenRoutedMLP",
    "VisionLanguageConnector",
]
