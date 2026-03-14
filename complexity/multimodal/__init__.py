"""
Multi-modal module for framework-complexity.

Supports:
- Vision encoding (ViT, CLIP, SigLIP) with token-routed MLP
- Audio encoding (Whisper-style, Mel) with token-routed MLP
- Video encoding (ViViT factorised) with token-routed MLP
- Multi-modal fusion with token-routed MLP
- OmniModel: unified any-to-any model (text + image + audio + video)

Usage:
    from complexity.multimodal import VisionEncoder, AudioEncoder, VideoEncoder
    from complexity.multimodal import OmniModel, OmniConfig

    # Vision
    vision = VisionEncoder(image_size=224, patch_size=16, hidden_size=768)
    image_features = vision(images)

    # Audio
    audio = AudioEncoder(n_mels=80, hidden_size=768)
    audio_features = audio(mel_spectrograms)

    # Video
    video = VideoEncoder(num_frames=16, hidden_size=768)
    video_features = video(video_tensor)   # [B, C, T, H, W]

    # Omni — any-to-any
    model = OmniModel(OmniConfig(hidden_size=1024, vocab_size=32000))
    out = model(text_ids=text, pixel_values=images, video_frames=video)
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

from .omni import (
    OmniModel,
    OmniConfig,
    OmniBlock,
    PositionRoutedMLP,
    TEXT,
    IMAGE,
    AUDIO,
    VIDEO,
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
    # Omni
    "OmniModel",
    "OmniConfig",
    "OmniBlock",
    "PositionRoutedMLP",
    "TEXT",
    "IMAGE",
    "AUDIO",
    "VIDEO",
]
