"""
Multi-modal module for framework-complexity.

Supports:
- Vision encoding (ViT, CLIP, SigLIP)
- Audio encoding (Whisper-style, Mel)
- Video encoding (ViViT factorised)
- Multi-modal fusion

The encoders below default to ``num_experts=4``, which routes internally
through a position-routed MLP (``*TokenRoutedMLP``: expert = sequence/patch
position mod num_experts) predating the framework's TR-Hash scoping — it is
not token-ID/hash-table routing and does not meet the current routing
guardrail. Pass ``num_experts=1`` explicitly to use a plain dense MLP inside
the encoder tower instead; that's what the compliant multimodal model below
does (its vision tower is constructed with ``num_experts=1``).

For a full, TR-Hash-compliant image + text -> text model — modality tokens
get deterministic route IDs and dispatch through the same TR-Hash MoE
decoder as real text tokens — use
``complexity.generative.vision_language.TRHashImageTextToText`` (or the
``Omni`` factory in ``complexity.api.multimodal``) instead of building a
custom any-to-any model from these encoders directly. The historical
``OmniModel`` (any-to-any, position-routed) was removed for the same
non-compliance reason.

For speech, see ``complexity.generative.audio`` (``TRHashSpeechToText``,
``TRHashTextToSpeech``) rather than this module's ``AudioEncoder``.

Usage:
    from complexity.multimodal import VisionEncoder, AudioEncoder, VideoEncoder

    # Vision
    vision = VisionEncoder(image_size=224, patch_size=16, hidden_size=768, num_experts=1)
    image_features = vision(images)

    # Audio
    audio = AudioEncoder(n_mels=80, hidden_size=768, num_experts=1)
    audio_features = audio(mel_spectrograms)

    # Video
    video = VideoEncoder(num_frames=16, hidden_size=768, num_experts=1)
    video_features = video(video_tensor)   # [B, C, T, H, W]
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
