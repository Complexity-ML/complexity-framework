"""TR-Hash speech-to-text (ASR) and text-to-speech (TTS) models.

Both are built on the canonical TR-Hash MoE decoder — audio is never
routed by a learned router. ``AudioEncoder`` itself also routes frames
through real TR-Hash MoE experts (by fixed frame position, same principle
as ``TRHashVisionTower``), not a plain dense pass-through. Speech-to-text
follows the same overall pattern as
``complexity.generative.vision_language`` (deterministic route IDs for a
fixed-size modality prefix, fed through the shared decoder). Text-to-speech
follows the same pattern as ``complexity.generative.image`` (a DiT-style
rectified-flow model whose FFN is a ``TRHashEngine``, route IDs keyed on
timestep bucket + position).

Usage:
    from complexity.generative.audio import (
        TRHashSpeechToText, TRHashSpeechToTextConfig,
        TRHashTextToSpeech, TRHashAudioConfig,
    )

    asr = TRHashSpeechToText(TRHashSpeechToTextConfig())
    out = asr(waveform, input_ids, labels=labels)

    tts = TRHashTextToSpeech(TRHashAudioConfig())
    loss = tts.flow_matching_loss(mel, caption_ids, caption_mask)
    mel = tts.sample(caption_ids, caption_mask, steps=30)
"""

from .config import TRHashAudioConfig, TRHashSpeechToTextConfig
from .encoder import AudioEncoder, AudioEncoderConfig
from .mel import LogMelSpectrogram, build_mel_filterbank
from .model import TokenResampler, TRHashSpeechToText, TRHashTextToSpeech

__all__ = [
    "TRHashAudioConfig",
    "TRHashSpeechToTextConfig",
    "AudioEncoder",
    "AudioEncoderConfig",
    "LogMelSpectrogram",
    "build_mel_filterbank",
    "TokenResampler",
    "TRHashSpeechToText",
    "TRHashTextToSpeech",
]
