"""Regression coverage for the multimodal fusion/audio bug fixes:

- ``GatedFusion`` raises a clear error instead of an ``assert`` on a
  modality-count mismatch (asserts are stripped under ``python -O``).
- ``MultimodalFusion.forward`` raises instead of silently returning ``None``
  for an unrecognized fusion type.
- ``MelSpectrogramEncoder`` raises a clear error instead of silently
  truncating ``expert_ids`` (and crashing deep inside the MLP with a
  confusing shape error) when the input exceeds the configured capacity.
"""

from __future__ import annotations

import pytest
import torch

from complexity.multimodal.audio import AudioConfig, MelSpectrogramEncoder
from complexity.multimodal.fusion import FusionConfig, GatedFusion, MultimodalFusion


def test_gated_fusion_raises_a_clear_error_on_modality_count_mismatch():
    fusion = GatedFusion(hidden_size=8, num_modalities=2)
    feature = torch.randn(2, 4, 8)

    with pytest.raises(ValueError, match="configured for 2 modalities"):
        fusion(feature)


def test_gated_fusion_still_fuses_the_expected_modality_count():
    fusion = GatedFusion(hidden_size=8, num_modalities=2)
    text = torch.randn(2, 4, 8)
    vision = torch.randn(2, 4, 8)
    output = fusion(text, vision)
    assert output.shape == (2, 4, 8)


def test_multimodal_fusion_raises_instead_of_returning_none_for_unknown_type():
    fusion = MultimodalFusion(hidden_size=8, num_heads=2, fusion_type="concat")
    # __init__ validates fusion_type, so simulate a type added to __init__
    # without a matching forward() branch by mutating the built instance.
    fusion.fusion_type = "bogus"
    text = torch.randn(2, 3, 8)
    other = torch.randn(2, 3, 8)

    with pytest.raises(ValueError, match="Unknown fusion type: bogus"):
        fusion(text, other)


def test_mel_spectrogram_encoder_rejects_input_exceeding_configured_capacity():
    config = AudioConfig(
        n_mels=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_length=32,
        num_experts=2,
    )
    encoder = MelSpectrogramEncoder(config)
    # After the encoder's stride-2 conv stack, this input produces more
    # frames than expert_ids_table (sized max_length // 2) can serve.
    long_mel = torch.randn(1, config.n_mels, config.max_length * 2)

    with pytest.raises(ValueError, match="exceeding max_length"):
        encoder(long_mel)


def test_mel_spectrogram_encoder_accepts_input_within_configured_capacity():
    config = AudioConfig(
        n_mels=8,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        intermediate_size=32,
        max_length=32,
        num_experts=2,
    )
    encoder = MelSpectrogramEncoder(config)
    mel = torch.randn(2, config.n_mels, config.max_length)
    output = encoder(mel)
    assert torch.isfinite(output["last_hidden_state"]).all()
