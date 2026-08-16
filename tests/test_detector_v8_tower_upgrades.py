"""Regression coverage for the v8 vision-tower upgrades: the progressive conv
stem and the windowed relative position bias.
"""

from __future__ import annotations

import pytest
import torch

from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.generative.detection.hierarchical_tower import (
    HierarchicalTRHashVisionTower,
    _build_patch_stem,
    _build_relative_position_index,
    _SpatialSelfAttention,
)


def _tiny_config(**overrides) -> TRHashDetectorConfig:
    fields = dict(
        architecture_version=8,
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_shared_width=16,
        vision_expert_width=8,
        num_classes=5,
        end_to_end=False,
    )
    fields.update(overrides)
    return TRHashDetectorConfig(**fields)


# --- progressive patch stem -------------------------------------------------


def test_patch_stem_rejects_non_power_of_two_patch_size():
    with pytest.raises(ValueError, match="power-of-two"):
        _build_patch_stem(3, 32, 12)


@pytest.mark.parametrize("patch_size", (2, 4, 8, 16))
def test_patch_stem_downsamples_to_exactly_one_token_per_patch(patch_size):
    hidden_size = 32
    stem = _build_patch_stem(3, hidden_size, patch_size)
    pixels = torch.randn(2, 3, patch_size * 5, patch_size * 5)
    output = stem(pixels)
    assert output.shape == (2, hidden_size, 5, 5)


def test_patch_stem_final_layer_is_a_real_conv_matching_hidden_size():
    stem = _build_patch_stem(3, 40, 8)
    final = stem[-1]
    assert isinstance(final, torch.nn.Conv2d)
    assert final.out_channels == 40


def test_patch_stem_is_cheaper_than_an_equivalent_single_large_kernel_conv():
    hidden_size, patch_size = 128, 8
    stem_params = sum(p.numel() for p in _build_patch_stem(3, hidden_size, patch_size).parameters())
    single_conv_params = sum(
        p.numel()
        for p in torch.nn.Conv2d(
            3, hidden_size, kernel_size=patch_size, stride=patch_size
        ).parameters()
    )
    assert stem_params < single_conv_params


def test_tower_patch_embed_is_the_progressive_stem_and_trains_end_to_end():
    config = _tiny_config()
    tower = HierarchicalTRHashVisionTower(config)
    assert isinstance(tower.patch_embed, torch.nn.Sequential)

    outputs = tower(torch.randn(2, 3, 32, 32))
    outputs[-1].square().mean().backward()
    assert tower.patch_embed[-1].weight.grad is not None
    assert tower.patch_embed[-1].weight.grad.abs().sum() > 0


# --- relative position bias -------------------------------------------------


def test_relative_position_index_covers_the_full_bias_table_range():
    window_size = 4
    index = _build_relative_position_index(window_size)
    assert index.shape == (window_size**2, window_size**2)
    assert index.min().item() == 0
    assert index.max().item() == (2 * window_size - 1) ** 2 - 1
    # A token's offset from itself is always the same canonical (center) bin.
    diagonal = index.diagonal()
    assert torch.all(diagonal == diagonal[0])


def test_attention_only_builds_relative_bias_for_positive_window_size():
    windowed = _SpatialSelfAttention(hidden_size=16, num_heads=4, dropout=0.0, window_size=4)
    assert windowed.relative_position_bias_table is not None

    global_attention = _SpatialSelfAttention(
        hidden_size=16, num_heads=4, dropout=0.0, window_size=0
    )
    assert global_attention.relative_position_bias_table is None


def test_windowed_attention_applies_bias_and_respects_padding_mask():
    torch.manual_seed(0)
    window_size = 4
    attention = _SpatialSelfAttention(
        hidden_size=16, num_heads=4, dropout=0.0, window_size=window_size
    )
    tokens = torch.randn(2, window_size**2, 16)

    unmasked = attention(tokens)
    assert torch.isfinite(unmasked).all()

    # Invalidate every key but the first: every query must collapse onto it.
    key_mask = torch.zeros(2, window_size**2, dtype=torch.bool)
    key_mask[:, 0] = True
    masked = attention(tokens, key_mask=key_mask)
    assert torch.isfinite(masked).all()
    # Every query attends entirely to the single valid key, so the attended
    # value collapses to that key's value projection regardless of query.
    assert torch.allclose(masked, masked[:, :1, :].expand_as(masked), atol=1e-5)


def test_relative_position_bias_table_receives_gradients_through_a_windowed_stage():
    config = _tiny_config(vision_window_size=2, vision_stage_depths=(1, 1, 1))
    model = TRHashObjectDetector(config)
    windowed_block = model.tower.stages[0][0]
    assert windowed_block.attention.relative_position_bias_table is not None

    raw = model(torch.randn(2, 3, 32, 32))
    raw.square().mean().backward()

    gradient = windowed_block.attention.relative_position_bias_table.grad
    assert gradient is not None
    assert gradient.abs().sum() > 0


def test_global_final_stage_has_no_relative_position_bias():
    config = _tiny_config(vision_window_size=2, vision_stage_depths=(1, 1, 1))
    model = TRHashObjectDetector(config)
    global_block = model.tower.stages[-1][0]
    assert global_block.window_size == 0
    assert global_block.attention.relative_position_bias_table is None
