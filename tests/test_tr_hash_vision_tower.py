import torch

from complexity.generative.vision_language.vision_tower import (
    TRHashVisionClassifier,
    TRHashVisionTower,
    TRHashVisionTowerConfig,
)


def _tiny_config(**overrides) -> TRHashVisionTowerConfig:
    fields = dict(
        image_size=32,
        patch_size=8,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_experts=4,
        top_k=2,
        expert_width=16,
    )
    fields.update(overrides)
    return TRHashVisionTowerConfig(**fields)


def test_patches_route_through_more_than_one_expert():
    """The whole point of this tower vs a plain ViT: patches actually
    dispatch across multiple experts by fixed spatial position."""
    config = _tiny_config()
    tower = TRHashVisionTower(config)
    routes = tower.route_ids
    assert routes.shape == (config.num_patches,)
    assert int(routes.min()) >= 0
    assert int(routes.max()) < config.route_vocab_size
    # Route table itself (per block) must actually use more than one expert.
    table = tower.blocks[0].mlp.route_table
    assert torch.unique(table).numel() > 1


def test_route_ids_depend_only_on_patch_position_not_pixel_content():
    """Two independently constructed towers with the same config must agree
    on route assignment — it's a fixed function of patch position, never
    learned from data or derived from pixel content."""
    config = _tiny_config()
    tower_a = TRHashVisionTower(config)
    tower_b = TRHashVisionTower(config)
    assert torch.equal(tower_a.route_ids, tower_b.route_ids)
    assert tower_a.route_ids.shape == (config.num_patches,)


def test_forward_shape_and_gradients_reach_expert_weights():
    torch.manual_seed(3)
    config = _tiny_config()
    tower = TRHashVisionTower(config)
    pixels = torch.randn(2, 3, 32, 32)
    features = tower(pixels)
    assert features.shape == (2, config.num_patches, config.hidden_size)

    features.float().square().mean().backward()
    for block in tower.blocks:
        grad = block.mlp.expert_gate.grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0


def test_rejects_wrong_pixel_shape():
    config = _tiny_config()
    tower = TRHashVisionTower(config)
    try:
        tower(torch.randn(2, 3, 16, 16))
    except ValueError as exc:
        assert "expected pixel_values shape" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_classifier_produces_logits_and_loss():
    torch.manual_seed(5)
    config = _tiny_config()
    classifier = TRHashVisionClassifier(config, num_classes=7)
    pixels = torch.randn(3, 3, 32, 32)
    labels = torch.randint(0, 7, (3,))

    out = classifier(pixels, labels=labels)
    assert out["logits"].shape == (3, 7)
    assert out["pooled_features"].shape == (3, config.hidden_size)
    assert out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])

    out["loss"].backward()
    assert classifier.head.weight.grad is not None
    assert classifier.tower.blocks[0].mlp.expert_gate.grad is not None


def test_top_k_cannot_exceed_num_experts():
    import pytest

    with pytest.raises(ValueError, match="top_k cannot exceed"):
        TRHashVisionTowerConfig(num_experts=2, top_k=4)
