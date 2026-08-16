import json

import torch
from safetensors.torch import save_file

from complexity.generative.detection.config import TRHashDetectorConfig
from complexity.generative.detection.hierarchical_tower import (
    HierarchicalTRHashVisionTower,
)
from complexity.generative.sensor_fusion import (
    SENSOR_MODALITIES,
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)
from complexity.generative.sensor_fusion.transfer import load_pretrained_visual_tower


def _tiny_config(**overrides) -> TRHashSensorFusionConfig:
    values = dict(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_experts=8,
        top_k=2,
        shared_width=32,
        expert_width=8,
        precision="fp32",
        classifier_dropout=0.0,
        visual_channels=(3, 1, 3),
        visual_token_grid=(2, 2, 2),
        vision_image_size=16,
        vision_patch_size=4,
        vision_hidden_size=32,
        vision_layers=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_shared_width=16,
        vision_expert_width=8,
        vision_stage_depths=(1, 1),
        vision_window_size=2,
        imu_features=45,
        radar_features=16,
        sequence_tokens=4,
    )
    values.update(overrides)
    return TRHashSensorFusionConfig(**values)


def _inputs(batch: int = 2) -> dict[str, torch.Tensor]:
    return {
        "depth": torch.randn(batch, 3, 4, 16, 16),
        "ir": torch.randn(batch, 1, 4, 16, 16),
        "thermal": torch.randn(batch, 3, 4, 16, 16),
        "imu": torch.randn(batch, 12, 45),
        "radar": torch.randn(batch, 10, 16),
        "skeleton": torch.randn(batch, 8, 17, 3),
    }


def test_config_uses_eight_experts_and_modality_position_routes():
    config = _tiny_config()
    model = TRHashSensorFusionClassifier(config)
    assert config.num_experts == 8
    assert model.route_ids.shape == (config.route_vocab_size,)
    assert torch.equal(model.route_ids, torch.arange(config.route_vocab_size))
    assert tuple(model.encoders) == SENSOR_MODALITIES
    for block in model.blocks:
        assert block.mlp.config.num_experts == 8
        assert block.mlp.config.top_k == 2
        assert torch.unique(block.mlp.route_table).numel() == 8


def test_all_modalities_forward_loss_and_expert_gradients():
    torch.manual_seed(7)
    model = TRHashSensorFusionClassifier(_tiny_config())
    labels = torch.tensor([1, 3])
    output = model(_inputs(), labels)
    assert output["logits"].shape == (2, 40)
    assert output["pooled_features"].shape == (2, 32)
    assert output["token_mask"].all()
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    for block in model.blocks:
        gradient = block.mlp.expert_gate.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_class_hash_head_is_residual_balanced_and_trainable():
    config = _tiny_config(
        num_classes=8,
        class_hash_expert_width=8,
        class_hash_initial_scale=0.05,
    )
    model = TRHashSensorFusionClassifier(config)
    assert model.class_hash_head is not None
    assert model.class_hash_head.mlp.route_table.shape == (config.top_k, 8)
    assert torch.unique(model.class_hash_head.mlp.route_table).numel() == 8

    output = model(_inputs(), torch.tensor([1, 3]))
    assert output["class_hash_logits"].shape == (2, 8)
    assert torch.allclose(
        output["fused_logits"],
        model.head(output["pooled_features"]) + output["class_hash_logits"],
    )
    output["loss"].backward()
    assert model.class_hash_head.mlp.expert_gate.grad is not None
    assert model.class_hash_head.scale.grad is not None


def test_class_hash_head_is_active_by_default():
    model = TRHashSensorFusionClassifier(_tiny_config())
    output = model(_inputs())
    assert model.class_hash_head is not None
    assert output["class_hash_logits"].shape == (2, model.config.num_classes)


def test_missing_modalities_are_masked_per_sample():
    model = TRHashSensorFusionClassifier(_tiny_config())
    inputs = {
        "depth": torch.randn(2, 3, 4, 16, 16),
        "imu": torch.randn(2, 12, 45),
    }
    output = model(
        inputs,
        modality_mask={
            "depth": torch.tensor([True, False]),
            "imu": torch.tensor([True, True]),
        },
    )
    depth_tokens = model.config.visual_tokens
    imu_offset = 1 + 3 * depth_tokens  # +1 for the fusion CLS token
    assert output["token_mask"][:, 0].all()  # CLS token always present
    assert output["token_mask"][0, 1 : 1 + depth_tokens].all()
    assert not output["token_mask"][1, 1 : 1 + depth_tokens].any()
    assert output["token_mask"][:, imu_offset : imu_offset + model.config.sequence_tokens].all()
    assert output["logits"].shape == (2, 40)


def test_default_model_is_a_real_sized_configuration():
    model = TRHashSensorFusionClassifier()
    assert model.config.num_experts == 8
    assert model.num_parameters() > 5_000_000


def test_cls_token_pools_the_fused_state():
    model = TRHashSensorFusionClassifier(_tiny_config())
    output = model(_inputs())
    assert model.cls_token is not None
    assert output["token_mask"].shape == (2, model.config.route_vocab_size)
    assert output["token_mask"][:, 0].all()


def test_shares_tr_hash_vision_tower_across_visual_modalities():
    model = TRHashSensorFusionClassifier(_tiny_config())
    output = model(_inputs())
    assert model.vision_tower is not None
    assert model.visual_projection is not None
    assert output["logits"].shape == (2, 40)
    tower_names = [name for name, _ in model.named_modules() if name == "vision_tower"]
    assert tower_names == ["vision_tower"]


def test_confidence_gates_available_modalities_and_backpropagates():
    config = _tiny_config()
    model = TRHashSensorFusionClassifier(config)
    masks = {name: torch.tensor([True, name == "thermal"]) for name in SENSOR_MODALITIES}
    output = model(_inputs(), modality_mask=masks)
    assert output["modality_logits"].shape == (2, len(SENSOR_MODALITIES), 40)
    assert output["modality_weights"].shape == (2, len(SENSOR_MODALITIES), 40)
    assert torch.allclose(
        output["modality_weights"].sum(dim=1),
        torch.ones(2, 40),
    )
    thermal = SENSOR_MODALITIES.index("thermal")
    assert torch.allclose(output["modality_weights"][1, thermal], torch.ones(40))
    output["logits"].sum().backward()
    assert all(head.weight.grad is not None for head in model.modality_heads.values())


def test_missing_modality_gate_is_safe_under_bfloat16_autocast():
    config = _tiny_config()
    model = TRHashSensorFusionClassifier(config)
    masks = {name: torch.tensor([True, False]) for name in SENSOR_MODALITIES}
    masks["depth"] = torch.tensor([True, True])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(_inputs(), modality_mask=masks)
    assert torch.isfinite(output["logits"]).all()
    assert torch.isfinite(output["modality_weights"]).all()


def test_hash_routes_every_class_modality_pair_through_fixed_experts():
    config = _tiny_config(num_classes=6)
    model = TRHashSensorFusionClassifier(config)
    assert model.class_modality_gate is not None
    gate = model.class_modality_gate
    assert gate.mlp.route_table.shape == (config.top_k, len(SENSOR_MODALITIES) * 6)
    assert torch.unique(gate.mlp.route_table).numel() == 8

    output = model(_inputs(), torch.tensor([1, 3]))
    assert output["modality_weights"].shape == (2, len(SENSOR_MODALITIES), 6)
    output["loss"].backward()
    assert gate.mlp.expert_gate.grad is not None
    assert gate.score.weight.grad is not None


def test_specializes_shared_visual_features_with_residual_adapters():
    model = TRHashSensorFusionClassifier(_tiny_config())
    assert set(model.visual_adapters) == {"depth", "ir", "thermal"}
    output = model(_inputs())
    output["loss"] = output["logits"].sum()
    output["loss"].backward()
    for adapter in model.visual_adapters.values():
        assert adapter.scale.grad is not None


def test_uses_structured_sensor_encoders_and_subject_adversary():
    model = TRHashSensorFusionClassifier(_tiny_config())
    from complexity.generative.sensor_fusion.structured_encoders import (
        IMUDeviceGraphTokenizer,
        SkeletonGraphTokenizer,
    )

    assert isinstance(model.encoders["imu"], IMUDeviceGraphTokenizer)
    assert isinstance(model.encoders["skeleton"], SkeletonGraphTokenizer)
    assert model.subject_head is not None

    output = model(_inputs(), torch.tensor([1, 3]), subject_adversarial_scale=0.5)
    assert output["subject_logits"].shape == (2, 18)
    (output["loss"] + output["subject_logits"].sum()).backward()
    assert model.subject_head[1].weight.grad is not None


def test_compact_profile_stays_under_ten_million_parameters():
    config = TRHashSensorFusionConfig(
        hidden_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        shared_width=96,
        expert_width=32,
        vision_hidden_size=64,
        vision_layers=2,
        vision_heads=4,
        vision_stage_depths=(1, 1),
        vision_shared_width=48,
        vision_expert_width=16,
        class_hash_shared_width=48,
        class_hash_expert_width=16,
    )
    model = TRHashSensorFusionClassifier(config)
    assert model.num_parameters() < 10_000_000


def test_loads_only_an_exact_compatible_pretrained_tower(tmp_path):
    config = _tiny_config()
    detector_config = TRHashDetectorConfig(
        image_size=config.vision_image_size,
        patch_size=config.vision_patch_size,
        vision_hidden_size=config.vision_hidden_size,
        vision_layers=config.vision_layers,
        vision_heads=config.vision_heads,
        vision_num_experts=config.vision_num_experts,
        vision_top_k=config.vision_top_k,
        vision_shared_width=config.vision_shared_width,
        vision_expert_width=config.vision_expert_width,
        vision_stage_depths=config.vision_stage_depths,
        vision_window_size=config.vision_window_size,
        vision_precision="fp32",
        num_classes=40,
        scale_factors=(1, 2),
    )
    source = HierarchicalTRHashVisionTower(detector_config)
    checkpoint = tmp_path / "vision"
    checkpoint.mkdir()
    save_file(
        {name: value.detach().contiguous() for name, value in source.state_dict().items()},
        checkpoint / "tower.safetensors",
    )
    # Task-aligned detector checkpoints keep the detector config at the root.
    (checkpoint / "config.json").write_text(json.dumps(detector_config.to_dict()))

    model = TRHashSensorFusionClassifier(config)
    transferred = load_pretrained_visual_tower(model, checkpoint)

    assert transferred > 0
    for name, value in source.state_dict().items():
        assert torch.equal(model.vision_tower.state_dict()[name], value)

    incompatible = tmp_path / "incompatible"
    incompatible.mkdir()
    save_file(
        {name: value.detach().contiguous() for name, value in source.state_dict().items()},
        incompatible / "tower.safetensors",
    )
    (incompatible / "config.json").write_text(
        json.dumps({**detector_config.to_dict(), "vision_hidden_size": 64})
    )
    try:
        load_pretrained_visual_tower(model, incompatible)
    except ValueError as error:
        assert "incompatible" in str(error)
    else:
        raise AssertionError("expected incompatible tower transfer to raise")


def test_rejects_invalid_sensor_shapes_and_empty_sample():
    model = TRHashSensorFusionClassifier(_tiny_config())
    inputs = _inputs()
    inputs["imu"] = torch.randn(2, 12, 5)
    try:
        model(inputs)
    except ValueError:
        pass
    else:
        raise AssertionError("expected malformed IMU input to raise")

    try:
        model({})
    except ValueError:
        pass
    else:
        raise AssertionError("expected an empty modality mapping to raise")
