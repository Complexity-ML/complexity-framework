"""Public API contract for the TR-Hash Robotics model family."""

import torch

from complexity.api import Robot
from complexity.generative.sensor_fusion import (
    SENSOR_MODALITIES,
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)


def _tiny_robot_config() -> TRHashSensorFusionConfig:
    return Robot.config(
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
        vision_expert_width=8,
        vision_stage_depths=(1, 1),
        vision_window_size=2,
        radar_features=5,
        sequence_tokens=4,
    )


def test_robot_factory_is_public_and_uses_the_sensor_fusion_engine() -> None:
    config = _tiny_robot_config()
    model = Robot.from_config(config)
    assert isinstance(model, TRHashSensorFusionClassifier)
    assert model.config == config
    assert Robot.MODALITIES == SENSOR_MODALITIES


def test_robot_model_preserves_shared_plus_sparse_tr_hash_contract() -> None:
    model = Robot.from_config(_tiny_robot_config())
    for block in model.blocks:
        assert block.mlp.config.shared_width > 0
        assert block.mlp.config.expert_width > 0
        assert block.mlp.config.top_k < block.mlp.config.num_experts


def test_robot_factory_accepts_multimodal_inputs() -> None:
    model = Robot.from_config(_tiny_robot_config())
    batch = 2
    output = model(
        {
            "depth": torch.randn(batch, 3, 4, 16, 16),
            "ir": torch.randn(batch, 1, 4, 16, 16),
            "thermal": torch.randn(batch, 3, 4, 16, 16),
            "imu": torch.randn(batch, 12, 45),
            "radar": torch.randn(batch, 10, 5),
            "skeleton": torch.randn(batch, 8, 17, 3),
        }
    )
    assert output["logits"].shape == (batch, model.config.num_classes)
