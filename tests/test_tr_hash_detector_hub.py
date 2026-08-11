import json
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import save_file

from complexity.generative.detection import (
    VOC_CLASS_NAMES,
    TRHashDetectorConfig,
    TRHashObjectDetector,
    export_detector_for_hub,
    load_detector_checkpoint,
    preprocess_detector_image,
    restore_detector_boxes,
)


def _checkpoint(path: Path) -> Path:
    config = TRHashDetectorConfig(
        image_size=32,
        patch_size=8,
        vision_hidden_size=16,
        vision_layers=1,
        vision_heads=4,
        vision_num_experts=2,
        vision_top_k=1,
        vision_expert_width=8,
        vision_precision="fp32",
        num_classes=len(VOC_CLASS_NAMES),
    )
    model = TRHashObjectDetector(config)
    path.mkdir()
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        str(path / "model.safetensors"),
    )
    save_file(
        {name: value.detach().contiguous() for name, value in model.tower.state_dict().items()},
        str(path / "tower.safetensors"),
    )
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    (path / "validation.json").write_text(
        json.dumps(
            {
                "map50": 0.25,
                "precision": 0.3,
                "recall": 0.4,
                "best_f1": 0.34,
                "best_confidence": 0.22,
            }
        )
    )
    return path


def test_preprocess_and_restore_boxes_round_trip():
    image = Image.new("RGB", (80, 40), (200, 100, 50))
    pixels, metadata = preprocess_detector_image(image, 32)
    # Source box (20, 10, 60, 30) becomes normalized letterbox coordinates.
    letterboxed = torch.tensor([[0.25, 0.375, 0.75, 0.625]])

    restored = restore_detector_boxes(letterboxed, metadata)

    assert pixels.shape == (3, 32, 32)
    assert torch.allclose(restored, torch.tensor([[20.0, 10.0, 60.0, 30.0]]))


def test_export_and_strict_reload_hub_folder(tmp_path: Path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    output = export_detector_for_hub(
        tmp_path / "hub",
        "AETHORIA-AI/TR-HASH-Vision-Test",
        checkpoint=checkpoint,
    )

    assert (output / "README.md").exists()
    assert (output / "model.safetensors").exists()
    assert json.loads((output / "class_names.json").read_text()) == list(VOC_CLASS_NAMES)
    assert json.loads((output / "preprocessor_config.json").read_text())["letterbox"]
    assert "0.2500" in (output / "README.md").read_text()
    loaded = load_detector_checkpoint(output)
    assert loaded.config.num_classes == len(VOC_CLASS_NAMES)


def test_training_draft_does_not_publish_checkpoint_weights(tmp_path: Path):
    output = export_detector_for_hub(
        tmp_path / "draft",
        "AETHORIA-AI/TR-HASH-Vision-0.8M-VOC",
        training=True,
    )

    assert "Training in progress" in (output / "README.md").read_text()
    assert not (output / "model.safetensors").exists()
