import json
from pathlib import Path

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from complexity.generative.detection import (
    COCO_CLASS_NAMES,
    VOC_CLASS_NAMES,
    TRHashDetectorConfig,
    TRHashObjectDetector,
    export_detector_for_hub,
    load_detector_checkpoint,
    preprocess_detector_image,
    restore_detector_boxes,
)
from complexity.generative.detection.provenance import (
    NATIVE_COCO_DATASET,
    NATIVE_DETECTOR_IMPLEMENTATION,
    PROVENANCE_FORMAT_VERSION,
)


def _release_metrics(map50: float) -> dict[str, object]:
    return {
        "map50": map50,
        "map50_95": map50 / 2,
        "ap_small": map50 / 3,
        "ap_medium": map50 / 2,
        "ap_large": map50 * 0.75,
        "precision": 0.3,
        "recall": 0.4,
        "f1": 0.34,
        "best_f1": 0.35,
        "best_confidence": 0.22,
        "coco_map50": map50,
        "coco_map50_95": map50 / 2,
        "coco_ap_small": map50 / 3,
        "coco_ap_medium": map50 / 2,
        "coco_ap_large": map50 * 0.75,
        "coco_ar100": 0.5,
        "official_coco": True,
        "evaluator_backend": "pycocotools",
        "checkpoint_selection_metric": "coco_map50_95",
    }


def _checkpoint(path: Path) -> Path:
    config = TRHashDetectorConfig(
        image_size=32,
        patch_size=8,
        vision_hidden_size=16,
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
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


def _native_coco_checkpoint(path: Path) -> Path:
    config = TRHashDetectorConfig(
        image_size=32,
        patch_size=8,
        vision_hidden_size=16,
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_window_size=2,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=8,
        vision_precision="fp32",
        num_classes=len(COCO_CLASS_NAMES),
        end_to_end=True,
    )
    model = TRHashObjectDetector(config)
    path.mkdir(parents=True)
    save_file(
        {name: value.detach().contiguous() for name, value in model.state_dict().items()},
        str(path / "model.safetensors"),
    )
    (path / "config.json").write_text(json.dumps(config.to_dict()))
    (path / "validation.json").write_text(json.dumps(_release_metrics(0.4)))
    (path / "provenance.json").write_text(
        json.dumps(
            {
                "format_version": PROVENANCE_FORMAT_VERSION,
                "implementation": NATIVE_DETECTOR_IMPLEMENTATION,
                "initialization": "random",
                "external_checkpoint": None,
                "dataset": NATIVE_COCO_DATASET,
            }
        )
    )
    nms_free = path.parent / "best_nms_free"
    nms_free.mkdir()
    (nms_free / "validation.json").write_text(json.dumps(_release_metrics(0.3)))
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
    raw_state = load_file(str(checkpoint / "model.safetensors"))
    save_file(
        {name: torch.zeros_like(value) for name, value in raw_state.items()},
        str(checkpoint / "ema.safetensors"),
    )
    output = export_detector_for_hub(
        tmp_path / "hub",
        "AETHORIA-AI/TR-HASH-Vision-Test",
        checkpoint=checkpoint,
    )

    assert (output / "README.md").exists()
    assert (output / "model.safetensors").exists()
    assert (output / "ema.safetensors").exists()
    assert json.loads((output / "class_names.json").read_text()) == list(VOC_CLASS_NAMES)
    assert json.loads((output / "preprocessor_config.json").read_text())["letterbox"]
    assert "0.2500" in (output / "README.md").read_text()
    assert "Optimizer: MuSGD" in (output / "README.md").read_text()
    assert "ID-hash-routed" in (output / "README.md").read_text()
    loaded = load_detector_checkpoint(output)
    assert loaded.config.num_classes == len(VOC_CLASS_NAMES)
    assert all(torch.count_nonzero(parameter) == 0 for parameter in loaded.parameters())


def test_export_includes_sibling_nms_free_metrics(tmp_path: Path):
    (tmp_path / "run").mkdir()
    checkpoint = _checkpoint(tmp_path / "run" / "best")
    nms_free = checkpoint.parent / "best_nms_free"
    nms_free.mkdir()
    (nms_free / "validation.json").write_text(
        json.dumps(
            {
                "map50": 0.2,
                "precision": 0.25,
                "recall": 0.35,
                "best_f1": 0.29,
                "best_confidence": 0.2,
            }
        )
    )

    output = export_detector_for_hub(
        tmp_path / "hub",
        "AETHORIA-AI/TR-HASH-Vision-Test",
        checkpoint=checkpoint,
    )

    card = (output / "README.md").read_text()
    assert "O2M + NMS" in card
    assert "NMS-free" in card
    assert "0.2000" in card
    assert (output / "validation_nms_free.json").is_file()


def test_training_draft_does_not_publish_checkpoint_weights(tmp_path: Path):
    output = export_detector_for_hub(
        tmp_path / "draft",
        "AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO",
        class_names=COCO_CLASS_NAMES,
        training=True,
    )

    card = (output / "README.md").read_text()
    assert "Training in progress" in card
    assert "COCO 2017" in card
    assert "Random initialization" in card
    assert "no external detector or classification backbone" in card
    assert "Full-detector COCO 2017 training" in card
    assert "load_detector_from_hub" not in card
    assert json.loads((output / "class_names.json").read_text()) == list(COCO_CLASS_NAMES)
    assert not (output / "model.safetensors").exists()


def test_native_coco_release_requires_and_copies_complete_provenance(tmp_path: Path):
    checkpoint = _native_coco_checkpoint(tmp_path / "run" / "best")

    output = export_detector_for_hub(
        tmp_path / "hub",
        "AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO",
        checkpoint=checkpoint,
        class_names=COCO_CLASS_NAMES,
        dataset="coco",
        require_native_random_init=True,
    )

    assert json.loads((output / "provenance.json").read_text())["initialization"] == "random"
    assert (output / "validation.json").is_file()
    assert (output / "validation_nms_free.json").is_file()
    card = (output / "README.md").read_text()
    assert "value: 0.400000" in card
    assert "| NMS-free | 0.3000 |" in card


def test_native_coco_release_rejects_external_weight_provenance(tmp_path: Path):
    checkpoint = _native_coco_checkpoint(tmp_path / "run" / "best")
    provenance_path = checkpoint / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["initialization"] = "detector-transfer"
    provenance["external_checkpoint"] = "external/best"
    provenance_path.write_text(json.dumps(provenance))

    with pytest.raises(ValueError, match="random initialization"):
        export_detector_for_hub(
            tmp_path / "hub",
            "AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO",
            checkpoint=checkpoint,
            class_names=COCO_CLASS_NAMES,
            dataset="coco",
            require_native_random_init=True,
        )
    assert not (tmp_path / "hub").exists()


def test_training_draft_clears_stale_release_weights(tmp_path: Path):
    output = tmp_path / "draft"
    output.mkdir()
    (output / "model.safetensors").write_bytes(b"stale")
    (output / "validation.json").write_text("{}")

    export_detector_for_hub(
        output,
        "AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO",
        class_names=COCO_CLASS_NAMES,
        training=True,
        dataset="coco",
    )

    assert not (output / "model.safetensors").exists()
    assert not (output / "validation.json").exists()
