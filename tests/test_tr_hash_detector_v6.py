"""Architecture-v6 coverage for hierarchical and resolution-flexible detection."""

from __future__ import annotations

import torch
from PIL import Image
from safetensors.torch import save_file

from complexity.generative.detection import TRHashDetectorConfig, TRHashObjectDetector
from complexity.generative.detection.data import YoloDetectionDataset
from complexity.generative.detection.ema import ModelEMA
from complexity.generative.detection.training import load_pretrained_tower


def _config(image_size: int = 32, *, end_to_end: bool = False):
    return TRHashDetectorConfig(
        architecture_version=6,
        image_size=image_size,
        patch_size=8,
        vision_hidden_size=16,
        vision_layers=3,
        vision_stage_depths=(1, 1, 1),
        vision_heads=4,
        vision_expert_width=4,
        vision_window_size=2,
        num_classes=2,
        p2_head=True,
        end_to_end=end_to_end,
    )


def test_v6_uses_real_hierarchical_maps_and_dynamic_geometry():
    model = TRHashObjectDetector(_config())
    full_features = model.tower(torch.randn(2, 3, 32, 32))
    assert [feature.shape[-2:] for feature in full_features] == [(4, 4), (2, 2), (1, 1)]

    raw = model(torch.randn(2, 3, 24, 24))
    assert model._grid_sizes_for_raw(raw) == (6, 3, 2, 1)
    targets = [torch.tensor([[0.5, 0.5, 0.25, 0.25, 1.0]]) for _ in range(2)]
    losses = model.compute_loss(raw, targets)
    losses["loss"].backward()
    assert torch.isfinite(losses["loss"])


def test_v6_one_to_one_branch_trains_and_decodes_without_nms():
    model = TRHashObjectDetector(_config(end_to_end=True))
    pixels = torch.randn(2, 3, 24, 24)
    branches = model(pixels, return_branches=True)
    assert branches[0].shape == branches[1].shape
    assert torch.equal(branches[0], branches[1])
    targets = [torch.tensor([[0.5, 0.5, 0.25, 0.25, 1.0]]) for _ in range(2)]
    losses = model.compute_loss(branches, targets, training_progress=0.0)
    assert "one_to_one_loss" in losses
    assert "one_to_one_monitor_loss" in losses
    assert losses["one_to_one_weight"].item() == 0.25
    assert torch.allclose(
        losses["monitor_loss"],
        0.5
        * (
            losses["one_to_many_monitor_loss"]
            + losses["one_to_one_monitor_loss"]
        ),
    )
    losses["loss"].backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad)
        for parameter in model.tower.parameters()
    )

    results = model.predict_end_to_end(
        pixels,
        confidence_threshold=0.0,
        max_detections=5,
    )
    assert all(len(result["boxes"]) == 5 for result in results)


def test_v6_one_to_one_assignment_keeps_unique_targets_and_uses_p2_for_small_objects():
    model = TRHashObjectDetector(_config(end_to_end=True))
    pixels = torch.randn(1, 3, 32, 32)
    one_to_many, _ = model.forward_branches(pixels)
    decoded = model.decode(one_to_many)
    targets = [
        torch.tensor(
            [
                [0.50, 0.50, 0.60, 0.60, 0.0],
                [0.50, 0.50, 0.60, 0.60, 1.0],
                [0.15, 0.15, 0.05, 0.05, 0.0],
            ]
        )
    ]

    assigned = model._assign_targets(
        targets,
        pixels.device,
        decoded=decoded,
        assignment_top_k=1,
        allow_stal=True,
        unique_per_target=True,
    )

    positive_targets = assigned["target_indices"][0][assigned["positive_mask"][0]]
    assert set(positive_targets.tolist()) == {0, 1, 2}
    assert len(positive_targets) == len(torch.unique(positive_targets))
    small_cell = torch.nonzero(assigned["target_indices"][0] == 2).item()
    assert small_cell < decoded["grid_sizes"][0] ** 2


def test_v6_position_grids_transfer_across_resolutions(tmp_path):
    source = TRHashObjectDetector(_config(32))
    checkpoint = tmp_path / "tower"
    checkpoint.mkdir()
    save_file(
        {
            name: value.detach().contiguous()
            for name, value in source.tower.state_dict().items()
        },
        str(checkpoint / "tower.safetensors"),
    )
    target = TRHashObjectDetector(_config(64))

    load_pretrained_tower(target, checkpoint)

    assert target.tower.position_rows[0].shape[-1] == 8
    assert target.tower.position_cols[0].shape[-1] == 8
    assert torch.equal(target.tower.patch_embed.weight, source.tower.patch_embed.weight)


def test_model_ema_updates_without_changing_training_model():
    model = TRHashObjectDetector(_config())
    ema = ModelEMA(model, decay=0.9999)
    name, parameter = next(iter(model.named_parameters()))
    before = ema.module.state_dict()[name].clone()
    with torch.no_grad():
        parameter.add_(1.0)

    ema.update(model)

    assert ema.updates == 1
    assert not torch.equal(ema.module.state_dict()[name], before)
    assert torch.equal(parameter, model.state_dict()[name])


def test_mosaic_is_deterministic_and_closes_for_final_epochs(tmp_path):
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    for index in range(4):
        Image.new("RGB", (24, 16), (index * 30, 50, 100)).save(images / f"{index}.jpg")
        (labels / f"{index}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    dataset = YoloDetectionDataset(
        images,
        labels,
        image_size=32,
        augmentation="strong",
        seed=7,
        mosaic_probability=1.0,
        total_epochs=10,
        close_mosaic_epochs=2,
    )

    first_pixels, first_targets = dataset[0]
    second_pixels, second_targets = dataset[0]
    assert torch.equal(first_pixels, second_pixels)
    assert torch.equal(first_targets, second_targets)
    assert len(first_targets) == 4
    assert torch.all((first_targets[:, :4] >= 0.0) & (first_targets[:, :4] <= 1.0))

    dataset.set_epoch(8)
    _, final_targets = dataset[0]
    assert len(final_targets) == 1
