import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from complexity.generative.sensor_fusion import (
    CUHKXPreprocessingConfig,
    TRHashSensorFusionClassifier,
    TRHashSensorFusionConfig,
)
from complexity.generative.sensor_fusion.checkpointing import (
    find_latest_resumable_checkpoint,
    load_sensor_fusion_checkpoint,
    save_sensor_fusion_checkpoint,
)
from complexity.training.musgd import MuSGD
from tests.test_cuhkx_sensor_data import _write_complete_trial


def _config() -> TRHashSensorFusionConfig:
    return TRHashSensorFusionConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_experts=8,
        top_k=2,
        shared_width=16,
        expert_width=4,
        precision="fp32",
        visual_token_grid=(1, 1, 1),
        vision_image_size=16,
        vision_patch_size=4,
        vision_hidden_size=16,
        vision_layers=1,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_expert_width=4,
        vision_stage_depths=(1,),
        vision_window_size=2,
        radar_features=5,
        sequence_tokens=2,
    )


def test_training_cli_accepts_preprocessing_version(monkeypatch):
    from complexity.generative.sensor_fusion import training

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cf-sensor-fusion-train",
            "--data-root",
            "/tmp/data",
            "--output",
            "/tmp/output",
            "--optimizer",
            "musgd",
            "--preprocessing-version",
            "3",
        ],
    )
    args = training.parse_args()
    assert args.preprocessing_version == 3
    assert not hasattr(args, "architecture_version")


def test_backbone_lr_defaults_to_full_unless_explicitly_overridden():
    from complexity.generative.sensor_fusion.training import (
        resolve_backbone_lr_multiplier,
    )

    assert resolve_backbone_lr_multiplier(None) == 1.0
    assert resolve_backbone_lr_multiplier(0.25) == 0.25
    with pytest.raises(ValueError, match="must be positive"):
        resolve_backbone_lr_multiplier(0.0)


def test_augmentation_scale_is_full_strength_when_clean_finetune_disabled():
    from complexity.generative.sensor_fusion.training import augmentation_scale

    for epoch in (0, 5, 49):
        assert augmentation_scale(epoch, total_epochs=50, clean_finetune_epochs=0) == 1.0


def test_augmentation_scale_anneals_to_zero_over_the_final_clean_epochs():
    """Regression guard: the sensor-fusion trainer's inspired-by-Vision-v8
    "noisy pretrain -> clean full-parameter SFT" recipe anneals batch-level
    augmentation (mixup/jitter/noise/modality-dropout) to zero over the final
    N epochs instead of keeping constant augmentation strength for the whole
    run."""
    from complexity.generative.sensor_fusion.training import augmentation_scale

    # 50 total epochs, last 10 are the clean fine-tune window (epochs 40..49).
    assert augmentation_scale(0, total_epochs=50, clean_finetune_epochs=10) == 1.0
    assert augmentation_scale(39, total_epochs=50, clean_finetune_epochs=10) == 1.0
    assert augmentation_scale(40, total_epochs=50, clean_finetune_epochs=10) == pytest.approx(1.0)
    assert augmentation_scale(45, total_epochs=50, clean_finetune_epochs=10) == pytest.approx(0.5)
    assert augmentation_scale(49, total_epochs=50, clean_finetune_epochs=10) == pytest.approx(0.1)


def test_augmentation_scale_never_goes_negative_past_the_final_epoch():
    from complexity.generative.sensor_fusion.training import augmentation_scale

    assert augmentation_scale(59, total_epochs=50, clean_finetune_epochs=10) == 0.0


def test_clean_finetune_anneal_start_mirrors_augmentation_scale_boundary():
    from complexity.generative.sensor_fusion.training import clean_finetune_anneal_start

    assert clean_finetune_anneal_start(total_epochs=50, clean_finetune_epochs=0) == 50
    assert clean_finetune_anneal_start(total_epochs=50, clean_finetune_epochs=10) == 40


def test_resolve_epoch_training_augmentation_is_always_off_when_disabled():
    from complexity.generative.sensor_fusion.training import resolve_epoch_training_augmentation

    for epoch in range(10):
        assert resolve_epoch_training_augmentation(
            base_enabled=False, aug_scale=1.0, seed=42, epoch=epoch
        ) is False


def test_resolve_epoch_training_augmentation_is_deterministic_and_bounded():
    """Regression guard: at aug_scale=1.0 (full noisy-pretrain strength) the
    epoch must always keep dataset-side augmentation on; at aug_scale=0.0
    (fully annealed) it must always be off; same (seed, epoch) must always
    give the same answer so a resumed run replays identically."""
    from complexity.generative.sensor_fusion.training import resolve_epoch_training_augmentation

    for epoch in range(10):
        assert resolve_epoch_training_augmentation(
            base_enabled=True, aug_scale=1.0, seed=42, epoch=epoch
        ) is True
        assert resolve_epoch_training_augmentation(
            base_enabled=True, aug_scale=0.0, seed=42, epoch=epoch
        ) is False

    first = resolve_epoch_training_augmentation(base_enabled=True, aug_scale=0.5, seed=7, epoch=3)
    second = resolve_epoch_training_augmentation(base_enabled=True, aug_scale=0.5, seed=7, epoch=3)
    assert first == second


def test_clean_finetune_train_loader_only_built_when_actually_needed():
    """Regression guard: persistent DataLoader workers pickle the Dataset
    once and never see the clean-finetune anneal's per-epoch mutations of
    visual_horizontal_flip/visual_crop_jitter/training_augmentation. A
    dedicated non-persistent loader must exist to serve exactly the epochs
    inside the anneal window -- but building it (and paying its per-epoch
    worker respawn cost) is wasted when there are no workers to go stale, or
    no anneal window that needs it."""
    from complexity.generative.sensor_fusion.training import (
        should_build_clean_finetune_train_loader,
    )

    assert should_build_clean_finetune_train_loader(workers=4, clean_finetune_epochs=10) is True
    assert should_build_clean_finetune_train_loader(workers=0, clean_finetune_epochs=10) is False
    assert should_build_clean_finetune_train_loader(workers=4, clean_finetune_epochs=0) is False


def test_clean_finetune_epochs_cannot_exceed_total_epochs(monkeypatch, tmp_path):
    from complexity.generative.sensor_fusion import training

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cf-sensor-fusion-train",
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "out"),
            "--optimizer",
            "musgd",
            "--epochs",
            "5",
            "--clean-finetune-epochs",
            "10",
        ],
    )
    with pytest.raises(ValueError, match="cannot exceed --epochs"):
        training.main()


def test_checkpoint_restores_weights_optimizer_scheduler_cursor_and_rng(tmp_path):
    torch.manual_seed(17)
    config = _config()
    preprocessing = CUHKXPreprocessingConfig()
    model = TRHashSensorFusionClassifier(config)
    optimizer = MuSGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    inputs = {
        "imu": torch.randn(2, 4, 45),
        "radar": torch.randn(2, 4, 5),
        "skeleton": torch.randn(2, 4, 17, 3),
    }
    loss = model(inputs, torch.tensor([1, 2]))["loss"]
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    options = {"batch_size": 2, "seed": 17}
    checkpoint = tmp_path / "step_1"
    save_sensor_fusion_checkpoint(
        checkpoint,
        model,
        optimizer,
        scheduler,
        model_config=config,
        preprocessing=preprocessing,
        epoch=0,
        batch_in_epoch=1,
        step=1,
        best_accuracy=0.25,
        total_epochs=3,
        steps_per_epoch=4,
        training_options=options,
        metrics={"top1_accuracy": 0.25},
    )

    torch.manual_seed(999)
    restored = TRHashSensorFusionClassifier(config)
    restored_optimizer = MuSGD(restored.parameters(), lr=1e-3, momentum=0.9)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda step: 1.0)
    state = load_sensor_fusion_checkpoint(
        checkpoint,
        restored,
        restored_optimizer,
        restored_scheduler,
        model_config=config,
        preprocessing=preprocessing,
        total_epochs=3,
        steps_per_epoch=4,
        training_options=options,
    )
    assert state["epoch"] == 0
    assert state["batch_in_epoch"] == 1
    assert state["step"] == 1
    assert state["best_accuracy"] == 0.25
    assert json.loads((checkpoint / "metrics.json").read_text())["top1_accuracy"] == 0.25
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])
    assert restored_scheduler.state_dict() == scheduler.state_dict()


def test_training_cli_requires_musgd(monkeypatch, tmp_path):
    from complexity.generative.sensor_fusion import training

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cf-sensor-fusion-train",
            "--data-root",
            str(tmp_path),
            "--output",
            str(tmp_path / "output"),
        ],
    )
    try:
        training.parse_args()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("--optimizer musgd must be explicit")


def test_inverse_sqrt_class_weights_ignore_absent_classes():
    from complexity.generative.sensor_fusion.training import build_class_weights

    records = [SimpleNamespace(action_id=0) for _ in range(4)] + [SimpleNamespace(action_id=1)]
    weights, counts = build_class_weights(
        records,
        num_classes=3,
        mode="inverse-sqrt",
        device=torch.device("cpu"),
    )
    assert counts.tolist() == [4, 1, 0]
    assert weights[1] == 2 * weights[0]
    assert weights[2] == 0


def test_modality_dropout_never_removes_every_available_stream():
    from complexity.generative.sensor_fusion.training import apply_modality_dropout

    torch.manual_seed(3)
    masks = {
        "depth": torch.tensor([True, False, True]),
        "imu": torch.tensor([True, True, False]),
        "radar": torch.tensor([False, False, True]),
    }
    dropped = apply_modality_dropout(masks, 0.95)
    stacked = torch.stack(list(dropped.values()), dim=1)
    assert stacked.any(dim=1).all()
    for name in masks:
        assert not (dropped[name] & ~masks[name]).any()


def test_sensor_augmentation_and_mixup_preserve_shapes_and_masks():
    from complexity.generative.sensor_fusion.training import (
        augment_sensor_inputs,
        mixup_sensor_batch,
    )

    torch.manual_seed(5)
    inputs = {
        "depth": torch.zeros(4, 3, 2, 8, 8),
        "imu": torch.zeros(4, 6, 45),
    }
    masks = {
        "depth": torch.tensor([True, False, True, True]),
        "imu": torch.tensor([True, True, False, True]),
    }
    augmented = augment_sensor_inputs(
        inputs,
        visual_jitter=0.1,
        sensor_noise=0.01,
    )
    assert augmented["depth"].shape == inputs["depth"].shape
    assert augmented["imu"].shape == inputs["imu"].shape
    assert torch.count_nonzero(augmented["imu"]) > 0

    mixed, mixed_masks, mixed_labels, weight, permutation = mixup_sensor_batch(
        augmented,
        masks,
        torch.arange(4),
        0.2,
    )
    assert 0.0 < weight < 1.0
    assert mixed_labels.shape == (4,)
    assert sorted(permutation.tolist()) == [0, 1, 2, 3]
    assert all(mixed[name].shape == inputs[name].shape for name in inputs)
    assert all((mixed_masks[name] | ~masks[name]).shape == masks[name].shape for name in masks)


def test_epoch_annotated_sampler_replays_the_same_indices_and_epoch():
    from complexity.generative.sensor_fusion.sampling import (
        EpochAnnotatedSampler,
        EpochRandomSampler,
    )

    dataset = list(range(9))
    sampler = EpochAnnotatedSampler(EpochRandomSampler(dataset, seed=5))
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert list(sampler) == first
    assert all(epoch == 3 for _, epoch in first)
    sampler.set_epoch(4)
    assert [index for index, _ in sampler] != [index for index, _ in first]


def test_weighted_sampler_is_deterministic_and_disjoint_across_ranks():
    from complexity.generative.sensor_fusion.sampling import EpochWeightedSampler

    weights = torch.tensor([1.0, 1.0, 8.0, 1.0, 1.0])
    rank0 = EpochWeightedSampler(weights, seed=9, rank=0, world_size=2)
    rank1 = EpochWeightedSampler(weights, seed=9, rank=1, world_size=2)
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    first0, first1 = list(rank0), list(rank1)
    assert len(first0) == len(first1) == 3
    rank0.set_epoch(3)
    rank1.set_epoch(3)
    assert list(rank0) == first0
    assert list(rank1) == first1
    generator = torch.Generator().manual_seed(12)
    global_draw = torch.multinomial(weights.double(), 6, True, generator=generator)
    assert first0 == global_draw[0::2].tolist()
    assert first1 == global_draw[1::2].tolist()


def test_sampling_and_auxiliary_losses_cover_rare_available_modalities():
    from complexity.generative.sensor_fusion import SENSOR_MODALITIES
    from complexity.generative.sensor_fusion.training import (
        auxiliary_modality_loss,
        build_sampling_weights,
    )

    records = [SimpleNamespace(action_id=0) for _ in range(4)] + [SimpleNamespace(action_id=1)]
    weights = build_sampling_weights(records, num_classes=3, mode="inverse-sqrt")
    assert weights[-1] == 2 * weights[0]

    logits = torch.randn(2, len(SENSOR_MODALITIES), 3, requires_grad=True)
    masks = {
        name: torch.tensor([True, index % 2 == 0]) for index, name in enumerate(SENSOR_MODALITIES)
    }
    loss = auxiliary_modality_loss(
        logits,
        masks,
        torch.tensor([0, 1]),
        class_weights=None,
        label_smoothing=0.0,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_two_dimensional_sampling_downweights_repeated_action_subject_pairs():
    from complexity.generative.sensor_fusion.training import build_sample_loss_weights

    records = [
        SimpleNamespace(action_id=0, user_id=1),
        SimpleNamespace(action_id=0, user_id=1),
        SimpleNamespace(action_id=0, user_id=1),
        SimpleNamespace(action_id=0, user_id=1),
        SimpleNamespace(action_id=0, user_id=2),
        SimpleNamespace(action_id=1, user_id=1),
    ]

    weights = build_sample_loss_weights(
        records,
        num_classes=2,
        mode="inverse-sqrt-2d",
    )

    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] == weights[3]
    assert weights[4] == pytest.approx(2 * weights[0])
    assert weights[5] > weights[4]


def test_three_dimensional_weights_balance_sensor_completeness_with_hard_bounds():
    from complexity.generative.sensor_fusion import SENSOR_MODALITIES
    from complexity.generative.sensor_fusion.training import build_sample_loss_weights

    complete = {name: object() for name in SENSOR_MODALITIES}
    incomplete = {"thermal": object()}
    records = [
        SimpleNamespace(action_id=0, user_id=1, paths=complete),
        SimpleNamespace(action_id=0, user_id=1, paths=complete),
        SimpleNamespace(action_id=0, user_id=1, paths=complete),
        SimpleNamespace(action_id=0, user_id=1, paths=incomplete),
        SimpleNamespace(action_id=1, user_id=2, paths=complete),
    ]

    weights = build_sample_loss_weights(
        records,
        num_classes=2,
        mode="inverse-sqrt-3d",
        minimum=0.5,
        maximum=2.0,
    )

    assert weights.mean() == pytest.approx(1.0)
    assert weights.min() >= 0.5
    assert weights.max() <= 2.0
    assert weights[3] > weights[0]


def test_weighted_cross_entropy_preserves_full_shard_rows():
    from complexity.generative.sensor_fusion.training import weighted_cross_entropy

    logits = torch.tensor([[3.0, -1.0], [-1.0, 3.0], [0.5, 0.5]])
    labels = torch.tensor([0, 1, 0])
    weights = torch.tensor([0.5, 1.0, 2.0])
    per_row = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    loss = weighted_cross_entropy(
        logits,
        labels,
        weights,
        class_weights=None,
        label_smoothing=0.0,
    )

    assert loss == pytest.approx(float((per_row * weights).sum() / weights.sum()))


def test_gate_calibration_prefers_the_more_accurate_available_modality():
    from complexity.generative.sensor_fusion import SENSOR_MODALITIES
    from complexity.generative.sensor_fusion.training import (
        modality_gate_calibration_loss,
    )

    logits = torch.zeros(1, len(SENSOR_MODALITIES), 3)
    depth = SENSOR_MODALITIES.index("depth")
    thermal = SENSOR_MODALITIES.index("thermal")
    logits[0, depth] = torch.tensor([8.0, -4.0, -4.0])
    logits[0, thermal] = torch.tensor([-4.0, 8.0, -4.0])
    masks = {name: torch.tensor([name in {"depth", "thermal"}]) for name in SENSOR_MODALITIES}
    balanced_scores = torch.zeros(1, len(SENSOR_MODALITIES), requires_grad=True)
    good_scores = torch.zeros(1, len(SENSOR_MODALITIES))
    good_scores[0, depth] = 3.0
    bad_scores = torch.zeros(1, len(SENSOR_MODALITIES))
    bad_scores[0, thermal] = 3.0

    def weights(scores):
        available = torch.stack(tuple(masks[name] for name in SENSOR_MODALITIES), dim=1)
        return scores.masked_fill(~available, -torch.inf).softmax(dim=1)

    kwargs = {
        "quality_temperature": 1.0,
        "target_smoothing": 0.1,
        "label_smoothing": 0.0,
    }
    balanced = modality_gate_calibration_loss(
        logits, weights(balanced_scores), masks, torch.tensor([0]), **kwargs
    )
    good = modality_gate_calibration_loss(
        logits, weights(good_scores), masks, torch.tensor([0]), **kwargs
    )
    bad = modality_gate_calibration_loss(
        logits, weights(bad_scores), masks, torch.tensor([0]), **kwargs
    )

    assert good < balanced < bad
    balanced.backward()
    assert balanced_scores.grad is not None
    assert balanced_scores.grad[0, depth] < 0
    assert balanced_scores.grad[0, thermal] > 0


def test_gate_calibration_ignores_unavailable_modalities_and_singletons():
    from complexity.generative.sensor_fusion import SENSOR_MODALITIES
    from complexity.generative.sensor_fusion.training import (
        modality_gate_calibration_loss,
    )

    logits = torch.randn(2, len(SENSOR_MODALITIES), 4, requires_grad=True)
    thermal = SENSOR_MODALITIES.index("thermal")
    masks = {
        name: torch.tensor([name == "thermal", name in {"depth", "skeleton"}])
        for name in SENSOR_MODALITIES
    }
    available = torch.stack(tuple(masks[name] for name in SENSOR_MODALITIES), dim=1)
    scores = torch.randn(2, len(SENSOR_MODALITIES), requires_grad=True)
    weights = scores.masked_fill(~available, -torch.inf).softmax(dim=1)
    loss = modality_gate_calibration_loss(
        logits,
        weights,
        masks,
        torch.tensor([1, 2]),
        torch.tensor([0.5, 2.0]),
        quality_temperature=0.7,
        target_smoothing=0.2,
        label_smoothing=0.1,
    )

    assert torch.isfinite(loss)
    assert weights[0, thermal] == 1
    loss.backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_gate_calibration_selects_the_target_class_from_two_dimensional_gates():
    from complexity.generative.sensor_fusion import SENSOR_MODALITIES
    from complexity.generative.sensor_fusion.training import (
        modality_gate_calibration_loss,
    )

    num_modalities = len(SENSOR_MODALITIES)
    logits = torch.zeros(1, num_modalities, 3)
    depth = SENSOR_MODALITIES.index("depth")
    skeleton = SENSOR_MODALITIES.index("skeleton")
    logits[0, depth, 2] = 8.0
    logits[0, skeleton, 0] = 8.0
    masks = {name: torch.tensor([name in {"depth", "skeleton"}]) for name in SENSOR_MODALITIES}
    available = torch.stack(tuple(masks[name] for name in SENSOR_MODALITIES), dim=1)
    scores = torch.zeros(1, num_modalities, 3, requires_grad=True)
    weights = scores.masked_fill(~available.unsqueeze(-1), -torch.inf).softmax(dim=1)

    loss = modality_gate_calibration_loss(
        logits,
        weights,
        masks,
        torch.tensor([2]),
        quality_temperature=1.0,
        target_smoothing=0.1,
        label_smoothing=0.0,
    )
    loss.backward()

    assert scores.grad is not None
    assert scores.grad[0, depth, 2] < 0
    assert scores.grad[0, skeleton, 2] > 0
    assert torch.all(scores.grad[0, :, :2] == 0)


def test_supervised_contrastive_loss_separates_actions_and_backpropagates():
    from complexity.generative.sensor_fusion.training import (
        supervised_contrastive_loss,
    )

    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1])

    loss = supervised_contrastive_loss(
        features,
        labels,
        temperature=0.1,
    )
    loss.backward()

    assert 0.0 <= loss.item() < 0.1
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_supervised_contrastive_loss_handles_batches_without_positive_pairs():
    from complexity.generative.sensor_fusion.training import (
        supervised_contrastive_loss,
    )

    features = torch.randn(3, 8, requires_grad=True)
    loss = supervised_contrastive_loss(
        features,
        torch.tensor([0, 1, 2]),
        temperature=0.2,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert features.grad is not None


def test_distributed_contrastive_zero_pair_rank_keeps_collective_in_graph(
    monkeypatch,
):
    from complexity.generative.detection.distributed import DistributedContext
    from complexity.generative.sensor_fusion import training

    features = torch.randn(2, 8, requires_grad=True)
    remote_features = torch.randn(2, 8, requires_grad=True)
    labels = torch.tensor([0, 1])

    monkeypatch.setattr(
        training.dist_nn,
        "all_gather",
        lambda anchors: (anchors, remote_features),
    )

    def gather_labels(outputs, local_labels):
        outputs[0].copy_(local_labels)
        outputs[1].copy_(torch.tensor([2, 3]))

    monkeypatch.setattr(training.dist, "all_gather", gather_labels)
    context = DistributedContext(device=torch.device("cpu"), world_size=2)

    loss = training.supervised_contrastive_loss(
        features,
        labels,
        temperature=0.2,
        context=context,
    )
    loss.backward()

    assert loss.item() == 0.0
    assert remote_features.grad is not None


def test_evaluation_reports_per_modality_accuracy_and_gate_weights():
    from complexity.generative.detection.distributed import DistributedContext
    from complexity.generative.sensor_fusion import SENSOR_MODALITIES
    from complexity.generative.sensor_fusion.training import evaluate

    config = _config()
    model = TRHashSensorFusionClassifier(config)
    batch = {
        "inputs": {
            "depth": torch.randn(2, 3, 2, 16, 16),
            "ir": torch.randn(2, 1, 2, 16, 16),
            "thermal": torch.randn(2, 3, 2, 16, 16),
            "imu": torch.randn(2, 4, 45),
            "radar": torch.randn(2, 4, 5),
            "skeleton": torch.randn(2, 4, 17, 3),
        },
        "modality_mask": {
            name: torch.tensor([True, name == "thermal"]) for name in SENSOR_MODALITIES
        },
        "labels": torch.tensor([0, 1]),
    }
    metrics = evaluate(
        model,
        [batch],
        torch.device("cpu"),
        "fp32",
        DistributedContext(device=torch.device("cpu")),
    )
    assert len(metrics["modality_accuracy"]) == len(SENSOR_MODALITIES)
    assert len(metrics["modality_gate_weights"]) == len(SENSOR_MODALITIES)
    assert abs(sum(metrics["modality_gate_weights"]) - 1.0) < 1e-6


def test_training_smoke_checkpoint_and_exact_resume(monkeypatch, tmp_path):
    from complexity.generative.sensor_fusion import training

    _write_complete_trial(tmp_path, user=1)
    _write_complete_trial(tmp_path, user=8)
    output = tmp_path / "run"
    common = [
        "cf-sensor-fusion-train",
        "--data-root",
        str(tmp_path),
        "--output",
        str(output),
        "--optimizer",
        "musgd",
        "--epochs",
        "2",
        "--batch-size",
        "1",
        "--eval-batch-size",
        "1",
        "--workers",
        "0",
        "--validation-users",
        "8",
        "--image-size",
        "16",
        "--clip-frames",
        "4",
        "--sensor-steps",
        "8",
        "--hidden-size",
        "16",
        "--layers",
        "1",
        "--heads",
        "4",
        "--shared-width",
        "16",
        "--expert-width",
        "4",
        "--sequence-tokens",
        "2",
        "--precision",
        "fp32",
        "--warmup-steps",
        "1",
        "--save-steps",
        "1",
        "--log-steps",
        "1",
        "--no-drop-last",
    ]
    monkeypatch.setattr(sys, "argv", common + ["--smoke-steps", "1"])
    training.main()
    first = torch.load(output / "smoke_final" / "training_state.pt", weights_only=True)
    assert (first["epoch"], first["batch_in_epoch"], first["step"]) == (1, 0, 1)
    assert not (output / "training_complete.json").exists()

    monkeypatch.setattr(
        sys,
        "argv",
        common
        + [
            "--resume",
            str(output / "smoke_final"),
            "--smoke-steps",
            "2",
        ],
    )
    training.main()
    resumed = torch.load(output / "smoke_final" / "training_state.pt", weights_only=True)
    assert (resumed["epoch"], resumed["batch_in_epoch"], resumed["step"]) == (2, 0, 2)
    assert (output / "best" / "model.safetensors").is_file()
    assert (output / "best_macro" / "model.safetensors").is_file()
    assert (output / "best_composite" / "model.safetensors").is_file()
    assert len((output / "metrics.jsonl").read_text().splitlines()) == 2
    completion = json.loads((output / "training_complete.json").read_text())
    assert completion["completed"] is True
    assert completion["epochs"] == 2
    assert completion["step"] == 2
    assert completion["validation_users"] == [8]


def test_find_latest_resumable_checkpoint_uses_training_step(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    for name, step in (("best", 10), ("step_0000020", 20), ("broken", 99)):
        checkpoint = output / name
        checkpoint.mkdir()
        if name == "broken":
            (checkpoint / "training_state.pt").write_bytes(b"not a checkpoint")
            continue
        torch.save(
            {
                "format_version": 1,
                "step": step,
                "epoch": step // 10,
                "batch_in_epoch": 0,
            },
            checkpoint / "training_state.pt",
        )
    assert find_latest_resumable_checkpoint(output) == output / "step_0000020"
    assert find_latest_resumable_checkpoint(tmp_path / "missing") is None
