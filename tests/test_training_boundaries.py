"""Numerical regressions for accumulation, data cursors, and WSD resume."""

import copy
import math
from pathlib import Path

import pytest
import torch

from complexity.training import Trainer, TrainingConfig


class UnsizedBatches:
    def __init__(self, values):
        self.values = values
        self.consumed = 0

    def __iter__(self):
        for value in self.values:
            self.consumed += 1
            yield {"target": torch.tensor([[float(value)]])}


def make_trainer(root, loader, *, accumulation=4, max_steps=4, max_epochs=None):
    model = torch.nn.Linear(1, 1, bias=False)
    torch.nn.init.zeros_(model.weight)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    config = TrainingConfig(
        precision="fp32", use_fsdp=False, grad_clip=0.0,
        skip_param_update_check=True, learning_rate=0.1,
        gradient_accumulation_steps=accumulation, max_steps=max_steps,
        max_epochs=max_epochs, checkpoint_dir=str(root / "checkpoints"),
        log_dir=str(root / "logs"), save_steps=1000,
    )

    def loss(wrapped, batch):
        return (wrapped(torch.ones(1, 1)) - batch["target"]).square().mean()

    return Trainer(model, config, loader, optimizer=optimizer,
                   scheduler=scheduler, compute_loss=loss)


@pytest.mark.parametrize("size", [1, 3, 4, 5, 8])
@pytest.mark.parametrize("accumulation", [1, 4])
def test_updates_match_mean_loss_reference_across_epochs(tmp_path, size, accumulation):
    values = list(range(1, size + 1))
    updates_per_epoch = math.ceil(size / accumulation)
    trainer = make_trainer(tmp_path, UnsizedBatches(values), accumulation=accumulation,
                           max_steps=2 * updates_per_epoch)
    reference = copy.deepcopy(trainer.model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    expected_losses = []
    for _ in range(2):
        for start in range(0, size, accumulation):
            targets = torch.tensor(values[start:start + accumulation], dtype=torch.float32)
            loss = (reference(torch.ones(1, 1)).reshape(()) - targets).square().mean()
            expected_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    trainer.train()

    assert trainer.global_step == 2 * updates_per_epoch
    assert trainer.epoch == 2
    torch.testing.assert_close(trainer.model.weight, reference.weight)
    actual_losses = [row["loss"] for row in trainer.metrics.history]
    assert actual_losses == pytest.approx(expected_losses)
    assert all(param.grad is None for param in trainer.model.parameters())


def test_checkpoint_boundary_does_not_consume_next_update(tmp_path):
    loader = UnsizedBatches([1, 2, 3, 4, 5])
    trainer = make_trainer(tmp_path, loader, max_steps=1)
    observed = []
    trainer.callbacks.append(lambda *args: observed.append(loader.consumed))
    trainer.train()
    assert observed == [4]
    assert loader.consumed == 4


def test_empty_loader_fails_without_spinning_or_final_checkpoint(tmp_path):
    trainer = make_trainer(tmp_path, UnsizedBatches([]))
    with pytest.raises(RuntimeError, match="no batches"):
        trainer.train()
    assert trainer.global_step == 0
    assert not list((tmp_path / "checkpoints").glob("final_*"))


def test_max_epochs_finishes_short_update(tmp_path):
    trainer = make_trainer(tmp_path, UnsizedBatches([1, 2, 3]), max_steps=10, max_epochs=1)
    trainer.train()
    assert trainer.global_step == 1
    torch.testing.assert_close(trainer.model.weight, torch.tensor([[0.4]]))


def test_stop_callback_runs_after_normalized_short_update(tmp_path):
    trainer = make_trainer(tmp_path, UnsizedBatches([1]), max_steps=10)

    def stop(current, step, loss):
        current.stop_requested = True

    trainer.callbacks.append(stop)
    trainer.train()
    assert trainer.global_step == 1
    torch.testing.assert_close(trainer.model.weight, torch.tensor([[0.2]]))
    assert (tmp_path / "checkpoints" / "interrupted_1").is_dir()
    assert not list((tmp_path / "checkpoints").glob("final_*"))


def test_wsd_resume_during_decay_matches_uninterrupted_next_update(tmp_path):
    def create(root, resume=None):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        config = TrainingConfig(
            precision="fp32", use_fsdp=False, grad_clip=0.0,
            skip_param_update_check=True, lr_scheduler="wsd", learning_rate=0.01,
            max_steps=10, warmup_steps=1, wsd_decay_ratio=0.5,
            checkpoint_dir=str(root / "checkpoints"), log_dir=str(root / "logs"),
            resume_from=resume,
        )
        return Trainer(model, config, UnsizedBatches([1]),
                       compute_loss=lambda m, b: (m(torch.ones(1, 1)) - b["target"]).square().mean())

    first = create(tmp_path / "first")

    def stop_at_eight(trainer, step, loss):
        trainer.stop_requested = step == 8

    first.callbacks.append(stop_at_eight)
    first.train()
    checkpoint = Path(first.config.checkpoint_dir) / "interrupted_8"
    expected = create(tmp_path / "expected")
    expected.model.load_state_dict(first.model.state_dict())
    expected.optimizer.load_state_dict(first.optimizer.state_dict())
    expected.scheduler.load_state_dict(first.scheduler.state_dict())
    expected.global_step = 8
    expected.train()

    resumed = create(tmp_path / "resumed", str(checkpoint))
    resumed.train()
    assert resumed.global_step == expected.global_step == 10
    torch.testing.assert_close(resumed.model.weight, expected.model.weight, rtol=0, atol=0)
    assert resumed.scheduler.state_dict() == expected.scheduler.state_dict()


def _ddp_accumulation_worker(rank, root, rendezvous):
    import datetime

    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=40))
    try:
        values = [value + rank for value in [1, 2, 3, 4, 5]]
        trainer = make_trainer(Path(root) / str(rank), UnsizedBatches(values))
        trainer.train()
        reference = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(reference.weight)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        for _ in range(2):
            for group in [[1, 2, 3, 4], [5]]:
                targets = torch.tensor(group + [value + 1 for value in group], dtype=torch.float32)
                loss = (reference(torch.ones(1, 1)).reshape(()) - targets).square().mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
        torch.testing.assert_close(trainer.model.module.weight, reference.weight)
        assert trainer.global_step == 4
    finally:
        dist.destroy_process_group()


def test_two_process_ddp_short_accumulation_matches_global_mean(tmp_path):
    torch.multiprocessing.spawn(_ddp_accumulation_worker,
                               args=(str(tmp_path), str(tmp_path / "rendezvous")),
                               nprocs=2, join=True)
