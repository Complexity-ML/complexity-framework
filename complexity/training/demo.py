"""Bounded native TR-HASH training, exact continuation and export diagnostics.

All fixtures are synthetic and local. This workflow measures integration,
not language quality, and never loads published models or training assets.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from complexity.config import ModelConfig
from complexity.models import ComplexityModel

from .config import TrainingConfig
from .trainer import Trainer

FORMAT = "complexity-local-training-demo-v1"
SEED = 42
HORIZON = 4
PAUSE_UPDATE = 2
BATCH_SIZE = 2
SEQUENCE_LENGTH = 12


def demo_model_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=64,
        shared_intermediate_size=64,
        vocab_size=32,
        max_position_embeddings=SEQUENCE_LENGTH,
        attention_type="gqa",
        attention_dropout=0.1,
        mlp_type="tr_hash_engine",
        num_experts=4,
        top_k=2,
        routing_strategy="token_id_composed_modulo",
        use_custom_kernels=False,
        use_cggr=False,
    )


def synthetic_fixtures() -> dict[str, torch.Tensor]:
    """Pre-shifted labels retain each terminal EOS and mask the first position."""
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    inputs = torch.randint(1, 32, (12, SEQUENCE_LENGTH), generator=generator, device="cpu")
    labels = torch.cat((inputs[:, 1:], torch.zeros(12, 1, dtype=torch.long, device="cpu")), dim=1)
    labels[:, 0] = -100
    return {"input_ids": inputs, "labels": labels}


class _CursorLoader:
    """Finite epochs with an explicit update cursor, without worker lookahead."""

    def __init__(self, fixtures, *, training):
        self.rows = {key: value[:8] if training else value[8:] for key, value in fixtures.items()}
        self.training = training
        self.cursor = 0
        self.last_ids = []
        self.updates_per_epoch = len(self.rows["input_ids"]) // BATCH_SIZE

    def __iter__(self):
        start = self.cursor if self.training else 0
        end = (start // self.updates_per_epoch + 1) * self.updates_per_epoch
        for update in range(start, end):
            row = (update % self.updates_per_epoch) * BATCH_SIZE
            self.last_ids = list(range(row, row + BATCH_SIZE))
            if self.training:
                self.cursor = update + 1
            yield {key: value[row : row + BATCH_SIZE] for key, value in self.rows.items()}


class _Progress:
    def __init__(self, loader, *, stop_at=None):
        self.loader = loader
        self.stop_at = stop_at
        self.records = []
        self.trainer = None

    def on_resume(self, step):
        if type(step) is not int or not 0 <= step <= HORIZON:
            raise ValueError("Demo checkpoint has an invalid update cursor")
        self.loader.cursor = step
        # Trainer enters an epoch at the top of its loop. Reconstruct the
        # partially consumed epoch rather than counting it twice on resume.
        self.trainer.epoch = step // self.loader.updates_per_epoch

    def __call__(self, trainer, step, loss):
        if self.loader.cursor != step or not math.isfinite(loss):
            raise RuntimeError("Demo update, data cursor or loss is invalid")
        trainer.state.total_tokens = step * BATCH_SIZE * SEQUENCE_LENGTH
        self.records.append(
            {
                "step": step,
                "epoch": trainer.epoch,
                "row_ids": self.loader.last_ids,
                "loss": loss,
                "learning_rate": trainer.scheduler.get_last_lr()[0],
            }
        )
        if step == self.stop_at:
            trainer.stop_requested = True


def _seed(device):
    torch.set_rng_state(torch.Generator(device="cpu").manual_seed(SEED).get_state())
    random.seed(SEED)
    np.random.seed(SEED)
    if device == "mps":
        torch.mps.manual_seed(SEED)


def _rng(device):
    result = {
        "torch": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    if device == "mps":
        result["mps"] = torch.mps.get_rng_state()
    return result


def _restore_rng(state):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    if "mps" in state:
        torch.mps.set_rng_state(state["mps"])


def _build_trainer(output, fixtures, device, *, resume=None, stop_at=None):
    config = demo_model_config()
    with torch.device("cpu"):
        model = ComplexityModel(config)
    training = TrainingConfig(
        max_steps=HORIZON,
        batch_size=BATCH_SIZE,
        learning_rate=0.002,
        weight_decay=0.01,
        warmup_steps=1,
        lr_scheduler="cosine",
        precision="fp32",
        use_fsdp=False,
        eval_steps=0,
        save_steps=HORIZON + 1,
        save_total_limit=4,
        num_workers=0,
        pin_memory=False,
        checkpoint_dir=str(output / "checkpoints"),
        log_dir=str(output / "logs"),
        resume_from=str(resume) if resume is not None else None,
    )
    loader = _CursorLoader(fixtures, training=True)
    progress = _Progress(loader, stop_at=stop_at)
    trainer = Trainer(
        model,
        training,
        loader,
        eval_dataloader=_CursorLoader(fixtures, training=False),
        callbacks=[progress],
        device=device,
    )
    progress.trainer = trainer
    trainer.checkpoint_manager.resume_contract = {
        "format": FORMAT,
        "device": device,
        "precision": "fp32",
        "model": config.to_dict(),
        "seed": SEED,
        "fixture_sha256": hashlib.sha256(
            fixtures["input_ids"].numpy().tobytes() + fixtures["labels"].numpy().tobytes()
        ).hexdigest(),
        "training": {
            key: value
            for key, value in training.to_dict().items()
            if key not in {"checkpoint_dir", "log_dir", "resume_from"}
        },
        "cursor": "fixed_order_two_rows_per_update",
    }
    return trainer, progress


def _compare(actual, expected, *, tolerance=0.0):
    """Compare complete state trees, including AdamW moments and RNG tensors."""
    maximum = 0.0
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
            return {"passed": False, "max_abs_diff": None}
        left, right = actual.detach().cpu(), expected.detach().cpu()
        if left.dtype != right.dtype:
            return {"passed": False, "max_abs_diff": None}
        if left.is_floating_point():
            maximum = float((left - right).abs().max()) if left.numel() else 0.0
            passed = bool(torch.allclose(left, right, atol=tolerance, rtol=tolerance))
        else:
            passed = torch.equal(left, right)
    elif isinstance(expected, np.ndarray):
        passed = isinstance(actual, np.ndarray) and np.array_equal(actual, expected)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            return {"passed": False, "max_abs_diff": None}
        children = [
            _compare(actual[key], value, tolerance=tolerance) for key, value in expected.items()
        ]
        passed = all(child["passed"] for child in children)
        maximum = max((child["max_abs_diff"] or 0.0 for child in children), default=0.0)
    elif isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            return {"passed": False, "max_abs_diff": None}
        children = [
            _compare(left, right, tolerance=tolerance) for left, right in zip(actual, expected)
        ]
        passed = all(child["passed"] for child in children)
        maximum = max((child["max_abs_diff"] or 0.0 for child in children), default=0.0)
    else:
        passed = actual == expected
    return {"passed": bool(passed), "max_abs_diff": maximum}


@torch.no_grad()
def _logits(model, fixtures, device):
    model.eval()
    result = model(fixtures["input_ids"][8:].to(device))["logits"].cpu()
    if not torch.isfinite(result).all():
        raise RuntimeError("Demo produced nonfinite logits")
    return result


def run_training_demo(output: str | Path, *, device="cpu") -> dict:
    """Run four updates, pause at two, restore and verify against a fresh run."""
    if device not in {"cpu", "mps"}:
        raise ValueError("Demo device must be cpu or mps")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; refusing CPU fallback")
    if dist.is_initialized() or int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("The local training demo requires one process; do not use torchrun")
    if torch.get_default_dtype() != torch.float32:
        raise ValueError("Demo requires the default dtype float32")
    output = Path(output).absolute()
    if output.is_symlink() or (output.exists() and (not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError("Demo output must be a new or empty directory")
    saved_rng = _rng(device)
    try:
        output.mkdir(parents=True, exist_ok=True)
        fixtures = synthetic_fixtures()
        (output / "fixtures.json").write_text(
            json.dumps({key: value.tolist() for key, value in fixtures.items()})
        )
        _seed(device)
        first, paused_progress = _build_trainer(
            output / "resumed", fixtures, device, stop_at=PAUSE_UPDATE
        )
        (output / "plan.json").write_text(
            json.dumps(first.checkpoint_manager.resume_contract, indent=2) + "\n"
        )
        initial_loss = first.evaluate()
        initial_weights = {
            name: tensor.detach().cpu().clone() for name, tensor in first.model.state_dict().items()
        }
        first.train()
        checkpoint = output / "resumed" / "checkpoints" / f"interrupted_{PAUSE_UPDATE}"
        if first.global_step != PAUSE_UPDATE or not (checkpoint / "complete.json").is_file():
            raise RuntimeError("Demo did not publish its bounded complete checkpoint")

        # Constructor initialization deliberately consumes a different RNG;
        # only a real checkpoint restore can recover the continued trajectory.
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(SEED + 99).get_state())
        if device == "mps":
            torch.mps.manual_seed(SEED + 99)
        resumed, resumed_progress = _build_trainer(
            output / "resumed", fixtures, device, resume=checkpoint
        )
        resumed.train()
        continued_rng = _rng(device)
        final_loss = resumed.evaluate()
        continued_logits = _logits(resumed.model, fixtures, device)
        if not math.isfinite(initial_loss) or not math.isfinite(final_loss):
            raise RuntimeError("Demo evaluation produced a nonfinite loss")

        _seed(device)
        reference, reference_progress = _build_trainer(output / "reference", fixtures, device)
        reference.train()
        reference_rng = _rng(device)
        tolerance = 0.0 if device == "cpu" else 1e-6
        resumed_records = paused_progress.records + resumed_progress.records
        checks = {
            "model": _compare(
                resumed.model.state_dict(), reference.model.state_dict(), tolerance=tolerance
            ),
            "optimizer": _compare(
                resumed.optimizer.state_dict(),
                reference.optimizer.state_dict(),
                tolerance=tolerance,
            ),
            "scheduler": _compare(resumed.scheduler.state_dict(), reference.scheduler.state_dict()),
            "rng": _compare(continued_rng, reference_rng),
            "logits": _compare(
                continued_logits, _logits(reference.model, fixtures, device), tolerance=tolerance
            ),
            "cursor": {
                "passed": resumed_progress.loader.cursor
                == reference_progress.loader.cursor
                == HORIZON
            },
            "sample_order": {
                "passed": [row["row_ids"] for row in resumed_records]
                == [row["row_ids"] for row in reference_progress.records]
            },
            "parameters_updated": {
                "passed": any(
                    not torch.equal(initial_weights[name], tensor.detach().cpu())
                    for name, tensor in resumed.model.named_parameters()
                )
            },
        }
        export = output / "export"
        resumed.model.save_pretrained(export)
        with torch.device("cpu"):
            reloaded = ComplexityModel.from_pretrained(export, device=device, dtype=torch.float32)
        checks["export_reload"] = _compare(
            continued_logits, _logits(reloaded, fixtures, device), tolerance=tolerance
        )
        passed = all(check["passed"] for check in checks.values())
        report = {
            "format": FORMAT,
            "device": device,
            "precision": "fp32",
            "synthetic_data": True,
            "downloaded_assets": False,
            "passed": passed,
            "model_parameters": sum(parameter.numel() for parameter in resumed.model.parameters()),
            "training_examples": 8,
            "evaluation_examples": 4,
            "planned_updates": HORIZON,
            "reference_updates": reference.global_step,
            "pause_update": PAUSE_UPDATE,
            "resumed_from_update": PAUSE_UPDATE,
            "final_update": resumed.global_step,
            "data_cursor": resumed_progress.loader.cursor,
            "initial_eval_loss": initial_loss,
            "final_eval_loss": final_loss,
            "checks": checks,
            "comparison_tolerance": tolerance,
            "updates": resumed_records,
            "checkpoint": str(checkpoint),
            "export": str(export),
            "report": str(output / "report.json"),
            "release_eligible": False,
        }
        (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        if not passed:
            raise RuntimeError(f"Demo parity failed; inspect {output / 'report.json'}")
        return report
    finally:
        _restore_rng(saved_rng)
