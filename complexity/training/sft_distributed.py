"""Explicit DDP continuation of the native SFT experiment.

The portable single-device runner remains unchanged. A migrated run preserves
its scientific plan, not a claim of bitwise equivalence between runtimes.
"""

from __future__ import annotations

import copy
import io
import json
import os
import random
import shutil
import sys
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from .sft import (
    EXPERIMENT,
    SFTTokenShardDataset,
    _autocast,
    _digest,
    _evaluation_identity,
    _export_epoch,
    _optimizer_schedule,
    _prepare_resume_metrics,
    _read_json,
    _restore_rng,
    _rng_state,
    _stack_items,
    _verify_plan,
    load_sft_config,
    load_sft_model,
    reference_epoch_batches,
    sft_loss,
)

RUNTIME_FORMAT = "complexity-sft-ddp-v1"


class SFTLossModule(torch.nn.Module):
    """Keep both the transformer and output projection inside DDP forward."""

    def __init__(self, model, *, precision="fp32"):
        super().__init__()
        self.model = model
        self.precision = precision

    def forward(self, batch):
        with _autocast(self.model, self.precision):
            return sft_loss(self.model, batch)


def _collective_errors(error, stage):
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, str(error) if error is not None else None)
    if any(errors):
        failures = "; ".join(f"rank {rank}: {value}" for rank, value in enumerate(errors) if value)
        raise RuntimeError(f"Distributed SFT {stage} failed: {failures}")


def _local_call(action, stage):
    result, error = None, None
    try:
        result = action()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    _collective_errors(error, stage)
    return result


def _agree(value, stage):
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    if any(item != value for item in values):
        raise ValueError(f"DDP ranks disagree on {stage}")


def _rank0_call(action, stage):
    """All peers wait on the same status broadcast, including write failures."""
    result = [None]
    if dist.get_rank() == 0:
        try:
            result[0] = {"value": action(), "error": None}
        except Exception as exc:
            result[0] = {"error": f"{type(exc).__name__}: {exc}"}
    dist.broadcast_object_list(result, src=0)
    if result[0]["error"] is not None:
        raise RuntimeError(f"Distributed SFT {stage} failed on rank 0: {result[0]['error']}")
    return result[0]["value"]


@torch.no_grad()
def evaluate_distributed_sft(model, data, *, batch_size, max_batches=None, precision="fp32"):
    """Evaluate on rank0, reporting status after every held-out batch.

    Peers never enter a single collective spanning the entire evaluation.
    A root read/forward failure is broadcast at the same batch boundary.
    """
    iterator = None

    def initialize():
        nonlocal iterator
        model.eval()
        iterator = iter(DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=0))

    _rank0_call(initialize, "evaluation setup")
    batches = (len(data) + batch_size - 1) // batch_size
    if max_batches is not None:
        batches = min(batches, max_batches)
    total, targets, examples = 0.0, 0, 0
    for _ in range(batches):

        def evaluate_batch():
            batch = next(iterator)
            count = int((batch["labels"] != -100).sum())
            with _autocast(model, precision):
                loss = sft_loss(model, batch)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("SFT evaluation produced a nonfinite loss")
            return float(loss) * count, count, len(batch["input_ids"])

        batch_loss, count, items = _rank0_call(evaluate_batch, "evaluation batch")
        total += batch_loss
        targets += count
        examples += items
    return {
        "loss": total / targets,
        "supervised_tokens": targets,
        "packed_items": examples,
        "complete": examples == len(data),
        "aggregation": "supervised_token_weighted",
    }


def distributed_sft_update(ddp, data, items, optimizer, scheduler, *, micro_batch_size):
    """Average reference-rank means once; mask-weight microbatches within a rank."""
    error = None
    if type(micro_batch_size) is not int or micro_batch_size < 1 or not items:
        error = "DDP microbatch and rank item list must be positive"
    _collective_errors(error, "update settings")
    _agree(micro_batch_size, "microbatch size")
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, len(items))
    if len(set(counts)) != 1:
        raise ValueError("DDP ranks must have equally padded reference batches")
    target_count = _local_call(
        lambda: sum(int((data[item]["labels"] != -100).sum()) for item in items), "rank targets"
    )
    _collective_errors(
        None if target_count > 0 else "rank has no supervised targets", "rank targets"
    )
    optimizer.zero_grad(set_to_none=True)
    ddp.train()
    rank_loss = 0.0
    for start in range(0, len(items), micro_batch_size):
        batch = _local_call(
            lambda: _stack_items(data, items[start : start + micro_batch_size]), "batch read"
        )
        targets = int((batch["labels"] != -100).sum())
        final = start + micro_batch_size >= len(items)
        context = nullcontext() if final else ddp.no_sync()
        with context:
            loss = _local_call(lambda: ddp(batch), "forward")
            _collective_errors(None if bool(torch.isfinite(loss)) else "nonfinite loss", "loss")
            fraction = targets / target_count
            # DDP averages rank gradients. Do not divide by world size twice.
            _local_call(lambda: (loss * fraction).backward(), "backward")
            rank_loss += float(loss.detach()) * fraction
    base = ddp.module.model
    norm = _local_call(
        lambda: torch.nn.utils.clip_grad_norm_(base.parameters(), 1.0, error_if_nonfinite=True),
        "gradient clipping",
    )
    _local_call(optimizer.step, "optimizer update")
    _local_call(scheduler.step, "scheduler update")
    optimizer.zero_grad(set_to_none=True)
    mean_loss = torch.tensor(rank_loss, dtype=torch.float64, device=next(base.parameters()).device)
    dist.all_reduce(mean_loss)
    return {
        "loss": float(mean_loss / dist.get_world_size()),
        "gradient_norm": float(norm),
        "lr": scheduler.get_last_lr()[0],
    }


def _artifact_roles(plan):
    roots = (("base", Path(plan["checkpoint"])), ("dataset", Path(plan["data"])))
    files = {}
    for filename, info in plan["files"].items():
        path = Path(filename)
        for role, root in roots:
            if path.is_relative_to(root):
                key = f"{role}/{path.relative_to(root)}"
                break
        else:
            raise ValueError(f"Unclassified SFT artifact: {filename}")
        if key in files:
            raise ValueError(f"Duplicate SFT artifact role: {key}")
        files[key] = info
    return files


def scientific_contract(plan):
    """Compare immutable content and settings independently of host-local paths."""
    ignored = {
        "checkpoint",
        "data",
        "files",
        "training_started",
        "limitations",
        "supported_devices",
    }
    return {
        **{key: value for key, value in plan.items() if key not in ignored},
        "artifacts": _artifact_roles(plan),
    }


def _eval_contract(evaluation):
    root = Path(evaluation["partition"])
    files = {}
    for filename, digest in evaluation["files"].items():
        path = Path(filename)
        relative = (
            path.relative_to(root)
            if path.is_relative_to(root)
            else Path("..") / path.relative_to(root.parent)
        )
        files[str(relative)] = digest
    return {"max_batches": evaluation["max_batches"], "files": files}


def distributed_identity(
    plan,
    *,
    device,
    precision,
    micro_batch_size,
    gradient_checkpointing,
    evaluation,
    eval_max_batches,
):
    return {
        "runtime_format": RUNTIME_FORMAT,
        "plan": plan,
        "scientific_contract": scientific_contract(plan),
        "world_size": plan["world_size"],
        "backend": "nccl" if device.type == "cuda" else "gloo",
        "device_type": device.type,
        "precision": precision,
        "master_parameters": "fp32",
        "loss_backend": "chunked_pytorch_cross_entropy",
        "micro_batch_size": micro_batch_size,
        "gradient_checkpointing": gradient_checkpointing,
        "loss_aggregation": "equal_reference_rank_means",
        "evaluation_aggregation": "supervised_token_weighted",
        "evaluation": _eval_contract(_evaluation_identity(evaluation, eval_max_batches)),
        "torch_version": str(torch.__version__),
    }


def validate_resume_payload(state, marker, identity, *, migrate_single=False):
    """Require explicit runtime migration; preserve optimizer horizon and data."""
    step = state.get("step")
    if marker.get("experiment") != EXPERIMENT or marker.get("release_eligible") is not False:
        raise ValueError("SFT complete marker experiment/release contract differs")
    if (
        type(step) is not int
        or not 0 <= step <= identity["plan"]["optimizer_updates"]
        or marker.get("step") != step
    ):
        raise ValueError("Invalid SFT checkpoint update cursor")
    previous = state.get("identity", {})
    distributed = state.get("runtime_format") == RUNTIME_FORMAT
    if distributed:
        if migrate_single:
            raise ValueError("migrate_single applies only to a portable single-device checkpoint")
        if previous != identity:
            raise ValueError("DDP resume runtime/rank identity differs from checkpoint")
        if marker.get("runtime_format") != RUNTIME_FORMAT:
            raise ValueError("DDP complete marker runtime differs")
        ranks = state.get("rng_by_rank", [])
        if len(ranks) != identity["world_size"] or [item.get("rank") for item in ranks] != list(
            range(identity["world_size"])
        ):
            raise ValueError("DDP checkpoint rank RNG identity is incomplete or reordered")
        return None
    if not migrate_single:
        raise ValueError("Portable single-device SFT requires explicit migrate_single")
    if state.get("runtime_format") is not None or "rng" not in state:
        raise ValueError("Unsupported source runtime for SFT migration")
    if marker.get("runtime_format") is not None:
        raise ValueError("Portable source complete marker runtime differs")
    if not isinstance(previous.get("plan"), dict):
        raise ValueError("SFT migration lacks its scientific source plan")
    if scientific_contract(previous.get("plan", {})) != identity["scientific_contract"]:
        raise ValueError("SFT migration scientific plan/data contract differs")
    for key in (
        "precision",
        "master_parameters",
        "micro_batch_size",
        "gradient_checkpointing",
        "loss_aggregation",
        "evaluation_aggregation",
    ):
        if previous.get(key) != identity[key]:
            raise ValueError(f"SFT migration setting differs: {key}")
    if _eval_contract(previous["evaluation"]) != identity["evaluation"]:
        raise ValueError("SFT migration evaluation contract differs")
    source_device = torch.device(previous["device"])
    if source_device.type != identity["device_type"]:
        raise ValueError("SFT migration device type differs")
    return {
        "source_runtime": "portable_single_device",
        "source_step": step,
        "rng_initialization": "copy_single_rng_to_each_rank",
        "bitwise_equivalence_claimed": False,
    }


def _cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_cpu_tree(item) for item in value)
    return value


def _finite_state(value):
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(_finite_state(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_state(item) for item in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def save_distributed_checkpoint(
    output, *, model, optimizer, scheduler, step, identity, device, history, migration=None
):
    """Gather each rank's RNG; publish payload+marker before latest/rotation."""
    from complexity.utils.local_checkpoint import rotate_local_checkpoints

    def capture_rng():
        stream = io.BytesIO()
        torch.save({"rank": dist.get_rank(), "state": _rng_state(device)}, stream)
        return stream.getvalue()

    # Object collectives carry bytes, never pickled torch storages. This also
    # keeps CUDA RNG tensors off the object-collective device conversion path.
    local_rng = _local_call(capture_rng, "RNG capture")
    encoded_states = [None] * dist.get_world_size()
    dist.all_gather_object(encoded_states, local_rng)
    output = Path(output)

    def save():
        output.mkdir(parents=True, exist_ok=True)
        target, pending = output / f"step_{step:06d}", output / f".step_{step:06d}.writing"
        if pending.exists():
            shutil.rmtree(pending)
        pending.mkdir()
        state = {
            "runtime_format": RUNTIME_FORMAT,
            "model": _cpu_tree(model.state_dict()),
            "optimizer": _cpu_tree(optimizer.state_dict()),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "identity": identity,
            "rng_by_rank": [
                torch.load(io.BytesIO(item), weights_only=True) for item in encoded_states
            ],
            "history": history,
            "migration": migration,
        }
        torch.save(state, pending / "checkpoint.pt")
        (pending / "complete.json").write_text(
            json.dumps(
                {
                    "runtime_format": RUNTIME_FORMAT,
                    "step": step,
                    "experiment": EXPERIMENT,
                    "release_eligible": False,
                    "checkpoint_sha256": _digest(pending / "checkpoint.pt"),
                },
                indent=2,
            )
            + "\n"
        )
        if target.exists():
            shutil.rmtree(target)
        pending.rename(target)
        latest = output / ".latest.writing"
        latest.write_text(target.name + "\n")
        latest.replace(output / "latest")
        rotate_local_checkpoints(output, total_limit=3)
        return str(target)

    return _rank0_call(save, "checkpoint publication")


def _device_policy(device, precision):
    """Check authored runtime settings without initializing any hardware."""
    device = torch.device(device)
    if device.type not in {"cpu", "cuda"} or (device.type == "cpu" and precision != "fp32"):
        raise ValueError("DDP SFT supports CPU/fp32 or CUDA/fp32,bf16")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("Invalid DDP precision")
    return device


def _setup(plan, device, precision):
    device = _device_policy(device, precision)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; refusing DDP fallback")
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(device)
        if precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("BF16 unavailable on this CUDA rank")
    backend = "nccl" if device.type == "cuda" else "gloo"
    owned = not dist.is_initialized()
    if owned:
        if int(os.environ.get("WORLD_SIZE", "1")) != plan["world_size"] or plan["world_size"] < 2:
            raise ValueError("DDP launch requires torchrun with the plan's exact world_size >=2")
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=120))
    if dist.get_world_size() != plan["world_size"] or dist.get_backend() != backend:
        raise ValueError("Process group differs from DDP plan world_size/backend")
    return device, owned


def run_distributed_sft(
    plan,
    output,
    *,
    device,
    micro_batch_size=1,
    steps=None,
    resume=None,
    eval_data=None,
    eval_max_batches=None,
    save_every=100,
    gradient_checkpointing=True,
    precision="fp32",
    migrate_single=False,
):
    """Run one declared reference batch per rank under an existing torchrun job."""
    device, owned = _setup(plan, device, precision)
    try:
        return _run(
            plan,
            Path(output),
            device=device,
            micro_batch_size=micro_batch_size,
            steps=steps,
            resume=resume,
            eval_data=eval_data,
            eval_max_batches=eval_max_batches,
            save_every=save_every,
            gradient_checkpointing=gradient_checkpointing,
            precision=precision,
            migrate_single=migrate_single,
        )
    finally:
        if owned and dist.is_initialized():
            dist.destroy_process_group()


def _run(
    plan,
    output,
    *,
    device,
    micro_batch_size,
    steps,
    resume,
    eval_data,
    eval_max_batches,
    save_every,
    gradient_checkpointing,
    precision,
    migrate_single,
):
    _agree(
        {
            "steps": steps,
            "resume": resume is not None,
            "migrate_single": migrate_single,
            "save_every": save_every,
            "output": str(output.resolve()),
        },
        "run controls",
    )

    def validate():
        _verify_plan(plan)
        if (
            type(micro_batch_size) is not int
            or micro_batch_size < 1
            or type(save_every) is not int
            or save_every < 1
        ):
            raise ValueError("DDP microbatch/save interval must be positive integers")
        if steps is not None and (
            type(steps) is not int or not 1 <= steps <= plan["optimizer_updates"]
        ):
            raise ValueError("DDP steps must be within the complete horizon")
        if eval_max_batches is not None and (
            type(eval_max_batches) is not int or eval_max_batches < 1
        ):
            raise ValueError("DDP evaluation limit must be positive")
        if (steps is None or steps == plan["optimizer_updates"]) and eval_max_batches is not None:
            raise ValueError("Full DDP experiment requires complete held-out evaluation")
        if migrate_single and resume is None:
            raise ValueError("migrate_single requires an explicit source checkpoint")
        if (resume is None or migrate_single) and output.exists() and any(output.iterdir()):
            raise FileExistsError("Fresh/migrated DDP run requires a new output directory")
        train = SFTTokenShardDataset(
            plan["data"],
            sequence_length=plan["sequence_length"],
            vocab_size=plan["model_config"]["vocab_size"],
        )
        evaluation = SFTTokenShardDataset(
            eval_data or Path(plan["data"]) / "eval",
            sequence_length=plan["sequence_length"],
            vocab_size=train.vocab_size,
        )
        identity = distributed_identity(
            plan,
            device=device,
            precision=precision,
            micro_batch_size=micro_batch_size,
            gradient_checkpointing=gradient_checkpointing,
            evaluation=evaluation,
            eval_max_batches=eval_max_batches,
        )
        return train, evaluation, identity

    train, evaluation, identity = _local_call(validate, "preflight")
    _agree(identity, "experiment identity")
    state, migration = None, None
    if resume is not None:

        def restore():
            from complexity.utils.local_checkpoint import resolve_checkpoint_path

            checkpoint = resolve_checkpoint_path(resume)
            marker = _read_json(checkpoint / "complete.json")
            if marker.get("checkpoint_sha256") != _digest(checkpoint / "checkpoint.pt"):
                raise ValueError("DDP source checkpoint integrity failed")
            saved = torch.load(checkpoint / "checkpoint.pt", map_location="cpu", weights_only=True)
            migration = validate_resume_payload(
                saved, marker, identity, migrate_single=migrate_single
            )
            if (
                saved.get("runtime_format") == RUNTIME_FORMAT
                and marker.get("runtime_format") != RUNTIME_FORMAT
            ):
                raise ValueError("DDP complete marker runtime differs")
            if steps is not None and steps < saved["step"]:
                raise ValueError("DDP absolute step limit precedes restored checkpoint")
            for key in ("model", "optimizer", "scheduler"):
                if not _finite_state(saved[key]):
                    raise ValueError(f"Nonfinite source {key} state")
            if saved["scheduler"].get("last_epoch") != saved["step"]:
                raise ValueError("Source scheduler cursor differs from optimizer update")
            if migration is not None:
                migration["source_checkpoint_sha256"] = marker["checkpoint_sha256"]
            return saved, migration or saved.get("migration"), marker["checkpoint_sha256"]

        state, migration, checkpoint_sha = _local_call(restore, "resume validation")
        _agree(
            {
                "step": state["step"],
                "sha256": checkpoint_sha,
                "runtime_format": state.get("runtime_format"),
            },
            "restored checkpoint",
        )
    rank = dist.get_rank()
    torch.manual_seed(plan["seed"] + rank)
    random.seed(plan["seed"] + rank)
    np.random.seed(plan["seed"] + rank)

    def initialize():
        model = load_sft_model(plan["checkpoint"]).to(device)
        if gradient_checkpointing:
            model.gradient_checkpointing_enable()
        optimizer, scheduler = _optimizer_schedule(model, plan)
        if state is not None:
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
        return model, optimizer, scheduler

    model, optimizer, scheduler = _local_call(initialize, "model/optimizer restore")
    ddp = _local_call(
        lambda: DDP(
            SFTLossModule(model, precision=precision),
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
            find_unused_parameters=False,
        ),
        "DDP initialization",
    )
    if state is not None:
        rng = state["rng"] if migrate_single else state["rng_by_rank"][rank]["state"]
        _local_call(lambda: _restore_rng(rng, device), "RNG restore")
    step, history = (state["step"], copy.deepcopy(state["history"])) if state else (0, [])
    _rank0_call(lambda: output.mkdir(parents=True, exist_ok=True), "output creation")
    if state is not None:
        _rank0_call(lambda: _prepare_resume_metrics(output, step), "metrics recovery")
    else:
        baseline = evaluate_distributed_sft(
            model,
            evaluation,
            batch_size=micro_batch_size,
            max_batches=eval_max_batches,
            precision=precision,
        )
        history.append({"step": 0, "epoch": 0, "evaluation": baseline})
    if state is None or migrate_single:
        save_distributed_checkpoint(
            output,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            step=step,
            identity=identity,
            device=device,
            history=history,
            migration=migration,
        )
    _rank0_call(
        lambda: (output / "run.json").write_text(
            json.dumps(identity, sort_keys=True, indent=2) + "\n"
        ),
        "run metadata",
    )
    if migration is not None:
        _rank0_call(
            lambda: (output / "migration.json").write_text(json.dumps(migration, indent=2) + "\n"),
            "migration metadata",
        )
    horizon = plan["optimizer_updates"] if steps is None else steps
    started = time.monotonic()
    while step < horizon:
        epoch, cursor = divmod(step, plan["steps_per_epoch"])
        for position, ranks in enumerate(reference_epoch_batches(plan, epoch)):
            if position < cursor:
                continue
            update_started = time.monotonic()
            values = distributed_sft_update(
                ddp, train, ranks[rank], optimizer, scheduler, micro_batch_size=micro_batch_size
            )
            step += 1
            record = {
                "step": step,
                "epoch": epoch + 1,
                **values,
                "seconds": time.monotonic() - update_started,
            }

            def write_metric():
                with (output / "metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
                    stream.flush()

            _rank0_call(write_metric, "metrics publication")
            if step % plan["steps_per_epoch"] == 0:
                held_out = evaluate_distributed_sft(
                    model,
                    evaluation,
                    batch_size=micro_batch_size,
                    max_batches=eval_max_batches,
                    precision=precision,
                )
                history.append({"step": step, "epoch": epoch + 1, "evaluation": held_out})
                _rank0_call(
                    lambda: _export_epoch(model, train, output, epoch + 1, step, identity),
                    "epoch export",
                )
            if step % save_every == 0 or step >= horizon or step % plan["steps_per_epoch"] == 0:
                save_distributed_checkpoint(
                    output,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=step,
                    identity=identity,
                    device=device,
                    history=history,
                    migration=migration,
                )
            if step >= horizon:
                break
    result = {
        "runtime_format": RUNTIME_FORMAT,
        "experiment": EXPERIMENT,
        "release_eligible": False,
        "step": step,
        "planned_updates": plan["optimizer_updates"],
        "world_size": dist.get_world_size(),
        "complete": step == plan["optimizer_updates"],
        "elapsed_seconds": time.monotonic() - started,
        "evaluations": history,
        "migration": migration,
    }
    _rank0_call(
        lambda: (output / "result.json").write_text(json.dumps(result, indent=2) + "\n"),
        "result publication",
    )
    return result


def execute_distributed_sft_config(
    path, *, run=False, steps=None, resume=None, migrate_single=False
):
    plan, execution = load_sft_config(path)
    if plan["world_size"] < 2:
        raise ValueError("Distributed SFT requires world_size >=2")
    if "device" in execution:
        _device_policy(execution["device"], execution.get("precision", "fp32"))
    if not run:
        if steps is not None or resume is not None or migrate_single:
            raise ValueError("DDP steps/resume/migration require explicit --run")
        return {
            **plan,
            "execution": execution,
            "runtime_format": RUNTIME_FORMAT,
            "supported_devices": ["cpu", "cuda"],
            "loss_backend": "chunked_pytorch_cross_entropy",
            "hardware_qualified": False,
            "launch_command": [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                str(plan["world_size"]),
                "-m",
                "complexity",
                "train",
                "sft",
                str(Path(path).resolve()),
                "--distributed",
                "--run",
            ],
        }
    if not {"output", "device"} <= execution.keys():
        raise ValueError("DDP execution requires output/device")
    output = execution.pop("output")
    return run_distributed_sft(
        plan, output, steps=steps, resume=resume, migrate_single=migrate_single, **execution
    )
