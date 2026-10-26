"""Upload backpressure and interruption without allocating a GPU model."""

from pathlib import Path

import pytest

from scripts.pretrain_1b_fsdp.flow_control import (
    CHECKPOINT,
    STOP,
    WAIT,
    upload_control_state,
    wait_for_upload_capacity,
)


def backlog(root):
    output, runtime = root / "run", root / "runtime"
    runtime.mkdir(parents=True)
    for step in range(3):
        path = output / f"step_{step:07d}"
        path.mkdir(parents=True)
        (path / "complete.json").write_text("{}")
    # An incomplete checkpoint must never contribute to backpressure.
    (output / "step_0000003").mkdir()
    return output, runtime


def test_only_complete_unacknowledged_checkpoints_block(tmp_path):
    output, runtime = backlog(tmp_path)
    assert upload_control_state(output, runtime) == WAIT
    (runtime / "step_0000000.uploaded.json").write_text("{}")
    assert upload_control_state(output, runtime) == 0


@pytest.mark.parametrize("names, expected", [
    (["request-checkpoint"], CHECKPOINT),
    (["request-stop"], STOP),
    (["request-checkpoint", "request-stop"], CHECKPOINT | STOP),
])
def test_requests_override_full_backlog(tmp_path, names, expected):
    output, runtime = backlog(tmp_path)
    for name in names:
        (runtime / name).touch()
    assert upload_control_state(output, runtime) == expected


def test_bounded_runs_bypass_backpressure_but_keep_stop_requests(tmp_path):
    output, runtime = backlog(tmp_path)
    assert upload_control_state(output, runtime, limited=True) == 0
    (runtime / "request-stop").touch()
    assert upload_control_state(output, runtime, limited=True) == STOP


def test_stop_is_polled_again_during_upload_outage(tmp_path):
    output, runtime = backlog(tmp_path)
    sleeps = []
    states = []

    def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) == 3:
            (runtime / "request-stop").touch()

    def synchronize(value):
        states.append(value)
        return value

    request = wait_for_upload_capacity(
        lambda: upload_control_state(output, runtime), synchronize,
        is_main=True, on_wait=lambda: None, sleep=sleep,
    )
    assert request == STOP
    assert states == [WAIT, WAIT, WAIT, STOP]
    assert sleeps == [30, 30, 30]


def test_follower_participates_without_reading_shared_files():
    values = iter([WAIT, WAIT, CHECKPOINT | STOP])
    sleeps = []

    def read_state():
        pytest.fail("Only rank zero should inspect upload state")

    result = wait_for_upload_capacity(read_state, lambda value: next(values),
                                     is_main=False, on_wait=lambda: pytest.fail("Main-only log"),
                                     sleep=sleeps.append)
    assert result == CHECKPOINT | STOP
    assert sleeps == [30, 30]


@pytest.mark.parametrize("release", ["upload", "request-stop", "request-checkpoint"])
def test_capacity_or_request_arriving_during_wait(tmp_path, release):
    output, runtime = backlog(tmp_path)

    def release_backlog():
        name = "step_0000000.uploaded.json" if release == "upload" else release
        (runtime / name).touch()

    result = wait_for_upload_capacity(
        lambda: upload_control_state(output, runtime), lambda state: state,
        is_main=True, on_wait=release_backlog, sleep=lambda seconds: None,
    )
    assert result == {"upload": 0, "request-stop": STOP, "request-checkpoint": CHECKPOINT}[release]


def _flow_control_worker(rank, root, rendezvous):
    import datetime

    import torch
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=datetime.timedelta(seconds=40))
    try:
        root = Path(root)
        output, runtime = root / "run", root / "runtime"
        for release in ["upload", "request-checkpoint", "request-stop"]:
            if rank == 0:
                (runtime / "step_0000000.uploaded.json").unlink(missing_ok=True)
                (runtime / "request-checkpoint").unlink(missing_ok=True)
                (runtime / "request-stop").unlink(missing_ok=True)
            dist.barrier()
            states = []
            polls = 0

            def synchronize(value):
                control = torch.tensor(value, dtype=torch.int64)
                dist.broadcast(control, src=0)
                state = int(control.item())
                states.append(state)
                return state

            def read_state():
                assert rank == 0
                return upload_control_state(output, runtime)

            def on_wait():
                nonlocal polls
                assert rank == 0
                polls += 1
                if polls == 3:
                    name = "step_0000000.uploaded.json" if release == "upload" else release
                    (runtime / name).touch()

            request = wait_for_upload_capacity(read_state, synchronize, is_main=rank == 0,
                                               on_wait=on_wait, sleep=lambda seconds: None)
            expected = {"upload": 0, "request-stop": STOP, "request-checkpoint": CHECKPOINT}[release]
            assert request == expected
            assert states == [WAIT, WAIT, WAIT, expected]
    finally:
        dist.destroy_process_group()


def test_two_process_upload_wait_coordinates_recovery_checkpoint_and_stop(tmp_path):
    import torch.multiprocessing as mp

    backlog(tmp_path)
    mp.spawn(_flow_control_worker, args=(str(tmp_path), str(tmp_path / "rendezvous")),
             nprocs=2, join=True)
