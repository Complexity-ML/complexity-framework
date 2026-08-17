from __future__ import annotations

import importlib.util
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path("scripts/watch_and_restart_stalled_process.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("watch_and_restart_stalled_process", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    return _load_module()


def test_seconds_since_last_write_returns_none_for_missing_file(mod, tmp_path) -> None:
    assert mod.seconds_since_last_write(tmp_path / "missing.log") is None


def test_seconds_since_last_write_reports_elapsed_time(mod, tmp_path) -> None:
    log_file = tmp_path / "run.log"
    log_file.write_text("step 1\n")
    mtime = log_file.stat().st_mtime

    idle = mod.seconds_since_last_write(log_file, now=mtime + 42.0)

    assert idle == pytest.approx(42.0)


def test_watch_once_does_not_restart_a_fresh_log(mod, tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "run.log"
    log_file.write_text("step 1\n")
    mtime = log_file.stat().st_mtime

    restart_mock = MagicMock()
    monkeypatch.setattr(mod, "restart_program", restart_mock)

    triggered = mod.watch_once(log_file, "some_program", stall_seconds=900.0, now=mtime + 10.0)

    assert triggered is False
    restart_mock.assert_not_called()


def test_watch_once_restarts_a_stale_log(mod, tmp_path, monkeypatch) -> None:
    log_file = tmp_path / "run.log"
    log_file.write_text("step 1\n")
    mtime = log_file.stat().st_mtime

    restart_mock = MagicMock()
    monkeypatch.setattr(mod, "restart_program", restart_mock)

    triggered = mod.watch_once(log_file, "some_program", stall_seconds=900.0, now=mtime + 901.0)

    assert triggered is True
    restart_mock.assert_called_once_with("some_program")


def test_watch_once_does_not_restart_before_the_log_file_exists(mod, tmp_path, monkeypatch) -> None:
    restart_mock = MagicMock()
    monkeypatch.setattr(mod, "restart_program", restart_mock)

    triggered = mod.watch_once(tmp_path / "not_yet.log", "some_program", stall_seconds=900.0)

    assert triggered is False
    restart_mock.assert_not_called()


def test_restart_program_calls_supervisorctl_and_returns_true_on_success(mod, monkeypatch) -> None:
    run_mock = MagicMock()
    run_mock.return_value = MagicMock(returncode=0, stdout="some_program: stopped\nsome_program: started\n", stderr="")
    monkeypatch.setattr(mod.subprocess, "run", run_mock)

    ok = mod.restart_program("some_program")

    assert ok is True
    run_mock.assert_called_once_with(
        ["supervisorctl", "restart", "some_program"], capture_output=True, text=True
    )


def test_restart_program_returns_false_on_nonzero_exit(mod, monkeypatch) -> None:
    run_mock = MagicMock()
    run_mock.return_value = MagicMock(returncode=1, stdout="", stderr="FAILED: not running")
    monkeypatch.setattr(mod.subprocess, "run", run_mock)

    ok = mod.restart_program("some_program")

    assert ok is False
