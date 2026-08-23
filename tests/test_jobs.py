from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

from complexity.jobs import Job, JobHandle, JobManager, parse_job_status
from complexity.training.supervisor import SupervisorManager


def _job(tmp_path: Path, **overrides: object) -> Job:
    values: dict[str, object] = {
        "name": "evaluation_job",
        "command": ("/usr/bin/env", "python", "-m", "evaluation.worker"),
        "directory": tmp_path,
        "log_path": tmp_path / "evaluation.log",
        "environment": {"NPROC_PER_NODE": "4"},
    }
    values.update(overrides)
    return Job(**values)  # type: ignore[arg-type]


def test_foreground_run_uses_argv_without_a_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager = JobManager(SupervisorManager(tmp_path / "supervisor"))
    result = manager.run(_job(tmp_path))

    assert isinstance(result, subprocess.CompletedProcess)
    assert calls[0][0] == ["/usr/bin/env", "python", "-m", "evaluation.worker"]
    assert calls[0][1]["cwd"] == tmp_path
    assert calls[0][1]["check"] is True
    assert "shell" not in calls[0][1]


def test_detached_run_installs_private_config_and_returns_handle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    config_directory = tmp_path / "supervisor"
    manager = JobManager(SupervisorManager(config_directory))
    result = manager.run(_job(tmp_path), detached=True)

    assert isinstance(result, JobHandle)
    assert result.name == "evaluation_job"
    generated = config_directory / "evaluation_job.conf"
    assert generated.exists()
    assert generated.stat().st_mode & 0o777 == 0o600
    assert calls == [["supervisorctl", "reread"], ["supervisorctl", "update"]]


def test_manager_reads_only_the_configured_log_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager = JobManager(SupervisorManager(tmp_path / "supervisor"))
    job = _job(tmp_path)
    manager.submit(job)
    job.log_path.write_text("one\ntwo\nthree\n", encoding="utf-8")

    assert manager.log_path(job.name) == job.log_path
    assert manager.logs(job.name, lines=2) == "two\nthree\n"


def test_job_reuses_supervisor_security_validation(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="secret environment"):
        _job(tmp_path, environment={"HF_TOKEN": "forbidden"})

    with pytest.raises(ValueError, match="program name"):
        _job(tmp_path, name="bad:name")


def test_job_environment_cannot_be_mutated_after_validation(tmp_path: Path) -> None:
    job = _job(tmp_path)

    with pytest.raises(TypeError):
        job.environment["HF_TOKEN"] = "late-secret"  # type: ignore[index]


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("train RUNNING pid 42, uptime 0:01:00", ("train", "RUNNING", "pid 42, uptime 0:01:00")),
        ("eval FATAL exited too quickly", ("eval", "FATAL", "exited too quickly")),
        ("", ("", "UNKNOWN", "")),
    ],
)
def test_parse_job_status(line: str, expected: tuple[str, str, str]) -> None:
    assert parse_job_status(line) == expected


def test_cli_status_uses_semantic_colors(monkeypatch: pytest.MonkeyPatch) -> None:
    jobs_cli = importlib.import_module("complexity.cli.commands.jobs")
    captured: dict[str, object] = {}

    def fake_table(title: str, columns: list[object], rows: list[object]) -> None:
        captured.update(title=title, columns=columns, rows=rows)

    monkeypatch.setattr(jobs_cli, "HAS_RICH", True)
    monkeypatch.setattr(jobs_cli, "print_table", fake_table)
    jobs_cli._print_status(
        "train RUNNING pid 42, uptime 0:01:00\n"
        "evaluation STOPPED Not started\n"
        "sync FATAL exited too quickly"
    )

    assert captured["title"] == "Framework jobs"
    assert captured["rows"] == [
        ("train", "[bold green]RUNNING[/bold green]", "pid 42, uptime 0:01:00"),
        ("evaluation", "[yellow]STOPPED[/yellow]", "Not started"),
        ("sync", "[bold red]FATAL[/bold red]", "exited too quickly"),
    ]
