from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import pytest

from complexity.jobs import (
    Job,
    JobHandle,
    JobManager,
    load_job_manifest,
    parse_job_status,
    wait_for_job_artifacts,
)
from complexity.training.supervisor import SupervisorManager

REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_submit_many_installs_group_and_applies_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager = JobManager(SupervisorManager(tmp_path / "supervisor"))
    first = _job(tmp_path, name="pretraining")
    second = _job(tmp_path, name="checkpoint_sync")

    handles = manager.submit_many((first, second))

    assert [handle.name for handle in handles] == ["pretraining", "checkpoint_sync"]
    assert (tmp_path / "supervisor" / "pretraining.conf").is_file()
    assert (tmp_path / "supervisor" / "checkpoint_sync.conf").is_file()
    assert calls == [["supervisorctl", "reread"], ["supervisorctl", "update"]]


def test_load_job_manifest_resolves_portable_paths_and_required_variables(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "pipeline.toml"
    manifest.write_text(
        """\
version = 1
required_variables = ["tokenizer"]

[variables]
log_dir = "{root}/logs"

[[jobs]]
name = "pretraining"
command = ["{python}", "-m", "training.worker"]
directory = "{root}"
log_path = "{log_dir}/pretraining.log"
autostart = false
startretries = 5

[jobs.environment]
TOKENIZER = "{tokenizer}"
""",
        encoding="utf-8",
    )
    project_root = tmp_path / "checkout"

    loaded = load_job_manifest(
        manifest,
        root=project_root,
        variables={"tokenizer": "/models/tokenizer"},
    )

    assert loaded.source == manifest.resolve()
    assert len(loaded.jobs) == 1
    job = loaded.jobs[0]
    assert job.name == "pretraining"
    assert job.directory == project_root.resolve()
    assert job.log_path == project_root.resolve() / "logs" / "pretraining.log"
    assert job.environment == {"TOKENIZER": "/models/tokenizer"}
    assert job.autostart is False
    assert job.startretries == 5

    with pytest.raises(ValueError, match="missing required job manifest variables"):
        load_job_manifest(manifest, root=project_root)


def test_load_job_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
    manifest = tmp_path / "pipeline.toml"
    manifest.write_text(
        """\
version = 1
[[jobs]]
name = "training"
command = ["python", "train.py"]
log_path = "training.log"
shell = true
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"unknown jobs\[0\] keys"):
        load_job_manifest(manifest)


def test_load_job_manifest_rejects_variable_cycles(tmp_path: Path) -> None:
    manifest = tmp_path / "pipeline.toml"
    manifest.write_text(
        """\
version = 1
[variables]
loop = "{loop}"
[[jobs]]
name = "training"
command = ["python", "train.py"]
log_path = "training.log"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reference cycle"):
        load_job_manifest(manifest)


def test_agentic_100m_pipeline_is_portable_and_uses_native_dependency_gate(
    tmp_path: Path,
) -> None:
    manifest = load_job_manifest(
        REPO_ROOT / "configs/jobs/tr_hash_agentic_100m_pipeline.toml",
        root=tmp_path,
        variables={
            "tokenizer": "/models/tr-hash-agentic-32k",
            "hf_token_file": "/run/secrets/huggingface-token",
        },
    )

    assert [job.name for job in manifest.jobs] == [
        "tr_hash_100m_pretraining",
        "tr_hash_100m_checkpoint_sync",
        "tr_hash_100m_refinement",
        "tr_hash_100m_refinement_sync",
    ]
    refinement = manifest.jobs[2]
    assert "run-after" in refinement.command
    assert "tr_hash_100m_pretraining" in refinement.command
    assert str(tmp_path / "artifacts/tr_hash_agentic_100m_pretraining/final/model.safetensors") in refinement.command
    rendered = "\n".join(job.as_supervisor_program().render() for job in manifest.jobs)
    assert "/home/boris" not in rendered
    assert "HF_TOKEN=" not in rendered
    assert "/run/secrets/huggingface-token" in rendered


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


def test_wait_for_job_artifacts_requires_exit_and_non_empty_files(tmp_path: Path) -> None:
    artifact = tmp_path / "final" / "model.safetensors"
    artifact.parent.mkdir()
    statuses = iter(
        (
            "pretraining RUNNING pid 42, uptime 0:01:00",
            "pretraining EXITED Sep 02 12:00 PM",
        )
    )

    class Manager:
        def status(self, name: str) -> str:
            assert name == "pretraining"
            status = next(statuses)
            if "EXITED" in status:
                artifact.write_bytes(b"weights")
            return status

    polls: list[str] = []
    wait_for_job_artifacts(
        Manager(),  # type: ignore[arg-type]
        "pretraining",
        [artifact],
        poll_seconds=1,
        sleep=lambda _seconds: None,
        on_poll=polls.append,
    )

    assert polls == ["job=pretraining state=RUNNING artifacts_ready=False"]


def test_wait_for_job_artifacts_rejects_incomplete_exit(tmp_path: Path) -> None:
    class Manager:
        def status(self, _name: str) -> str:
            return "pretraining EXITED Sep 02 12:00 PM"

    with pytest.raises(RuntimeError, match="exited without required artifacts"):
        wait_for_job_artifacts(
            Manager(),  # type: ignore[arg-type]
            "pretraining",
            [tmp_path / "missing.safetensors"],
            poll_seconds=1,
        )


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


@pytest.mark.parametrize(
    ("line", "style"),
    [
        ("[preflight] ready", "green"),
        ("20:00:00 | INFO | model ready", "bright_blue"),
        ("WARNING low disk", "yellow"),
        ("torch.OutOfMemoryError", "bold red"),
        ("TR-HASH MoE SFT", "cyan"),
    ],
)
def test_cli_log_colors_follow_semantics(line: str, style: str) -> None:
    jobs_cli = importlib.import_module("complexity.cli.commands.jobs")
    assert jobs_cli._log_style(line) == style
