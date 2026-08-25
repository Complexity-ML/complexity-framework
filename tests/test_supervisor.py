from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from complexity.training.supervisor import (
    SupervisorManager,
    SupervisorProgram,
)


def _program(**overrides: object) -> SupervisorProgram:
    values: dict[str, object] = {
        "name": "training_job",
        "command": ("/bin/bash", "/workspace/repo/run training.sh"),
        "directory": Path("/workspace/repo"),
        "stdout_logfile": Path("/workspace/repo/artifacts/training.log"),
        "environment": {"NPROC_PER_NODE": "4", "OUTPUT": "artifacts/run,one"},
    }
    values.update(overrides)
    return SupervisorProgram(**values)  # type: ignore[arg-type]


def test_program_renders_a_deterministic_shell_escaped_configuration() -> None:
    rendered = _program().render()

    assert rendered.startswith("[program:training_job]\n")
    assert "command=/bin/bash '/workspace/repo/run training.sh'" in rendered
    assert "autostart=true" in rendered
    assert "autorestart=unexpected" in rendered
    assert "stopasgroup=true" in rendered
    assert "killasgroup=true" in rendered
    assert "stdout_logfile=/workspace/repo/artifacts/training.log" in rendered
    assert 'environment=NPROC_PER_NODE="4",OUTPUT="artifacts/run,one"' in rendered
    assert "/dev/stdout" not in rendered


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"name": "bad:name"}, "program name"),
        ({"command": ()}, "command"),
        ({"directory": Path("relative")}, "absolute"),
        ({"stdout_logfile": Path("relative.log")}, "absolute"),
        ({"environment": {"HF_TOKEN": "secret"}}, "secret environment"),
        ({"environment": {"GOOD": "line\nbreak"}}, "NUL/newline"),
    ],
)
def test_program_rejects_unsafe_configuration(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _program(**overrides)


def test_tokenizer_path_is_not_mistaken_for_a_secret() -> None:
    program = _program(environment={"TOKENIZER": "/workspace/tokenizer"})
    assert 'environment=TOKENIZER="/workspace/tokenizer"' in program.render()


def test_manager_installs_atomically_with_private_permissions(tmp_path: Path) -> None:
    manager = SupervisorManager(tmp_path)
    destination = manager.install(_program())

    assert destination == tmp_path / "training_job.conf"
    assert destination.read_text(encoding="utf-8") == _program().render()
    assert destination.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob("*.tmp"))
    assert manager.configuration_path("training_job") == destination

    assert manager.uninstall("training_job") == destination
    assert not destination.exists()


def test_manager_uses_argv_without_a_shell(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    manager = SupervisorManager("/etc/supervisor/conf.d")

    manager.apply()
    manager.status("training_job")
    manager.restart("training_job")

    assert [call[0] for call in calls] == [
        ["supervisorctl", "reread"],
        ["supervisorctl", "update"],
        ["supervisorctl", "status", "training_job"],
        ["supervisorctl", "restart", "training_job"],
    ]
    assert all(call[1].get("check") is True for call in calls)
    assert all("shell" not in call[1] for call in calls)
