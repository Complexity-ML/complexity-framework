"""Common process lifecycle for framework and CLI operations.

``JobManager`` is deliberately small: a caller describes one command as a
validated :class:`Job`, then chooses foreground execution or detached,
Supervisor-managed execution.  The same abstraction can therefore be used by
Python APIs and Typer commands without duplicating shell and process handling.
"""

from __future__ import annotations

import os
import subprocess
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Literal, cast

from complexity.training.supervisor import AutoRestart, SupervisorManager, SupervisorProgram


@dataclass(frozen=True, slots=True)
class Job:
    """A validated, shell-free command that can run locally or under Supervisor."""

    name: str
    command: tuple[str, ...]
    directory: Path
    log_path: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    autostart: bool = True
    autorestart: AutoRestart = "unexpected"
    startsecs: int = 5
    stopwaitsecs: int = 300

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(self, "log_path", Path(self.log_path))
        object.__setattr__(self, "command", tuple(str(item) for item in self.command))
        object.__setattr__(
            self,
            "environment",
            MappingProxyType({str(key): str(value) for key, value in self.environment.items()}),
        )
        # SupervisorProgram owns the shared security contract: safe names and
        # text, absolute paths, structured argv, and no secret environment.
        self.as_supervisor_program()

    def as_supervisor_program(self) -> SupervisorProgram:
        """Translate the generic job into the current detached backend format."""

        return SupervisorProgram(
            name=self.name,
            command=self.command,
            directory=self.directory,
            stdout_logfile=self.log_path,
            environment=self.environment,
            autostart=self.autostart,
            autorestart=self.autorestart,
            startsecs=self.startsecs,
            stopwaitsecs=self.stopwaitsecs,
        )


@dataclass(frozen=True, slots=True)
class JobHandle:
    """Handle returned for a detached job."""

    name: str
    manager: "JobManager" = field(repr=False, compare=False)

    def status(self) -> str:
        return self.manager.status(self.name)

    def start(self) -> str:
        return self.manager.start(self.name)

    def stop(self) -> str:
        return self.manager.stop(self.name)

    def restart(self) -> str:
        return self.manager.restart(self.name)

    def remove(self, *, missing_ok: bool = False) -> None:
        self.manager.remove(self.name, missing_ok=missing_ok)


class JobManager:
    """Run framework jobs in the foreground or manage them through Supervisor."""

    def __init__(self, supervisor: SupervisorManager | None = None) -> None:
        self.supervisor = SupervisorManager() if supervisor is None else supervisor

    def run(
        self,
        job: Job,
        *,
        detached: bool = False,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes] | JobHandle:
        """Run ``job`` directly, or submit it to Supervisor when detached."""

        if detached:
            return self.submit(job)

        environment = os.environ.copy()
        environment.update(job.environment)
        return subprocess.run(
            list(job.command),
            cwd=job.directory,
            env=environment,
            check=check,
        )

    def submit(self, job: Job) -> JobHandle:
        """Install and apply one detached job without invoking a shell."""

        self.supervisor.install(job.as_supervisor_program())
        self.supervisor.apply()
        return JobHandle(name=job.name, manager=self)

    def list(self) -> str:
        return self._output(self.supervisor.status())

    def status(self, name: str) -> str:
        return self._output(self.supervisor.status(name))

    def start(self, name: str) -> str:
        return self._output(self.supervisor.start(name))

    def stop(self, name: str) -> str:
        return self._output(self.supervisor.stop(name))

    def restart(self, name: str) -> str:
        return self._output(self.supervisor.restart(name))

    def remove(self, name: str, *, missing_ok: bool = False) -> None:
        self.supervisor.uninstall(name, missing_ok=missing_ok)
        self.supervisor.apply()

    def log_path(self, name: str) -> Path:
        """Resolve the log path from a framework-generated Supervisor file."""

        configuration = self.supervisor.configuration_path(name)
        for line in configuration.read_text(encoding="utf-8").splitlines():
            if line.startswith("stdout_logfile="):
                path = Path(line.removeprefix("stdout_logfile="))
                if not path.is_absolute():
                    raise ValueError(f"job {name!r} has a non-absolute log path")
                return path
        raise ValueError(f"job {name!r} has no configured log path")

    def logs(self, name: str, *, lines: int = 100) -> str:
        """Return the last ``lines`` from one managed job without invoking a shell."""

        if lines <= 0:
            raise ValueError("lines must be greater than zero")
        with self.log_path(name).open(encoding="utf-8", errors="replace") as handle:
            return "".join(deque(handle, maxlen=lines))

    def follow_logs(self, name: str, *, lines: int = 100) -> subprocess.CompletedProcess[bytes]:
        """Follow one managed log with structured argv and no shell."""

        if lines <= 0:
            raise ValueError("lines must be greater than zero")
        return subprocess.run(
            ["tail", "-n", str(lines), "-F", str(self.log_path(name))],
            check=False,
        )

    @staticmethod
    def _output(result: subprocess.CompletedProcess[str]) -> str:
        return result.stdout.rstrip("\n")


JobState = Literal[
    "RUNNING",
    "STARTING",
    "STOPPING",
    "STOPPED",
    "BACKOFF",
    "EXITED",
    "FATAL",
    "UNKNOWN",
]


def parse_job_status(line: str) -> tuple[str, JobState, str]:
    """Parse one Supervisor status line for API and colored CLI presentation."""

    parts = line.split(maxsplit=2)
    if not parts:
        return "", "UNKNOWN", ""
    name = parts[0]
    raw_state = parts[1].upper() if len(parts) > 1 else "UNKNOWN"
    known_states = {
        "RUNNING",
        "STARTING",
        "STOPPING",
        "STOPPED",
        "BACKOFF",
        "EXITED",
        "FATAL",
    }
    state = cast(JobState, raw_state) if raw_state in known_states else "UNKNOWN"
    details = parts[2] if len(parts) > 2 else ""
    return name, state, details


__all__ = ["Job", "JobHandle", "JobManager", "JobState", "parse_job_status"]
