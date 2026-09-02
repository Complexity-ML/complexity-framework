"""Common process lifecycle for framework and CLI operations.

``JobManager`` is deliberately small: a caller describes one command as a
validated :class:`Job`, then chooses foreground execution or detached,
Supervisor-managed execution.  The same abstraction can therefore be used by
Python APIs and Typer commands without duplicating shell and process handling.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from types import MappingProxyType
from typing import Any, Literal, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

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
    exitcodes: tuple[int, ...] = (0,)
    startsecs: int = 5
    startretries: int = 3
    stopwaitsecs: int = 300
    stdout_logfile_maxbytes: str = "50MB"
    stdout_logfile_backups: int = 3
    priority: int = 999

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
            exitcodes=self.exitcodes,
            startsecs=self.startsecs,
            startretries=self.startretries,
            stopwaitsecs=self.stopwaitsecs,
            stdout_logfile_maxbytes=self.stdout_logfile_maxbytes,
            stdout_logfile_backups=self.stdout_logfile_backups,
            priority=self.priority,
        )


@dataclass(frozen=True, slots=True)
class JobManifest:
    """A validated collection of portable jobs loaded from one TOML file."""

    jobs: tuple[Job, ...]
    source: Path

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("job manifest must contain at least one job")
        names = [job.name for job in self.jobs]
        if len(names) != len(set(names)):
            raise ValueError("job manifest contains duplicate job names")
        object.__setattr__(self, "source", Path(self.source))


_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_JOB_KEYS = {
    "name",
    "command",
    "directory",
    "log_path",
    "environment",
    "autostart",
    "autorestart",
    "exitcodes",
    "startsecs",
    "startretries",
    "stopwaitsecs",
    "stdout_logfile_maxbytes",
    "stdout_logfile_backups",
    "priority",
}


def _format_manifest_value(value: str, variables: Mapping[str, str]) -> str:
    for _literal, field_name, format_spec, conversion in Formatter().parse(value):
        if field_name is None:
            continue
        if not _VARIABLE_NAME.fullmatch(field_name) or format_spec or conversion:
            raise ValueError(f"invalid job manifest placeholder: {field_name!r}")
        if field_name not in variables:
            raise ValueError(f"missing job manifest variable: {field_name}")
    return value.format_map(variables)


def _resolve_manifest_variables(
    raw_variables: Mapping[str, Any], overrides: Mapping[str, str], *, root: Path
) -> dict[str, str]:
    variables = {
        "root": str(root),
        "python": sys.executable,
        "python_dir": str(Path(sys.executable).parent),
    }
    for key, value in raw_variables.items():
        if not _VARIABLE_NAME.fullmatch(str(key)):
            raise ValueError(f"invalid job manifest variable: {key!r}")
        if not isinstance(value, (str, int, float, bool)):
            raise ValueError(f"job manifest variable {key!r} must be scalar")
        variables[str(key)] = str(value)
    for key, value in overrides.items():
        if not _VARIABLE_NAME.fullmatch(str(key)):
            raise ValueError(f"invalid job manifest variable override: {key!r}")
        variables[str(key)] = str(value)

    # Variables may refer to built-ins or earlier variables. Resolve without a
    # shell and stop deterministically if references are cyclic.
    for _ in range(len(variables) + 1):
        updated = {
            key: _format_manifest_value(value, variables)
            for key, value in variables.items()
        }
        if updated == variables:
            unresolved = [
                key
                for key, value in updated.items()
                if any(field_name is not None for _, field_name, _, _ in Formatter().parse(value))
            ]
            if unresolved:
                raise ValueError(
                    "job manifest variables contain a reference cycle: "
                    + ", ".join(sorted(unresolved))
                )
            return updated
        variables = updated
    raise ValueError("job manifest variables contain a reference cycle")


def _manifest_path(value: str, *, root: Path, variables: Mapping[str, str]) -> Path:
    path = Path(_format_manifest_value(value, variables))
    return path if path.is_absolute() else root / path


def load_job_manifest(
    path: str | Path,
    *,
    root: str | Path | None = None,
    variables: Mapping[str, str] | None = None,
) -> JobManifest:
    """Load shell-free Supervisor jobs from a portable TOML manifest."""

    source = Path(path).resolve()
    payload = tomllib.loads(source.read_text(encoding="utf-8"))
    unknown_top_level = set(payload) - {"version", "required_variables", "variables", "jobs"}
    if unknown_top_level:
        raise ValueError(f"unknown job manifest keys: {sorted(unknown_top_level)}")
    if payload.get("version") != 1:
        raise ValueError("job manifest version must be 1")

    resolved_root = Path(root).resolve() if root is not None else source.parent
    overrides = dict(variables or {})
    raw_variables = payload.get("variables", {})
    if not isinstance(raw_variables, dict):
        raise ValueError("job manifest variables must be a table")
    resolved_variables = _resolve_manifest_variables(
        raw_variables,
        overrides,
        root=resolved_root,
    )
    required_variables = payload.get("required_variables", [])
    if not isinstance(required_variables, list) or not all(
        isinstance(name, str) for name in required_variables
    ):
        raise ValueError("required_variables must be an array of names")
    missing = [name for name in required_variables if name not in overrides]
    if missing:
        raise ValueError(f"missing required job manifest variables: {', '.join(missing)}")

    raw_jobs = payload.get("jobs")
    if not isinstance(raw_jobs, list) or not raw_jobs:
        raise ValueError("job manifest jobs must be a non-empty array of tables")
    jobs: list[Job] = []
    for index, raw_job in enumerate(raw_jobs):
        if not isinstance(raw_job, dict):
            raise ValueError(f"jobs[{index}] must be a table")
        unknown_job_keys = set(raw_job) - _JOB_KEYS
        if unknown_job_keys:
            raise ValueError(f"unknown jobs[{index}] keys: {sorted(unknown_job_keys)}")
        try:
            name = _format_manifest_value(str(raw_job["name"]), resolved_variables)
            command_values = raw_job["command"]
            directory_value = str(raw_job.get("directory", "{root}"))
            log_value = str(raw_job["log_path"])
        except KeyError as exc:
            raise ValueError(f"jobs[{index}] is missing {exc.args[0]!r}") from exc
        if not isinstance(command_values, list) or not command_values:
            raise ValueError(f"jobs[{index}].command must be a non-empty array")
        raw_environment = raw_job.get("environment", {})
        if not isinstance(raw_environment, dict):
            raise ValueError(f"jobs[{index}].environment must be a table")
        environment = {
            str(key): _format_manifest_value(str(value), resolved_variables)
            for key, value in raw_environment.items()
        }
        autorestart = raw_job.get("autorestart", "unexpected")
        if autorestart not in (True, False, "unexpected"):
            raise ValueError(f"jobs[{index}].autorestart must be true, false, or 'unexpected'")
        jobs.append(
            Job(
                name=name,
                command=tuple(
                    _format_manifest_value(str(value), resolved_variables)
                    for value in command_values
                ),
                directory=_manifest_path(
                    directory_value,
                    root=resolved_root,
                    variables=resolved_variables,
                ),
                log_path=_manifest_path(
                    log_value,
                    root=resolved_root,
                    variables=resolved_variables,
                ),
                environment=environment,
                autostart=bool(raw_job.get("autostart", True)),
                autorestart=autorestart,
                exitcodes=tuple(int(code) for code in raw_job.get("exitcodes", [0])),
                startsecs=int(raw_job.get("startsecs", 5)),
                startretries=int(raw_job.get("startretries", 3)),
                stopwaitsecs=int(raw_job.get("stopwaitsecs", 300)),
                stdout_logfile_maxbytes=str(
                    raw_job.get("stdout_logfile_maxbytes", "50MB")
                ),
                stdout_logfile_backups=int(raw_job.get("stdout_logfile_backups", 3)),
                priority=int(raw_job.get("priority", 999)),
            )
        )
    return JobManifest(jobs=tuple(jobs), source=source)


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

        return self.submit_many((job,))[0]

    def submit_many(self, jobs: Sequence[Job]) -> tuple[JobHandle, ...]:
        """Install a validated job group and apply Supervisor exactly once."""

        prepared = tuple(jobs)
        if not prepared:
            raise ValueError("jobs must contain at least one job")
        names = [job.name for job in prepared]
        if len(names) != len(set(names)):
            raise ValueError("jobs contain duplicate names")
        programs = tuple(job.as_supervisor_program() for job in prepared)
        for job, program in zip(prepared, programs):
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.supervisor.install(program)
        self.supervisor.apply()
        return tuple(JobHandle(name=job.name, manager=self) for job in prepared)

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

    def follow_logs(self, name: str, *, lines: int = 100) -> Iterator[str]:
        """Yield one managed log with structured argv and no shell."""

        if lines <= 0:
            raise ValueError("lines must be greater than zero")
        process = subprocess.Popen(
            ["tail", "-n", str(lines), "-F", str(self.log_path(name))],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        def iterate() -> Iterator[str]:
            try:
                if process.stdout is None:
                    raise RuntimeError("tail did not expose stdout")
                yield from process.stdout
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

        return iterate()

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


def wait_for_job_artifacts(
    manager: JobManager,
    name: str,
    required_paths: Sequence[str | Path],
    *,
    poll_seconds: float = 30.0,
    timeout_seconds: float | None = None,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    on_poll: Callable[[str], None] | None = None,
) -> None:
    """Wait for a successful managed job exit plus non-empty output artifacts."""

    paths = tuple(Path(path) for path in required_paths)
    if not paths:
        raise ValueError("required_paths must contain at least one path")
    if any(not path.is_absolute() for path in paths):
        raise ValueError("required artifact paths must be absolute")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be greater than zero")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")

    started_at = monotonic()
    while True:
        raw_status = manager.status(name)
        status_name, state, details = parse_job_status(raw_status.splitlines()[0])
        if status_name != name:
            raise RuntimeError(f"Supervisor returned status for {status_name!r}, expected {name!r}")
        complete = all(path.is_file() and path.stat().st_size > 0 for path in paths)
        if state == "EXITED":
            if complete:
                return
            missing = ", ".join(str(path) for path in paths if not path.is_file() or path.stat().st_size <= 0)
            raise RuntimeError(f"job {name!r} exited without required artifacts: {missing}")
        if state == "FATAL":
            raise RuntimeError(f"job {name!r} entered FATAL state: {details}")
        if timeout_seconds is not None and monotonic() - started_at >= timeout_seconds:
            raise TimeoutError(f"timed out waiting for job {name!r}")
        if on_poll is not None:
            on_poll(f"job={name} state={state} artifacts_ready={complete}")
        sleep(poll_seconds)


__all__ = [
    "Job",
    "JobHandle",
    "JobManager",
    "JobManifest",
    "JobState",
    "load_job_manifest",
    "parse_job_status",
    "wait_for_job_artifacts",
]
