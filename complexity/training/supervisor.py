"""Validated Supervisor integration for long-running training jobs.

The framework owns this configuration contract instead of committing a growing
set of instance-specific ``.conf`` files. Commands are represented as argv,
rendered deterministically, and never passed through a shell by this module.
Secrets are rejected from the rendered environment: instance credentials must
come from protected local state.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

_PROGRAM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ENVIRONMENT_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SECRET_NAME = re.compile(r"(?:^|_)(?:TOKEN|PASSWORD|PASSWD|SECRET|API_KEY|PRIVATE_KEY)(?:$|_)")
_FORBIDDEN_TEXT = ("\x00", "\n", "\r")

AutoRestart = bool | Literal["unexpected"]


def _require_safe_text(value: str, *, label: str) -> str:
    if not value or any(character in value for character in _FORBIDDEN_TEXT):
        raise ValueError(f"{label} must be non-empty and contain no NUL/newline")
    return value


def _supervisor_bool(value: bool) -> str:
    return "true" if value else "false"


def _quote_environment(value: str) -> str:
    escaped = value.replace("%", "%%").replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


@dataclass(frozen=True, slots=True)
class SupervisorProgram:
    """Portable, validated description of one Supervisor-managed process."""

    name: str
    command: tuple[str, ...]
    directory: Path
    stdout_logfile: Path
    environment: Mapping[str, str] = field(default_factory=dict)
    autostart: bool = True
    autorestart: AutoRestart = "unexpected"
    startsecs: int = 5
    stopwaitsecs: int = 300
    stopsignal: str = "TERM"
    stopasgroup: bool = True
    killasgroup: bool = True
    redirect_stderr: bool = True
    stdout_logfile_maxbytes: str = "0"
    stdout_logfile_backups: int = 0
    priority: int = 999

    def __post_init__(self) -> None:
        if not _PROGRAM_NAME.fullmatch(self.name):
            raise ValueError(f"invalid Supervisor program name: {self.name!r}")
        if not self.command:
            raise ValueError("command must contain at least one argv item")
        normalized_command = tuple(
            _require_safe_text(str(argument), label=f"command[{index}]")
            for index, argument in enumerate(self.command)
        )
        if not Path(self.directory).is_absolute():
            raise ValueError("directory must be an absolute path")
        if not Path(self.stdout_logfile).is_absolute():
            raise ValueError("stdout_logfile must be an absolute path")
        if self.autorestart not in (True, False, "unexpected"):
            raise ValueError("autorestart must be true, false, or 'unexpected'")
        if self.startsecs < 0 or self.stopwaitsecs < 0:
            raise ValueError("startsecs and stopwaitsecs must be non-negative")
        if self.stdout_logfile_backups < 0 or self.priority < 0:
            raise ValueError("stdout_logfile_backups and priority must be non-negative")
        _require_safe_text(str(self.stdout_logfile_maxbytes), label="stdout_logfile_maxbytes")
        if not re.fullmatch(r"[A-Z][A-Z0-9]*", self.stopsignal):
            raise ValueError(f"invalid stop signal: {self.stopsignal!r}")

        normalized_environment: dict[str, str] = {}
        for key, raw_value in self.environment.items():
            if not _ENVIRONMENT_NAME.fullmatch(key):
                raise ValueError(f"invalid environment name: {key!r}")
            if _SECRET_NAME.search(key.upper()):
                raise ValueError(f"secret environment variable is forbidden: {key}")
            normalized_environment[key] = _require_safe_text(
                str(raw_value), label=f"environment[{key}]"
            )
        object.__setattr__(self, "directory", Path(self.directory))
        object.__setattr__(self, "stdout_logfile", Path(self.stdout_logfile))
        object.__setattr__(self, "command", normalized_command)
        object.__setattr__(self, "environment", normalized_environment)

    def render(self) -> str:
        """Render a deterministic Supervisor INI fragment."""

        autorestart = (
            self.autorestart
            if self.autorestart == "unexpected"
            else _supervisor_bool(self.autorestart)
        )
        lines = [
            f"[program:{self.name}]",
            f"directory={self.directory}",
            f"command={shlex.join(self.command)}",
            f"autostart={_supervisor_bool(self.autostart)}",
            f"autorestart={autorestart}",
            f"startsecs={self.startsecs}",
            f"stopasgroup={_supervisor_bool(self.stopasgroup)}",
            f"killasgroup={_supervisor_bool(self.killasgroup)}",
            f"stopsignal={self.stopsignal}",
            f"stopwaitsecs={self.stopwaitsecs}",
            f"redirect_stderr={_supervisor_bool(self.redirect_stderr)}",
            f"stdout_logfile={self.stdout_logfile}",
            f"stdout_logfile_maxbytes={self.stdout_logfile_maxbytes}",
            f"stdout_logfile_backups={self.stdout_logfile_backups}",
            f"priority={self.priority}",
        ]
        if self.environment:
            environment = ",".join(
                f"{key}={_quote_environment(value)}"
                for key, value in sorted(self.environment.items())
            )
            lines.append(f"environment={environment}")
        return "\n".join(lines) + "\n"


class SupervisorManager:
    """Install programs atomically and control them through ``supervisorctl``."""

    def __init__(
        self,
        config_directory: str | Path = "/etc/supervisor/conf.d",
        *,
        supervisorctl: Sequence[str] = ("supervisorctl",),
    ) -> None:
        self.config_directory = Path(config_directory)
        if not self.config_directory.is_absolute():
            raise ValueError("config_directory must be an absolute path")
        if not supervisorctl:
            raise ValueError("supervisorctl argv must not be empty")
        self.supervisorctl = tuple(
            _require_safe_text(str(item), label="supervisorctl argv") for item in supervisorctl
        )

    def install(self, program: SupervisorProgram) -> Path:
        """Atomically install one generated configuration with mode ``0600``."""

        self.config_directory.mkdir(parents=True, exist_ok=True)
        destination = self.configuration_path(program.name)
        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.config_directory,
                prefix=f".{program.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_name = handle.name
                handle.write(program.render())
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_name, 0o600)
            os.replace(temporary_name, destination)
        finally:
            if temporary_name is not None and os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return destination

    def uninstall(self, name: str, *, missing_ok: bool = False) -> Path:
        """Remove exactly one validated generated configuration."""

        destination = self.configuration_path(name)
        destination.unlink(missing_ok=missing_ok)
        return destination

    def configuration_path(self, name: str) -> Path:
        """Return the exact generated configuration path for one validated job."""

        _validate_program_name(name)
        return self.config_directory / f"{name}.conf"

    def _control(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.supervisorctl, *arguments],
            check=True,
            capture_output=True,
            text=True,
        )

    def apply(self) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str]]:
        """Make Supervisor discover and apply newly installed configurations."""

        return self._control("reread"), self._control("update")

    def status(self, name: str | None = None) -> subprocess.CompletedProcess[str]:
        if name is None:
            return self._control("status")
        _validate_program_name(name)
        return self._control("status", name)

    def start(self, name: str) -> subprocess.CompletedProcess[str]:
        return self._named_control("start", name)

    def stop(self, name: str) -> subprocess.CompletedProcess[str]:
        return self._named_control("stop", name)

    def restart(self, name: str) -> subprocess.CompletedProcess[str]:
        return self._named_control("restart", name)

    def _named_control(self, action: str, name: str) -> subprocess.CompletedProcess[str]:
        _validate_program_name(name)
        return self._control(action, name)


def _validate_program_name(name: str) -> None:
    if not _PROGRAM_NAME.fullmatch(name):
        raise ValueError(f"invalid Supervisor program name: {name!r}")


__all__ = [
    "AutoRestart",
    "SupervisorManager",
    "SupervisorProgram",
]
