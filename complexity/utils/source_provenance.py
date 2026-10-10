"""Resolve source provenance from the physically installed framework files."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def framework_source_commit() -> str:
    """Return the installed framework checkout's HEAD, or ``unknown``.

    Installed wheels do not carry a verified source commit. Neither the caller's
    working directory nor CI environment variables can supply that evidence.
    Editable checkouts must contain this project's pyproject and own Git root.
    """
    try:
        source_file = Path(__file__).resolve()
        source_root = source_file.parents[2]
        git_metadata = source_root / ".git"
        if not source_file.is_file() or not git_metadata.exists() or git_metadata.is_symlink():
            return "unknown"

        try:
            import tomllib
        except ImportError:  # Python 3.10; tomli is a core dependency.
            import tomli as tomllib

        with (source_root / "pyproject.toml").open("rb") as handle:
            project = tomllib.load(handle).get("project", {})
        if not isinstance(project, dict) or project.get("name") != "complexity-framework":
            return "unknown"

        # Git variables can redirect even an explicit -C lookup to another
        # repository. Resolve the source checkout using its own metadata.
        environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
        top_level = _git_output(source_root, environment, "--show-toplevel")
        if top_level is None or Path(top_level).resolve() != source_root:
            return "unknown"
        commit = _git_output(source_root, environment, "--verify", "HEAD")
        if commit is not None and re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", commit):
            return commit
    except (OSError, ValueError, ImportError, IndexError):
        pass
    return "unknown"


def _git_output(source_root: Path, environment: dict[str, str], *arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", "-C", str(source_root), "rev-parse", *arguments),
            capture_output=True,
            text=True,
            check=False,
            env=environment,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None
