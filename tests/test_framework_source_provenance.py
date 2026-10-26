"""Installed tools must never attribute a consumer's commit to the framework."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from complexity.utils import source_provenance
from complexity.vision.detection.evaluation import _framework_commit
from complexity.vision.detection.onnx_quantization import framework_commit as quantization_commit
from complexity.vision.detection.onnx_release import framework_commit as release_commit

READERS = (
    source_provenance.framework_source_commit,
    _framework_commit,
    quantization_commit,
    release_commit,
)


def _git(directory: Path, *arguments: str) -> str:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(
        GIT_AUTHOR_NAME="Framework provenance test",
        GIT_AUTHOR_EMAIL="test@example.invalid",
        GIT_COMMITTER_NAME="Framework provenance test",
        GIT_COMMITTER_EMAIL="test@example.invalid",
    )
    return subprocess.check_output(
        ("git", "-C", str(directory), *arguments),
        env=environment,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def _repository(directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "fixture.txt").write_text(directory.name, encoding="utf-8")
    _git(directory, "init", "-q")
    _git(directory, "add", ".")
    _git(directory, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "fixture")
    return _git(directory, "rev-parse", "HEAD")


def _source_file(directory: Path, *, pyproject: str | None = None) -> Path:
    source = directory / "complexity/utils/source_provenance.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("# fixture source\n", encoding="utf-8")
    if pyproject is not None:
        (directory / "pyproject.toml").write_text(pyproject, encoding="utf-8")
    return source


@pytest.mark.parametrize("reader", READERS, ids=("helper", "evaluation", "quantization", "release"))
@pytest.mark.parametrize("ci_commit", ("invalid", "consumer", "framework"))
def test_readers_use_their_own_checkout_from_a_foreign_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader, ci_commit: str
) -> None:
    source = _source_file(
        tmp_path / "framework", pyproject='[project]\nname = "complexity-framework"\n'
    )
    expected = _repository(tmp_path / "framework")
    consumer_commit = _repository(tmp_path / "consumer")
    assert expected != consumer_commit
    monkeypatch.setattr(source_provenance, "__file__", str(source))
    monkeypatch.chdir(tmp_path / "consumer")
    monkeypatch.setenv(
        "GITHUB_SHA", {"invalid": "not-a-commit", "consumer": consumer_commit, "framework": expected}[ci_commit]
    )
    assert reader() == expected


@pytest.mark.parametrize("reader", READERS, ids=("helper", "evaluation", "quantization", "release"))
def test_wheel_source_without_project_metadata_never_uses_the_consumers_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reader
) -> None:
    consumer_commit = _repository(tmp_path / "consumer")
    source = _source_file(tmp_path / "wheel-site-packages")
    monkeypatch.setattr(source_provenance, "__file__", str(source))
    monkeypatch.chdir(tmp_path / "consumer")
    monkeypatch.setenv("GITHUB_SHA", consumer_commit)
    assert reader() == "unknown"


def test_non_git_source_and_a_nested_source_in_an_unrelated_repo_are_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_file(
        tmp_path / "consumer/framework", pyproject='[project]\nname = "complexity-framework"\n'
    )
    _repository(tmp_path / "consumer")
    monkeypatch.setattr(source_provenance, "__file__", str(source))
    monkeypatch.chdir(tmp_path / "consumer")
    assert source_provenance.framework_source_commit() == "unknown"
    # Even when a local .git marker exists, an ancestor's top-level is not the
    # installed framework root and cannot establish framework provenance.
    (source.parents[2] / ".git").symlink_to(tmp_path / "consumer/.git", target_is_directory=True)
    assert source_provenance.framework_source_commit() == "unknown"


@pytest.mark.parametrize(
    "pyproject", (None, "malformed = [", '[project]\nname = "another-project"\n')
)
def test_source_metadata_must_identify_the_framework(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pyproject: str | None
) -> None:
    source = _source_file(tmp_path / "framework", pyproject=pyproject)
    _repository(tmp_path / "framework")
    monkeypatch.setattr(source_provenance, "__file__", str(source))
    assert source_provenance.framework_source_commit() == "unknown"


def test_git_environment_cannot_redirect_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_file(
        tmp_path / "framework", pyproject='[project]\nname = "complexity-framework"\n'
    )
    expected = _repository(tmp_path / "framework")
    _repository(tmp_path / "consumer")
    monkeypatch.setattr(source_provenance, "__file__", str(source))
    monkeypatch.chdir(tmp_path / "consumer")
    monkeypatch.setenv("GIT_DIR", str(tmp_path / "consumer/.git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(tmp_path / "consumer"))
    assert source_provenance.framework_source_commit() == expected


def test_physical_source_path_is_used_through_a_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_file(
        tmp_path / "framework", pyproject='[project]\nname = "complexity-framework"\n'
    )
    expected = _repository(tmp_path / "framework")
    link = tmp_path / "consumer-link.py"
    link.symlink_to(source)
    monkeypatch.setattr(source_provenance, "__file__", str(link))
    assert source_provenance.framework_source_commit() == expected


def test_unavailable_git_does_not_invent_a_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source_file(
        tmp_path / "framework", pyproject='[project]\nname = "complexity-framework"\n'
    )
    _repository(tmp_path / "framework")
    monkeypatch.setattr(source_provenance, "__file__", str(source))
    monkeypatch.setenv("PATH", str(tmp_path / "missing-binaries"))
    assert source_provenance.framework_source_commit() == "unknown"


def test_actual_editable_checkout_is_resolved_even_from_a_foreign_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = Path(source_provenance.__file__).resolve().parents[2]
    expected = _git(source_root, "rev-parse", "HEAD")
    _repository(tmp_path / "consumer")
    monkeypatch.chdir(tmp_path / "consumer")
    monkeypatch.setenv("GITHUB_SHA", "0" * 40)
    assert source_provenance.framework_source_commit() == expected
