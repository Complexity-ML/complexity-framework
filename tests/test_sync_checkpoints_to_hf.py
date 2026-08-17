from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

SCRIPT = Path("scripts/sync_checkpoints_to_hf.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("sync_checkpoints_to_hf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    return _load_module()


def _make_checkpoint(base: Path, name: str, complete: bool = True) -> Path:
    pack_dir = base / name
    pack_dir.mkdir(parents=True)
    if complete:
        (pack_dir / "checkpoint.pt").write_bytes(b"fake")
    return pack_dir


def test_checkpoint_step_parses_trailing_segment(mod, tmp_path) -> None:
    assert mod.checkpoint_step(Path("token_pack_003_18596")) == 18596
    assert mod.checkpoint_step(Path("token_pack_garbage")) == -1


def test_is_complete_checkpoint_requires_checkpoint_file(mod, tmp_path) -> None:
    complete = _make_checkpoint(tmp_path, "token_pack_001_100", complete=True)
    partial = _make_checkpoint(tmp_path, "token_pack_002_200", complete=False)

    assert mod.is_complete_checkpoint(complete) is True
    assert mod.is_complete_checkpoint(partial) is False


def test_sync_once_uploads_new_checkpoints_and_records_state(mod, tmp_path, monkeypatch) -> None:
    _make_checkpoint(tmp_path, "token_pack_001_100")
    _make_checkpoint(tmp_path, "token_pack_002_200")

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub", fake_module
    )

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=5)

    assert fake_api.upload_folder.call_count == 2
    state = json.loads((tmp_path / mod.STATE_FILENAME).read_text())
    assert state["uploaded"] == ["token_pack_001_100", "token_pack_002_200"]


def test_sync_once_never_uploads_a_partially_written_checkpoint(mod, tmp_path, monkeypatch) -> None:
    _make_checkpoint(tmp_path, "token_pack_001_100", complete=True)
    _make_checkpoint(tmp_path, "token_pack_002_200", complete=False)

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub", fake_module
    )

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=5)

    assert fake_api.upload_folder.call_count == 1
    uploaded_paths = [call.kwargs["path_in_repo"] for call in fake_api.upload_folder.call_args_list]
    assert uploaded_paths == ["token_pack_001_100"]


def test_sync_once_prunes_local_copies_beyond_keep_local(mod, tmp_path, monkeypatch) -> None:
    for step in (100, 200, 300):
        _make_checkpoint(tmp_path, f"token_pack_00{step // 100}_{step}")

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub", fake_module
    )

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=1)

    remaining = sorted(p.name for p in tmp_path.glob("token_pack_*"))
    assert remaining == ["token_pack_003_300"]


def test_sync_once_never_prunes_a_checkpoint_that_was_not_uploaded(mod, tmp_path, monkeypatch) -> None:
    _make_checkpoint(tmp_path, "token_pack_001_100")

    def failing_upload_folder(*args, **kwargs):
        raise RuntimeError("network error")

    fake_api = MagicMock()
    fake_api.upload_folder.side_effect = failing_upload_folder
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(
        __import__("sys").modules, "huggingface_hub", fake_module
    )

    with pytest.raises(RuntimeError):
        mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=0)

    assert (tmp_path / "token_pack_001_100").exists()
