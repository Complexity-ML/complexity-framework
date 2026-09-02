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


def test_load_hf_token_prefers_a_protected_file(mod, tmp_path, monkeypatch) -> None:
    token_file = tmp_path / ".hf_token"
    token_file.write_text("file-token\n")
    token_file.chmod(0o600)
    monkeypatch.setenv("TEST_HF_TOKEN", "environment-token")

    assert mod.load_hf_token(token_file, "TEST_HF_TOKEN") == "file-token"


def test_load_hf_token_rejects_group_or_world_permissions(mod, tmp_path) -> None:
    token_file = tmp_path / ".hf_token"
    token_file.write_text("secret")
    token_file.chmod(0o644)

    with pytest.raises(SystemExit, match="must not be group/world accessible"):
        mod.load_hf_token(token_file, "TEST_HF_TOKEN")


def test_load_hf_token_falls_back_to_environment(mod, monkeypatch) -> None:
    monkeypatch.setenv("TEST_HF_TOKEN", "environment-token")

    assert mod.load_hf_token(None, "TEST_HF_TOKEN") == "environment-token"


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
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=5)

    assert fake_api.upload_folder.call_count == 2
    state = json.loads((tmp_path / mod.STATE_FILENAME).read_text())
    assert state["uploaded"] == ["token_pack_001_100", "token_pack_002_200"]
    assert state["destinations"] == {
        "token_pack_001_100": "token_pack_001_100",
        "token_pack_002_200": "token_pack_002_200",
    }


def test_sync_once_supports_a_repository_path_prefix(mod, tmp_path, monkeypatch) -> None:
    _make_checkpoint(tmp_path, "step_001563")
    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(
        tmp_path,
        "org/repo",
        token="fake",
        private=False,
        keep_local=3,
        path_prefix="training/sft-v2-300k/checkpoints/",
    )

    assert fake_api.upload_folder.call_args.kwargs["path_in_repo"] == (
        "training/sft-v2-300k/checkpoints/step_001563"
    )


def test_sync_once_can_publish_epoch_boundaries_with_short_names(
    mod, tmp_path, monkeypatch
) -> None:
    _make_checkpoint(tmp_path, "step_003040")
    _make_checkpoint(tmp_path, "step_006080")
    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(
        tmp_path,
        "org/repo",
        token="fake",
        private=False,
        keep_local=3,
        path_prefix="checkpoints",
        steps_per_epoch=3040,
    )

    uploaded_paths = [
        call.kwargs["path_in_repo"] for call in fake_api.upload_folder.call_args_list
    ]
    assert uploaded_paths == ["checkpoints/epoch_1", "checkpoints/epoch_2"]


def test_repository_checkpoint_name_keeps_non_boundary_checkpoints(mod) -> None:
    assert [
        mod.repository_checkpoint_name(Path(f"step_{step:06d}"), 3040)
        for step in (3040, 6080, 9120)
    ] == ["epoch_1", "epoch_2", "epoch_3"]
    assert mod.repository_checkpoint_name(Path("step_003041"), 3040) == "step_003041"
    assert mod.repository_checkpoint_name(Path("interrupted_003040"), 3040) == (
        "interrupted_003040"
    )


def test_legacy_state_is_reuploaded_when_repository_layout_changes(
    mod, tmp_path, monkeypatch
) -> None:
    _make_checkpoint(tmp_path, "step_003040")
    (tmp_path / mod.STATE_FILENAME).write_text(
        json.dumps({"uploaded": ["step_003040"]}), encoding="utf-8"
    )
    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(
        tmp_path,
        "org/repo",
        token="fake",
        private=False,
        keep_local=3,
        path_prefix="checkpoints",
        steps_per_epoch=3040,
    )

    assert fake_api.upload_folder.call_count == 1
    assert fake_api.upload_folder.call_args.kwargs["path_in_repo"] == "checkpoints/epoch_1"
    state = json.loads((tmp_path / mod.STATE_FILENAME).read_text())
    assert state["destinations"] == {"step_003040": "checkpoints/epoch_1"}


def test_current_destination_is_not_uploaded_twice(mod, tmp_path, monkeypatch) -> None:
    _make_checkpoint(tmp_path, "step_003040")
    (tmp_path / mod.STATE_FILENAME).write_text(
        json.dumps(
            {
                "uploaded": ["step_003040"],
                "destinations": {"step_003040": "checkpoints/epoch_1"},
            }
        ),
        encoding="utf-8",
    )
    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(
        tmp_path,
        "org/repo",
        token="fake",
        private=False,
        keep_local=3,
        path_prefix="checkpoints",
        steps_per_epoch=3040,
    )

    assert fake_api.upload_folder.call_count == 0


def test_sync_once_never_uploads_a_partially_written_checkpoint(mod, tmp_path, monkeypatch) -> None:
    _make_checkpoint(tmp_path, "token_pack_001_100", complete=True)
    _make_checkpoint(tmp_path, "token_pack_002_200", complete=False)

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

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
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=1)

    remaining = sorted(p.name for p in tmp_path.glob("token_pack_*"))
    assert remaining == ["token_pack_003_300"]


def test_sync_once_backs_up_final_and_interrupted_checkpoints_too(
    mod, tmp_path, monkeypatch
) -> None:
    """Regression guard: the sync used to glob only token_pack_*, so a run
    that finished cleanly (final_N) or crashed (interrupted_N) never got its
    last checkpoint backed up to HF."""
    _make_checkpoint(tmp_path, "token_pack_001_100")
    _make_checkpoint(tmp_path, "final_247946")
    _make_checkpoint(tmp_path, "interrupted_212")

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=5)

    uploaded_names = {call.kwargs["path_in_repo"] for call in fake_api.upload_folder.call_args_list}
    assert uploaded_names == {"token_pack_001_100", "final_247946", "interrupted_212"}


def test_sync_once_ignores_the_tensorboard_directory(mod, tmp_path, monkeypatch) -> None:
    (tmp_path / "tensorboard").mkdir()

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=5)

    assert fake_api.upload_folder.call_count == 0


def test_sync_once_backs_up_the_plain_final_export_too(mod, tmp_path, monkeypatch) -> None:
    """Regression guard: runner.py's plain "final" HF export (model.safetensors,
    no checkpoint.pt) used to be invisible to is_complete_checkpoint, so it
    never got backed up to HF even though it's the run's authoritative
    completed-training artifact."""
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "model.safetensors").write_bytes(b"weights")

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=5)

    uploaded_names = {call.kwargs["path_in_repo"] for call in fake_api.upload_folder.call_args_list}
    assert uploaded_names == {"final"}


def test_sync_once_never_prunes_the_final_export_even_when_oldest_by_sort(
    mod, tmp_path, monkeypatch
) -> None:
    """Regression guard: checkpoint_step("final") returns -1 (no numeric
    suffix), which would sort it as the OLDEST checkpoint and make
    keep_local pruning delete it first — the one directory that must never
    be deleted here (cleanup_tr_hash_200m_checkpoints.py owns its lifecycle,
    and refuses to run at all if it's missing)."""
    final_dir = tmp_path / "final"
    final_dir.mkdir()
    (final_dir / "model.safetensors").write_bytes(b"weights")
    _make_checkpoint(tmp_path, "token_pack_001_100")
    _make_checkpoint(tmp_path, "token_pack_002_200")

    fake_api = MagicMock()
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=1)

    assert final_dir.is_dir()
    assert (final_dir / "model.safetensors").read_bytes() == b"weights"
    remaining_packs = sorted(p.name for p in tmp_path.glob("token_pack_*"))
    assert remaining_packs == ["token_pack_002_200"]


def test_sync_once_never_prunes_a_checkpoint_that_was_not_uploaded(
    mod, tmp_path, monkeypatch
) -> None:
    _make_checkpoint(tmp_path, "token_pack_001_100")

    def failing_upload_folder(*args, **kwargs):
        raise RuntimeError("network error")

    fake_api = MagicMock()
    fake_api.upload_folder.side_effect = failing_upload_folder
    fake_module = type(
        "hub", (), {"HfApi": lambda token=None: fake_api, "create_repo": MagicMock()}
    )
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", fake_module)

    with pytest.raises(RuntimeError):
        mod.sync_once(tmp_path, "org/repo", token="fake", private=True, keep_local=0)

    assert (tmp_path / "token_pack_001_100").exists()


def test_run_pass_with_timeout_kills_a_stalled_pass_and_returns_false(mod, tmp_path) -> None:
    # Simulates a network drop mid-upload: the real huggingface_hub call blocks
    # forever on a socket read with no exception ever raised, so this has to be
    # a real subprocess that actually hangs, not a mock -- the fix under test
    # is specifically that subprocess.run(timeout=...) kills it.
    stuck_script = tmp_path / "stuck.py"
    stuck_script.write_text("import time\ntime.sleep(60)\n")

    import sys

    ok = mod.run_pass_with_timeout([sys.executable, str(stuck_script)], pass_timeout=0.5)

    assert ok is False


def test_run_pass_with_timeout_returns_true_for_a_clean_pass(mod, tmp_path) -> None:
    fast_script = tmp_path / "fast.py"
    fast_script.write_text("print('done')\n")

    import sys

    ok = mod.run_pass_with_timeout([sys.executable, str(fast_script)], pass_timeout=10.0)

    assert ok is True


def test_run_pass_with_timeout_returns_false_for_a_nonzero_exit(mod, tmp_path) -> None:
    failing_script = tmp_path / "failing.py"
    failing_script.write_text("import sys\nsys.exit(1)\n")

    import sys

    ok = mod.run_pass_with_timeout([sys.executable, str(failing_script)], pass_timeout=10.0)

    assert ok is False


def test_transport_fallback_retries_failed_xet_pass_over_http(mod, monkeypatch) -> None:
    calls: list[tuple[float, dict[str, str]]] = []

    def fake_run(_args, timeout, *, environment=None):
        calls.append((timeout, dict(environment or {})))
        return len(calls) == 2

    monkeypatch.setattr(mod, "run_pass_with_timeout", fake_run)
    monkeypatch.setattr(mod, "xet_is_available", lambda _environment: True)

    assert mod.run_pass_with_transport_fallback(
        ["sync"],
        pass_timeout=1800.0,
        xet_timeout=300.0,
        environment={"HF_TOKEN": "secret"},
    )
    assert calls == [
        (300.0, {"HF_TOKEN": "secret"}),
        (1800.0, {"HF_TOKEN": "secret", "HF_HUB_DISABLE_XET": "1"}),
    ]


def test_transport_fallback_does_not_duplicate_http_only_pass(mod, monkeypatch) -> None:
    calls: list[tuple[float, dict[str, str]]] = []

    def fake_run(_args, timeout, *, environment=None):
        calls.append((timeout, dict(environment or {})))
        return False

    monkeypatch.setattr(mod, "run_pass_with_timeout", fake_run)
    monkeypatch.setattr(mod, "xet_is_available", lambda _environment: False)

    assert not mod.run_pass_with_transport_fallback(
        ["sync"],
        pass_timeout=1800.0,
        xet_timeout=300.0,
        environment={"HF_HUB_DISABLE_XET": "1"},
    )
    assert calls == [(1800.0, {"HF_HUB_DISABLE_XET": "1"})]


def test_transport_fallback_uses_http_during_bounded_xet_cooldown(
    mod, monkeypatch
) -> None:
    calls: list[tuple[float, dict[str, str]]] = []
    now = {"value": 100.0}
    outcomes = iter((False, True, True, True))

    def fake_run(_args, timeout, *, environment=None):
        calls.append((timeout, dict(environment or {})))
        return next(outcomes)

    monkeypatch.setattr(mod, "run_pass_with_timeout", fake_run)
    monkeypatch.setattr(mod.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(mod.time, "monotonic", lambda: now["value"])

    assert mod.run_pass_with_transport_fallback(
        ["sync"],
        pass_timeout=1800.0,
        xet_timeout=300.0,
        xet_cooldown=900.0,
        environment={},
    )
    assert mod.run_pass_with_transport_fallback(
        ["sync"],
        pass_timeout=1800.0,
        xet_timeout=300.0,
        xet_cooldown=900.0,
        environment={},
    )
    now["value"] = 1001.0
    assert mod.run_pass_with_transport_fallback(
        ["sync"],
        pass_timeout=1800.0,
        xet_timeout=300.0,
        xet_cooldown=900.0,
        environment={},
    )

    assert calls == [
        (300.0, {}),
        (1800.0, {"HF_HUB_DISABLE_XET": "1"}),
        (1800.0, {"HF_HUB_DISABLE_XET": "1"}),
        (300.0, {}),
    ]
