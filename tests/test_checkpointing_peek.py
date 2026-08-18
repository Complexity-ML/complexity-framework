"""Tests for peek_latest_checkpoint_step -- reads a checkpoint's step for
computing a resume-time data-loading skip, without loading any model/optimizer
state (see corpus_mixture.PretokenizedCorpusMixtureDataset.resume_skip_rows)."""

from __future__ import annotations

import json

from complexity.utils.checkpointing import peek_latest_checkpoint_step


def _make_checkpoint(base, name, step, *, complete=True, has_state=True):
    ckpt_dir = base / name
    ckpt_dir.mkdir(parents=True)
    if complete:
        (ckpt_dir / "checkpoint.pt").write_bytes(b"fake")
    if has_state:
        (ckpt_dir / "training_state.json").write_text(json.dumps({"step": step}))
    return ckpt_dir


def test_returns_none_for_a_missing_directory(tmp_path):
    assert peek_latest_checkpoint_step(tmp_path / "does_not_exist") is None


def test_returns_none_when_no_checkpoints_exist(tmp_path):
    assert peek_latest_checkpoint_step(tmp_path) is None


def test_picks_the_highest_step_across_mixed_tags(tmp_path):
    _make_checkpoint(tmp_path, "token_pack_001_4133", step=4133)
    _make_checkpoint(tmp_path, "token_pack_002_8265", step=8265)
    _make_checkpoint(tmp_path, "interrupted_5000", step=5000)

    assert peek_latest_checkpoint_step(tmp_path) == 8265


def test_ignores_a_directory_left_partially_written_by_an_interrupted_save(tmp_path):
    _make_checkpoint(tmp_path, "token_pack_001_4133", step=4133)
    # A crash mid-save can leave a directory named for a later step with no
    # checkpoint.pt inside -- must not be treated as the latest checkpoint.
    (tmp_path / "token_pack_002_9999").mkdir()

    assert peek_latest_checkpoint_step(tmp_path) == 4133


def test_falls_back_to_the_directory_name_step_when_training_state_is_missing(tmp_path):
    _make_checkpoint(tmp_path, "token_pack_001_4133", step=4133, has_state=False)

    assert peek_latest_checkpoint_step(tmp_path) == 4133


def test_metadata_checkpoint_without_checkpoint_pt_still_counts_as_complete(tmp_path):
    ckpt_dir = tmp_path / "step_500"
    ckpt_dir.mkdir()
    (ckpt_dir / ".metadata").write_bytes(b"fake")
    (ckpt_dir / "training_state.json").write_text(json.dumps({"step": 500}))

    assert peek_latest_checkpoint_step(tmp_path) == 500
