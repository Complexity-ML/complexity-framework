from __future__ import annotations

import itertools
import json
import os
import shutil
from pathlib import Path

import pytest
import torch

from complexity.training import (
    PretokenizedCorpusMixtureDataset,
    TextCorpusSource,
    WeightedStreamingTextDataset,
    allocate_weighted_counts,
)
from scripts.build_corrective_replay_plan import build_corrective_replay_plan
from scripts.build_tr_hash_70b_replay_plan import (
    DEFAULT_REPLAY_PASSES,
    DEFAULT_UNIQUE_BUDGETS,
    build_replay_plan,
)
from scripts.tokenize_tr_hash_200m_200b import (
    DEFAULT_HF_REPO,
    TokenShardWriter,
    resolve_layout,
    upload_dataset_subset,
    write_mixture_manifest,
)


class _Tokenizer:
    eos_token_id = 0

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(token) for token in text.split()]


def test_each_corpus_is_losslessly_packed_in_its_own_buffer() -> None:
    sources = (
        TextCorpusSource("a", 0.5, dataset_id="unused"),
        TextCorpusSource("b", 0.5, dataset_id="unused"),
    )
    streams = {
        "a": itertools.repeat({"text": "1 2 3 4"}),
        "b": itertools.repeat({"text": "7 8 9 10"}),
    }
    dataset = WeightedStreamingTextDataset(
        tokenizer=_Tokenizer(), seq_len=4, sources=sources, streams=streams
    )

    samples = list(itertools.islice(dataset, 4))

    assert torch.equal(samples[0]["input_ids"], torch.tensor([1, 2, 3, 4]))
    assert torch.equal(samples[1]["input_ids"], torch.tensor([7, 8, 9, 10]))
    assert all(sample["input_ids"].shape == (4,) for sample in samples)


@pytest.mark.parametrize(
    "sources,match",
    (
        ((TextCorpusSource("a", 0.9, dataset_id="a"),), "sum to 1.0"),
        (
            (
                TextCorpusSource("a", 0.5, dataset_id="a"),
                TextCorpusSource("a", 0.5, dataset_id="b"),
            ),
            "unique",
        ),
        ((TextCorpusSource("a", 1.0),), "exactly one"),
    ),
)
def test_invalid_mixtures_fail_before_loading_remote_data(sources, match) -> None:
    with pytest.raises(ValueError, match=match):
        WeightedStreamingTextDataset(
            tokenizer=_Tokenizer(), seq_len=4, sources=sources
        )


def test_weighted_count_allocation_conserves_every_sequence() -> None:
    sources = (
        TextCorpusSource("general", 0.45, dataset_id="general"),
        TextCorpusSource("edu", 0.30, dataset_id="edu"),
        TextCorpusSource("code", 0.10, dataset_id="code"),
        TextCorpusSource("math_a", 0.05, dataset_id="math_a"),
        TextCorpusSource("math_b", 0.05, dataset_id="math_b"),
        TextCorpusSource("synthetic", 0.05, dataset_id="synthetic"),
    )
    total_rows, actual_tokens, counts = resolve_layout(
        target_tokens=200_000_000_000,
        seq_len=1024,
        global_batch_sequences=512,
        sources=sources,
    )

    assert total_rows == 195_312_640
    assert actual_tokens == 200_000_143_360
    assert sum(counts.values()) == total_rows
    assert counts == allocate_weighted_counts(total_rows, sources)
    assert all(count % 32 == 0 for count in counts.values())


def test_uint16_shards_round_trip_through_pretokenized_reader(tmp_path) -> None:
    source = TextCorpusSource("tiny", 1.0, data_files="unused.jsonl")
    source_root = tmp_path / "corpora" / source.name
    writer = TokenShardWriter(
        source_root,
        seq_len=4,
        total_rows=5,
        rows_per_shard=2,
    )
    assert writer.feed(torch.arange(21, dtype=torch.int64).numpy()) == 21
    source_manifest = writer.write_manifest(source=source)
    source_metadata = json.loads(source_manifest.read_text())
    assert source_metadata["trained_tokens"] == 20
    assert all(shard["bytes"] == shard["tokens"] * 2 for shard in source_metadata["shards"])
    assert all(len(shard["sha256"]) == 64 for shard in source_metadata["shards"])

    write_mixture_manifest(
        output_root=tmp_path,
        sources=(source,),
        seq_len=4,
        requested_tokens=20,
        actual_tokens=20,
        global_batch_sequences=1,
        rows_by_source={"tiny": 5},
    )
    samples = list(PretokenizedCorpusMixtureDataset(tmp_path))

    assert len(samples) == 5
    for index, sample in enumerate(samples):
        assert torch.equal(
            sample["input_ids"], torch.arange(index * 4, index * 4 + 4)
        )
        assert torch.equal(
            sample["labels"], torch.arange(index * 4 + 1, index * 4 + 5)
        )


def _remote_token_fixture(tmp_path: Path):
    remote = tmp_path / "remote"
    source = TextCorpusSource("tiny", 1.0, data_files="unused.jsonl")
    writer = TokenShardWriter(
        remote / "corpora" / source.name,
        seq_len=4,
        total_rows=5,
        rows_per_shard=2,
    )
    writer.feed(torch.arange(21, dtype=torch.int64).numpy())
    writer.write_manifest(source=source)
    write_mixture_manifest(
        output_root=remote,
        sources=(source,),
        seq_len=4,
        requested_tokens=20,
        actual_tokens=20,
        global_batch_sequences=1,
        rows_by_source={"tiny": 5},
    )
    files = {
        str(path.relative_to(remote))
        for path in remote.rglob("*")
        if path.is_file()
    }
    downloads: list[str] = []

    def download(*, filename, local_dir, **_kwargs):
        downloads.append(filename)
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote / filename, destination)
        return str(destination)

    def list_files(**_kwargs):
        return sorted(files)

    return remote, files, downloads, download, list_files


def test_list_files_retries_a_transient_dns_failure(tmp_path, monkeypatch) -> None:
    """Regression guard: a 10-rank run bursts simultaneous DNS lookups for
    the same host on startup, and one dropping crashed the whole
    distributed job (then looped under autorestart) over a failure that
    clears on its own within a couple seconds."""
    from complexity.training.corpus_mixture import _HubShardCache

    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    attempts = {"count": 0}

    def flaky_list_files(**_kwargs):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise OSError("[Errno -3] Temporary failure in name resolution")
        return ["a.bin", "b.bin"]

    cache = _HubShardCache(
        repo_id="owner/repo",
        cache_dir=tmp_path,
        revision="main",
        token=None,
        max_cache_bytes=10**9,
        prefetch_shards=0,
        file_lister=flaky_list_files,
    )

    assert cache.list_files() == {"a.bin", "b.bin"}
    assert attempts["count"] == 3


def test_list_files_gives_up_after_max_attempts(tmp_path, monkeypatch) -> None:
    from complexity.training.corpus_mixture import _HubShardCache

    monkeypatch.setattr("time.sleep", lambda _seconds: None)

    def always_fails(**_kwargs):
        raise OSError("[Errno -3] Temporary failure in name resolution")

    cache = _HubShardCache(
        repo_id="owner/repo",
        cache_dir=tmp_path,
        revision="main",
        token=None,
        max_cache_bytes=10**9,
        prefetch_shards=0,
        file_lister=always_fails,
    )

    with pytest.raises(OSError, match="name resolution"):
        cache.list_files(max_attempts=3)


def test_hub_download_falls_back_to_http_after_xet_transport_failure(
    tmp_path, monkeypatch
) -> None:
    from complexity.training.corpus_mixture import _HubShardCache

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    attempts: list[str | None] = []

    def flaky_download(*, filename, local_dir, **_kwargs):
        attempts.append(os.environ.get("HF_HUB_DISABLE_XET"))
        if len(attempts) == 1:
            raise RuntimeError(
                "File reconstruction error: CAS Client Error: Request middleware error"
            )
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"tokens")
        return str(destination)

    cache = _HubShardCache(
        repo_id="owner/repo",
        cache_dir=tmp_path,
        revision="main",
        token=None,
        max_cache_bytes=10**9,
        prefetch_shards=0,
        downloader=flaky_download,
    )

    downloaded = cache._download("corpora/source/tokens-00000.bin")

    assert downloaded.read_bytes() == b"tokens"
    assert attempts == [None, "1"]
    assert "HF_HUB_DISABLE_XET" not in os.environ

    second = cache._download("corpora/source/tokens-00001.bin")
    assert second.read_bytes() == b"tokens"
    assert attempts == [None, "1", "1"]


def test_hub_download_does_not_hide_non_transient_errors(tmp_path) -> None:
    from complexity.training.corpus_mixture import _HubShardCache

    attempts = 0

    def rejected_download(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise PermissionError("repository access denied")

    cache = _HubShardCache(
        repo_id="owner/repo",
        cache_dir=tmp_path,
        revision="main",
        token=None,
        max_cache_bytes=10**9,
        prefetch_shards=0,
        downloader=rejected_download,
    )

    with pytest.raises(PermissionError, match="access denied"):
        cache._download("corpora/source/tokens-00000.bin")
    assert attempts == 1


@pytest.mark.skipif(os.name != "posix", reason="shared shard pins use POSIX flock")
def test_hub_cache_allows_concurrent_readers_but_blocks_eviction(tmp_path) -> None:
    from complexity.training.corpus_mixture import _HubShardCache

    pin_path = tmp_path / "pins" / "shard.lock"
    with _HubShardCache._pin_lock(pin_path, shared=True):
        with _HubShardCache._pin_lock(pin_path, shared=True, blocking=False):
            pass
        with pytest.raises(BlockingIOError):
            with _HubShardCache._pin_lock(
                pin_path,
                shared=False,
                blocking=False,
            ):
                pass


def test_remote_shards_stream_to_completion_with_bounded_cache(tmp_path) -> None:
    _remote, files, downloads, download, list_files = _remote_token_fixture(tmp_path)
    dataset = PretokenizedCorpusMixtureDataset(
        "hf://datasets/test-owner/test-tokens",
        cache_dir=tmp_path / "cache",
        cache_max_bytes=20,
        prefetch_shards=1,
        hub_downloader=download,
        hub_file_lister=list_files,
    )

    # Construction fetches only tiny manifests. Binary shards remain lazy.
    assert not any(filename.endswith(".bin") for filename in downloads)
    samples = list(dataset)

    assert len(samples) == 5
    assert {name for name in downloads if name.endswith(".bin")} == {
        name for name in files if name.endswith(".bin")
    }
    for index, sample in enumerate(samples):
        assert torch.equal(
            sample["input_ids"], torch.arange(index * 4, index * 4 + 4)
        )
    cached_shards = list((tmp_path / "cache").rglob("*.bin"))
    assert sum(path.stat().st_size for path in cached_shards) <= 20


def test_remote_preflight_rejects_a_missing_shard_before_training(tmp_path) -> None:
    _remote, files, _downloads, download, _list_files = _remote_token_fixture(tmp_path)
    missing = next(filename for filename in files if filename.endswith("tokens-00001.bin"))

    with pytest.raises(FileNotFoundError, match="remote token mixture is incomplete"):
        PretokenizedCorpusMixtureDataset(
            "hf://datasets/test-owner/test-tokens",
            cache_dir=tmp_path / "cache",
            hub_downloader=download,
            hub_file_lister=lambda **_kwargs: sorted(files - {missing}),
        )


def test_remote_cache_redownloads_a_corrupt_shard_before_exposing_tokens(tmp_path) -> None:
    remote, files, _downloads, _download, list_files = _remote_token_fixture(tmp_path)
    attempts = {}

    def corrupt_once(*, filename, local_dir, **_kwargs):
        attempts[filename] = attempts.get(filename, 0) + 1
        destination = Path(local_dir) / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if filename.endswith("tokens-00000.bin") and attempts[filename] == 1:
            destination.write_bytes(b"broken")
        else:
            shutil.copy2(remote / filename, destination)
        return str(destination)

    dataset = PretokenizedCorpusMixtureDataset(
        "hf://datasets/test-owner/test-tokens",
        cache_dir=tmp_path / "cache",
        prefetch_shards=0,
        hub_downloader=corrupt_once,
        hub_file_lister=list_files,
    )

    first = next(iter(dataset))

    assert first["input_ids"].tolist() == [0, 1, 2, 3]
    assert attempts["corpora/tiny/tokens-00000.bin"] == 2
    assert "corpora/tiny/tokens-00000.bin" in files


def test_manifest_preflight_rejects_incomplete_shard_coverage(tmp_path) -> None:
    remote, _files, _downloads, _download, _list_files = _remote_token_fixture(tmp_path)
    source_manifest = remote / "corpora" / "tiny" / "manifest.json"
    metadata = json.loads(source_manifest.read_text(encoding="utf-8"))
    metadata["shards"] = metadata["shards"][:-1]
    source_manifest.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="shard coverage mismatch"):
        PretokenizedCorpusMixtureDataset(remote)


def test_replay_plan_reuses_selected_rows_without_copying_shards(tmp_path) -> None:
    remote, _files, _downloads, _download, _list_files = _remote_token_fixture(tmp_path)
    plan = {
        "format": "tr-hash-token-replay-plan-v1",
        "seq_len": 4,
        "unique_tokens": 8,
        "trained_tokens": 16,
        "phases": [
            {
                "name": "selected",
                "passes": 2,
                "sources": {
                    "tiny": [{"file": "tokens-00001.bin", "rows": 2}]
                },
            }
        ],
    }
    dataset = PretokenizedCorpusMixtureDataset(remote, replay_plan=plan)

    samples = list(dataset)

    assert dataset.unique_tokens == 8
    assert dataset.trained_tokens == 16
    assert len(samples) == 4
    assert [sample["input_ids"].tolist() for sample in samples] == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]


def test_resume_skip_rows_skips_already_trained_rows_without_disturbing_the_plan(tmp_path) -> None:
    """Regression guard: a process restart re-instantiates the dataset, and
    __iter__() always starts at phase 1 / shard 0 with no persisted position
    -- observed live on the 200M run, where a supervisorctl restart re-served
    ~28B already-trained-on tokens. resume_skip_rows lets a resumed run
    discard exactly the rows it already consumed before the restart, landing
    on the same continuation point a never-interrupted run would have
    reached -- same plan, same total passes, nothing re-served twice."""
    remote, _files, _downloads, _download, _list_files = _remote_token_fixture(tmp_path)
    plan = {
        "format": "tr-hash-token-replay-plan-v1",
        "seq_len": 4,
        "unique_tokens": 8,
        "trained_tokens": 16,
        "phases": [
            {
                "name": "selected",
                "passes": 2,
                "sources": {
                    "tiny": [{"file": "tokens-00001.bin", "rows": 2}]
                },
            }
        ],
    }
    # Without resume_skip_rows, this plan yields 4 samples: pass 1 (rows 0,1)
    # then pass 2 (rows 0,1 again). Skipping 2 rows should land exactly at
    # the start of pass 2 -- i.e. skip all of pass 1, yield only pass 2.
    dataset = PretokenizedCorpusMixtureDataset(remote, replay_plan=plan, resume_skip_rows=2)

    samples = list(dataset)

    assert dataset.unique_tokens == 8  # plan metadata is untouched by the skip
    assert dataset.trained_tokens == 16
    assert len(samples) == 2
    assert [sample["input_ids"].tolist() for sample in samples] == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]


def test_resume_skip_rows_defaults_to_zero_and_changes_nothing(tmp_path) -> None:
    remote, _files, _downloads, _download, _list_files = _remote_token_fixture(tmp_path)
    plan = {
        "format": "tr-hash-token-replay-plan-v1",
        "seq_len": 4,
        "unique_tokens": 8,
        "trained_tokens": 16,
        "phases": [
            {"name": "selected", "passes": 2, "sources": {"tiny": [{"file": "tokens-00001.bin", "rows": 2}]}}
        ],
    }
    dataset = PretokenizedCorpusMixtureDataset(remote, replay_plan=plan)
    dataset_explicit_zero = PretokenizedCorpusMixtureDataset(remote, replay_plan=plan, resume_skip_rows=0)

    def as_lists(ds):
        return [sample["input_ids"].tolist() for sample in ds]

    assert as_lists(dataset) == as_lists(dataset_explicit_zero)


def test_resume_skip_rows_does_not_download_fully_skipped_remote_shards(tmp_path) -> None:
    _remote, _files, downloads, download, list_files = _remote_token_fixture(tmp_path)
    plan = {
        "format": "tr-hash-token-replay-plan-v1",
        "seq_len": 4,
        "unique_tokens": 16,
        "trained_tokens": 16,
        "phases": [
            {
                "name": "already_trained",
                "passes": 1,
                "sources": {"tiny": [{"file": "tokens-00000.bin", "rows": 2}]},
            },
            {
                "name": "continuation",
                "passes": 1,
                "sources": {"tiny": [{"file": "tokens-00001.bin", "rows": 2}]},
            },
        ],
    }
    dataset = PretokenizedCorpusMixtureDataset(
        "hf://datasets/test-owner/test-tokens",
        cache_dir=tmp_path / "cache",
        prefetch_shards=0,
        hub_downloader=download,
        hub_file_lister=list_files,
        replay_plan=plan,
        resume_skip_rows=2,
    )

    samples = list(dataset)

    assert [sample["input_ids"].tolist() for sample in samples] == [
        [8, 9, 10, 11],
        [12, 13, 14, 15],
    ]
    assert "corpora/tiny/tokens-00000.bin" not in downloads
    assert "corpora/tiny/tokens-00001.bin" in downloads


def test_resume_skip_rows_rejects_negative_values(tmp_path) -> None:
    remote, _files, _downloads, _download, _list_files = _remote_token_fixture(tmp_path)
    with pytest.raises(ValueError, match="non-negative"):
        PretokenizedCorpusMixtureDataset(remote, resume_skip_rows=-1)


def test_corrective_replay_plan_actually_loads_through_the_real_validator(tmp_path) -> None:
    """Regression guard: build_corrective_replay_plan's unique_tokens field
    was copied verbatim from the uncorrected plan instead of recomputed --
    passed its own unit tests (which checked the dict directly) but failed
    live the moment the real loader validated it, because
    PretokenizedCorpusMixtureDataset._load_replay_plan defines unique_tokens
    as every DISTINCT shard touched anywhere in the plan, and the whole
    point of the correction is to point later phases at shards phase 1 never
    touched -- so the two numbers necessarily diverge. This instantiates a
    real dataset with the corrected plan (not a stub) so a wrong metadata
    field fails the same way it did on the actual 200M run instead of only
    a hand-checked assertion."""
    remote = tmp_path / "remote"
    source = TextCorpusSource("tiny", 1.0, data_files="unused.jsonl")
    writer = TokenShardWriter(
        remote / "corpora" / source.name, seq_len=4, total_rows=10, rows_per_shard=2,
    )
    writer.feed(torch.arange(41, dtype=torch.int64).numpy())
    writer.write_manifest(source=source)
    write_mixture_manifest(
        output_root=remote,
        sources=(source,),
        seq_len=4,
        requested_tokens=40,
        actual_tokens=40,
        global_batch_sequences=1,
        rows_by_source={"tiny": 10},
    )

    dataset = PretokenizedCorpusMixtureDataset(remote)
    corrected_plan = build_corrective_replay_plan(
        dataset,
        unique_token_budgets={"tiny": 16},  # 4 rows @ seq_len=4 -> phase 1 uses shards 0,1
        replay_passes={"tiny": 2},
        already_double_exposed_shards={"tiny": 1},  # shard 0 already got an extra pass
        row_alignment=1,
    )

    # The bug under test: this used to raise ValueError("unique_tokens
    # mismatch") the instant a real dataset tried to load the plan.
    resumed = PretokenizedCorpusMixtureDataset(remote, replay_plan=corrected_plan)

    samples = list(resumed)
    seen_shards = {
        s["file"]
        for phase in corrected_plan["phases"]
        for s in phase["sources"]["tiny"]
    }
    assert "tokens-00000.bin" not in {
        s["file"] for s in corrected_plan["phases"][1]["sources"]["tiny"]
    }  # the burned shard must not reappear in the corrected replay phase
    assert len(samples) > 0
    assert seen_shards  # sanity: the plan actually references shards at all


def test_quality_plan_selects_highest_scored_shards_and_records_exposure(tmp_path) -> None:
    remote, _files, _downloads, _download, _list_files = _remote_token_fixture(tmp_path)
    dataset = PretokenizedCorpusMixtureDataset(remote)
    scores = {
        "corpora/tiny/tokens-00000.bin": 0.1,
        "corpora/tiny/tokens-00001.bin": 0.9,
        "corpora/tiny/tokens-00002.bin": 0.5,
    }

    plan = build_replay_plan(
        dataset,
        unique_token_budgets={"tiny": 8},
        replay_passes={"tiny": 3},
        row_alignment=1,
        quality_scores=scores,
    )

    assert plan["selection_mode"] == "quality_score"
    assert plan["unique_tokens"] == 8
    assert plan["trained_tokens"] == 24
    assert plan["phases"][0]["sources"]["tiny"] == [
        {"file": "tokens-00001.bin", "rows": 2}
    ]
    assert [phase["name"] for phase in plan["phases"]] == [
        "unique_core",
        "quality_replay_2",
        "quality_replay_3",
    ]


def test_default_70b_plan_separates_unique_tokens_from_replayed_exposure() -> None:
    assert sum(DEFAULT_UNIQUE_BUDGETS.values()) == 70_000_000_000
    assert sum(
        DEFAULT_UNIQUE_BUDGETS[name] * DEFAULT_REPLAY_PASSES[name]
        for name in DEFAULT_UNIQUE_BUDGETS
    ) == 130_000_000_000


def test_hub_upload_preserves_corpus_paths_and_uses_resumable_api(tmp_path) -> None:
    class _Api:
        def __init__(self) -> None:
            self.calls = []

        def upload_large_folder(self, **kwargs) -> None:
            self.calls.append(kwargs)

    api = _Api()
    upload_dataset_subset(
        output_root=tmp_path,
        repo_id=DEFAULT_HF_REPO,
        allow_patterns=("corpora/dclm/**",),
        token=None,
        workers=64,
        api=api,
    )

    assert api.calls == [
        {
            "repo_id": "Pacific-i64/data-32k-200b-tokens",
            "repo_type": "dataset",
            "folder_path": tmp_path,
            "allow_patterns": ["corpora/dclm/**"],
            "num_workers": 64,
            "print_report": True,
            "print_report_every": 60,
        }
    ]
