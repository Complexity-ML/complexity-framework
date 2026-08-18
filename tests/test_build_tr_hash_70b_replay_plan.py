"""Regression guard for the 200M pretrain data mix.

build_tr_hash_70b_replay_plan.py has no free "--target-tokens" knob: the
trained-token total is *derived* from per-source unique_token_budgets x
replay_passes. Bumping the total (e.g. to extend training past the current
130B-token plan) means changing one or both of those maps -- and doing that
carelessly changes each source's *relative* share of the trained stream, not
just the total. These tests lock in the source proportions the 200M run was
actually trained on, and demonstrate the one scaling approach (uniform
unique-budget scale-up, passes untouched) that provably preserves them
exactly for a bigger target.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path("scripts/build_tr_hash_70b_replay_plan.py")


def _load_module():
    spec = importlib.util.spec_from_file_location("build_tr_hash_70b_replay_plan", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def mod():
    return _load_module()


def _fake_dataset(unique_token_budgets: dict[str, int], *, headroom: float = 1.0):
    """A duck-typed PretokenizedCorpusMixtureDataset stand-in: seq_len=1 makes
    "rows" and "tokens" the same number, so the budgets below need no unit
    conversion, and one oversized shard per source avoids any row_alignment
    rounding in the plan (each shard has exactly enough -- or, with headroom
    > 1, more than enough -- rows to satisfy that source's budget)."""
    from complexity.training.corpus_mixture import PretokenizedCorpusMixtureDataset

    dataset = PretokenizedCorpusMixtureDataset.__new__(PretokenizedCorpusMixtureDataset)
    dataset.seq_len = 1
    dataset.sources = tuple(SimpleNamespace(name=name) for name in unique_token_budgets)
    dataset._source_manifests = {
        name: {"shards": [{"file": f"{name}_shard0.bin", "rows": int(budget * headroom)}]}
        for name, budget in unique_token_budgets.items()
    }
    return dataset


# The 200M run's actual, already-trained-on mix (source: DEFAULT_UNIQUE_BUDGETS
# / DEFAULT_REPLAY_PASSES in the script, and the training log's replay-schedule
# table: DCLM 20B x1, FineWeb-Edu-Dedup 25B x2, Stack-Edu 8B x2, FineMath 5B x3,
# InfiWebMath 5B x3, Cosmopedia v2 7B x2 = 70B unique / 130B trained).
EXPECTED_TRAINED_TOKENS = 130_000_000_000
EXPECTED_SOURCE_SHARE = {
    "dclm": 20 / 130,
    "fineweb_edu_dedup": 50 / 130,
    "stack_edu": 16 / 130,
    "finemath": 15 / 130,
    "infiwebmath": 15 / 130,
    "cosmopedia_v2": 14 / 130,
}


def test_default_replay_plan_matches_the_trained_200m_mix(mod):
    dataset = _fake_dataset(mod.DEFAULT_UNIQUE_BUDGETS)

    plan = mod.build_replay_plan(
        dataset,
        unique_token_budgets=mod.DEFAULT_UNIQUE_BUDGETS,
        replay_passes=mod.DEFAULT_REPLAY_PASSES,
        row_alignment=1,
    )

    assert plan["trained_tokens"] == EXPECTED_TRAINED_TOKENS
    assert plan["unique_tokens"] == 70_000_000_000

    for source, expected_share in EXPECTED_SOURCE_SHARE.items():
        actual_tokens = plan["source_unique_tokens"][source] * plan["source_passes"][source]
        actual_share = actual_tokens / plan["trained_tokens"]
        assert actual_share == pytest.approx(expected_share, abs=1e-9), (
            f"{source}: mix share drifted from the trained-200M baseline "
            f"({actual_share:.4%} vs {expected_share:.4%}) -- if this is "
            f"intentional, update EXPECTED_SOURCE_SHARE; if not, "
            f"DEFAULT_UNIQUE_BUDGETS / DEFAULT_REPLAY_PASSES changed by accident"
        )


@pytest.mark.parametrize("scale", [250 / 130, 2.0, 1.923])
def test_scaling_unique_budgets_uniformly_preserves_mix_shares_at_a_bigger_target(mod, scale):
    """The safe way to extend past 130B trained tokens without disturbing the
    mix: scale every source's unique_token_budgets by the same factor and
    leave replay_passes untouched. Because trained_tokens is linear in each
    source's unique budget, this scales the total exactly and leaves every
    source's share bit-for-bit unchanged -- unlike bumping replay_passes
    per-source, which can't hit an arbitrary target while also being integers,
    and unlike scaling budgets non-uniformly, which directly changes shares."""
    scaled_budgets = {
        name: int(budget * scale) for name, budget in mod.DEFAULT_UNIQUE_BUDGETS.items()
    }
    dataset = _fake_dataset(scaled_budgets)

    plan = mod.build_replay_plan(
        dataset,
        unique_token_budgets=scaled_budgets,
        replay_passes=mod.DEFAULT_REPLAY_PASSES,
        row_alignment=1,
    )

    for source, expected_share in EXPECTED_SOURCE_SHARE.items():
        actual_tokens = plan["source_unique_tokens"][source] * plan["source_passes"][source]
        actual_share = actual_tokens / plan["trained_tokens"]
        assert actual_share == pytest.approx(expected_share, abs=1e-6)


def test_bumping_only_one_sources_replay_passes_does_change_its_mix_share(mod):
    """Negative control: the failure mode this guards against. Bumping a
    single source's replay_passes (the naive way to "just add more tokens")
    inflates that source's share at every other source's expense."""
    budgets = dict(mod.DEFAULT_UNIQUE_BUDGETS)
    passes = dict(mod.DEFAULT_REPLAY_PASSES)
    passes["dclm"] = 5  # was 1
    dataset = _fake_dataset(budgets)

    plan = mod.build_replay_plan(
        dataset, unique_token_budgets=budgets, replay_passes=passes, row_alignment=1,
    )

    dclm_share = (
        plan["source_unique_tokens"]["dclm"] * plan["source_passes"]["dclm"]
        / plan["trained_tokens"]
    )
    assert dclm_share != pytest.approx(EXPECTED_SOURCE_SHARE["dclm"], abs=1e-9)
    assert dclm_share > EXPECTED_SOURCE_SHARE["dclm"]
