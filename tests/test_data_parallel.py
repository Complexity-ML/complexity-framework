"""Tests for complexity.parallel.data_parallel utilities."""

from complexity.parallel.data_parallel import resolve_dist_timeout_s


def test_single_node_timeout_is_generous_enough_for_cold_remote_shards():
    """Regression guard: 8 single-node ranks pulling independent shards
    from a remote/streamed dataset warmed up at different speeds and
    tripped the old 120s NCCL watchdog on a healthy run."""
    assert resolve_dist_timeout_s(is_multi_node=False, env={}) == 600


def test_multi_node_timeout_stays_higher_for_ib_queue_pair_setup():
    assert resolve_dist_timeout_s(is_multi_node=True, env={}) == 1800


def test_timeout_is_overridable_via_env():
    assert resolve_dist_timeout_s(
        is_multi_node=False, env={"COMPLEXITY_DIST_TIMEOUT_S": "900"}
    ) == 900
    assert resolve_dist_timeout_s(
        is_multi_node=True, env={"COMPLEXITY_DIST_TIMEOUT_S": "60"}
    ) == 60
