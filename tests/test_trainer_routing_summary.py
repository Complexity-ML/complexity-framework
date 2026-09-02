from types import SimpleNamespace

from complexity.training.trainer import _format_moe_routing_summary


def test_moe_routing_summary_uses_configured_weight() -> None:
    config = SimpleNamespace(top_k=2, top_k_primary_weight=0.5)

    assert _format_moe_routing_summary(config) == (
        "2  (per-layer routing always on; primary weight 0.5; secondary weight 0.5)"
    )


def test_moe_routing_summary_resolves_equal_weight_default() -> None:
    config = SimpleNamespace(top_k=4, top_k_primary_weight=None)

    assert _format_moe_routing_summary(config) == (
        "4  (per-layer routing always on; primary weight 0.25; secondary weight 0.25)"
    )


def test_moe_routing_summary_handles_single_route() -> None:
    config = SimpleNamespace(top_k=1, top_k_primary_weight=0.95)

    assert _format_moe_routing_summary(config) == (
        "1  (per-layer routing always on; route weight 1)"
    )
