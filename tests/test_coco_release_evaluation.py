import pytest

from scripts.evaluate_tr_hash_coco import (
    BRANCHES,
    _branches_to_run,
    _percentile,
    _timing_summary,
)


def test_both_branches_require_end_to_end_head():
    with pytest.raises(ValueError, match="NMS-free"):
        _branches_to_run("both", has_nms_free=False)


def test_o2m_only_does_not_require_end_to_end_head():
    assert _branches_to_run("o2m-nms", has_nms_free=False) == ("o2m-nms",)


def test_both_branches_keep_fixed_comparison_order():
    assert _branches_to_run("both", has_nms_free=True) == BRANCHES


def test_timing_summary_uses_measured_batches_and_image_count():
    summary = _timing_summary([0.1, 0.3, 0.2], image_count=12)

    assert summary["mean_batch_ms"] == pytest.approx(200.0)
    assert summary["p50_batch_ms"] == pytest.approx(200.0)
    assert summary["p95_batch_ms"] == pytest.approx(300.0)
    assert summary["images_per_second"] == pytest.approx(20.0)
    assert summary["measured_seconds"] == pytest.approx(0.6)


def test_percentile_handles_empty_measurements():
    assert _percentile([], 0.95) == 0.0
