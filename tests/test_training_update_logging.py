from __future__ import annotations

from complexity.training.runner import (
    build_training_metric_record,
    format_optimizer_update_contract,
)


def test_metric_record_exposes_optimizer_updates_without_breaking_step_readers() -> None:
    record = build_training_metric_record(
        step=8_156,
        max_steps=17_801,
        loss=2.320845,
        ppl=10.18,
        lr=6.144879e-5,
        tokens_per_step=3_932_160,
        expert_shares=[0.25, 0.25, 0.25, 0.25],
        expert_dead_count=0,
        elapsed_s=47_060.2,
    )

    assert record["optimizer_update"] == record["step"] == 8_156
    assert record["optimizer_updates_planned"] == record["max_steps"] == 17_801
    assert record["tokens_per_optimizer_update"] == record["tokens_per_step"] == 3_932_160
    assert record["tokens_trained"] == 32_070_696_960


def test_startup_contract_makes_refinement_update_count_obvious() -> None:
    lines = format_optimizer_update_contract(
        phase="refinement",
        max_updates=17_801,
        tokens_per_update=3_932_160,
    )

    assert lines[1] == "OPTIMIZER UPDATE CONTRACT"
    rendered = "\n".join(lines)
    assert "phase: refinement" in rendered
    assert "planned optimizer updates: 17,801" in rendered
    assert "tokens per optimizer update: 3,932,160" in rendered
    assert "scheduled token exposure: 69,996,380,160 (~69.996B)" in rendered
