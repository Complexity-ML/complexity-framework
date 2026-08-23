from scripts.select_reasoning_preservation_checkpoint import evaluation_equivalent


def candidate(**updates):
    value = {
        "arc_reasoning_native_accuracy": 0.25,
        "behavior_passes": 6,
        "piqa_acc": 0.68,
        "piqa_acc_norm": 0.69,
        "arc_acc": 0.47,
        "arc_acc_norm": 0.48,
    }
    value.update(updates)
    return value


def test_final_tie_accepts_one_example_resolution() -> None:
    assert evaluation_equivalent(
        candidate(piqa_acc_norm=0.69 + 1 / 1_838),
        candidate(),
    )


def test_final_tie_rejects_different_reasoning_result() -> None:
    assert not evaluation_equivalent(
        candidate(arc_reasoning_native_accuracy=0.25 + 1 / 64),
        candidate(),
    )


def test_final_tie_rejects_more_than_one_arc_example() -> None:
    assert not evaluation_equivalent(
        candidate(arc_acc_norm=0.48 + 2 / 3_548),
        candidate(),
    )
