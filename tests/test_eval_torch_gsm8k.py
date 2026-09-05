from decimal import Decimal

import pytest

from scripts.eval_torch_gsm8k import extract_answer, ground_truth, load_examples


def test_extract_answer_prefers_explicit_final_answer() -> None:
    assert extract_answer("I first obtained 12. Final answer: 15") == Decimal("15")


def test_extract_answer_supports_boxed_fraction() -> None:
    assert extract_answer(r"Therefore \\boxed{5/4}") == Decimal("1.25")


def test_ground_truth_uses_gsm8k_marker() -> None:
    assert ground_truth("work mentions 12\n#### 1,234") == Decimal("1234")


def test_supervised_evaluation_rejects_non_test_split() -> None:
    with pytest.raises(ValueError, match="split=test"):
        load_examples(None, None, split="train")
