from decimal import Decimal

import pytest

from scripts.eval_torch_gsm8k import (
    GSM8K_COT_FEWSHOT,
    GSM8K_SYSTEM_PROMPT,
    build_gsm8k_prompt,
    extract_answer,
    ground_truth,
    load_examples,
)


def test_extract_answer_prefers_explicit_final_answer() -> None:
    assert extract_answer("I first obtained 12. Final answer: 15") == Decimal("15")


def test_extract_answer_accepts_canonical_cot_answer() -> None:
    assert extract_answer("Work. The answer is 29.") == Decimal("29")


def test_extract_answer_supports_boxed_fraction() -> None:
    assert extract_answer(r"Therefore \\boxed{5/4}") == Decimal("1.25")


def test_ground_truth_uses_gsm8k_marker() -> None:
    assert ground_truth("work mentions 12\n#### 1,234") == Decimal("1234")


def test_supervised_evaluation_rejects_non_test_split() -> None:
    with pytest.raises(ValueError, match="split=test"):
        load_examples(None, None, split="train")


def test_native_eight_shot_prompt_uses_short_system_and_all_demos() -> None:
    template = {
        "id": "tr-hash-agentic-chat-v1",
        "version": 1,
        "system_prompt": "",
        "system_format": "<|system|>{content}<|end_of_turn|>",
        "user_format": "<|user|>{content}<|end_of_turn|>",
        "assistant_prefix": "<|assistant|>",
        "turn_separator": "",
        "eos_token": "<|end|>",
        "end_of_turn_token": "<|end_of_turn|>",
        "assistant_only_loss": True,
        "training_projection": "native_agentic_prompt_completion",
        "assistant_envelope": {
            "type": "optional_think_final",
            "think_start": "<|think_start|>",
            "think_end": "<|think_end|>",
            "final_start": "<|final_start|>",
            "final_end": "<|final_end|>",
            "scope": "reasoning_tasks",
        },
    }
    prompt = build_gsm8k_prompt("Target?", template, num_fewshot=8)
    assert prompt.startswith(f"<|system|>{GSM8K_SYSTEM_PROMPT}<|end_of_turn|>")
    assert prompt.count("<|user|>") == 9
    assert prompt.count("<|assistant|>") == 9
    assert prompt.count("<|end_of_turn|>") == 18
    assert GSM8K_COT_FEWSHOT[-1][0] in prompt
    assert prompt.endswith("<|user|>Target?<|end_of_turn|><|assistant|>")
    assert "<|think_start|>" not in prompt
