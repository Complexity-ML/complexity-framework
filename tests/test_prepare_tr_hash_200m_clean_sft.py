from pathlib import Path

from scripts.prepare_tr_hash_200m_clean_sft import (
    assistant_turn_examples,
    fit_complete_turns,
    load_recipe,
    normalize_source_row,
    rejection_reasons,
    strip_training_artifacts,
)
from scripts.train_tr_hash_200m_200b import make_config


class _CharacterTokenizer:
    eos_token_id = 2

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(len(text)))


def test_clean_recipe_selects_sources_and_uses_2048_tokens() -> None:
    recipe = load_recipe(Path("configs/tr_hash_200m_clean_sft_v2.json"))
    model_config = make_config()

    assert recipe["sequence_length"] == model_config.max_position_embeddings == 2048
    assert recipe["max_train_examples"] == 300_000
    assert sum(source["train_target"] for source in recipe["sources"]) == 300_000
    assert "context_qa_hotpot" in recipe["excluded_sources"]
    assert {source["capability"] for source in recipe["sources"]} >= {
        "verified_code",
        "verified_math",
        "multi_turn",
        "instruction",
    }
    math_source = next(
        source for source in recipe["sources"] if source["name"] == "openr1_math_verified"
    )
    assert math_source["dataset"] == "open-r1/OpenR1-Math-220k"
    assert math_source["adapter"] == "openr1_math_verified"
    assert "min_assistant_chars" not in math_source


def test_multiturn_expansion_keeps_complete_recent_turns() -> None:
    messages = [
        {"role": "user", "content": "old " * 30},
        {"role": "assistant", "content": "old answer " * 20},
        {"role": "user", "content": "My name is Boris."},
        {"role": "assistant", "content": "Hello Boris."},
        {"role": "user", "content": "What is my name?"},
        {"role": "assistant", "content": "Your name is Boris."},
    ]

    examples = list(
        assistant_turn_examples(
            messages,
            expand=True,
            tokenizer=_CharacterTokenizer(),
            sequence_length=180,
        )
    )

    assert examples
    assert examples[-1][-2]["content"] == "What is my name?"
    assert examples[-1][-1]["content"] == "Your name is Boris."
    assert all(example[-1]["role"] == "assistant" for example in examples)


def test_complete_turn_fitting_accepts_tokenizer_aligned_eos() -> None:
    messages = [
        {"role": "user", "content": "Remember blue."},
        {"role": "assistant", "content": "I will remember blue."},
        {"role": "user", "content": "What color?"},
        {"role": "assistant", "content": "Blue."},
    ]
    template = {
        "id": "test",
        "version": 1,
        "system_prompt": "",
        "system_format": "System:\n{content}\n\n",
        "user_format": "User:\n{content}\n\n",
        "assistant_prefix": "Assistant:\n",
        "turn_separator": "\n\n",
        "eos_token": "#",
        "assistant_only_loss": True,
        "training_projection": "card_corpus_v2_direct",
    }

    fitted = fit_complete_turns(messages, _CharacterTokenizer(), 500, template)

    assert fitted == messages


def test_quality_gate_strips_empty_think_and_rejects_refusal() -> None:
    messages = strip_training_artifacts(
        [
            {"role": "user", "content": "Write code."},
            {"role": "assistant", "content": "<think></think> I'm unable to assist with that."},
        ]
    )
    reasons = rejection_reasons(
        messages,
        require_valid_python=False,
        policy={
            "min_assistant_chars": 16,
            "reject_refusals": True,
            "reject_repeated_lines": True,
            "reject_template_artifacts": True,
        },
    )

    assert "<think>" not in messages[-1]["content"]
    assert "refusal" in reasons


def test_clean_sft_never_silently_truncates_single_turn_code() -> None:
    tokenizer = _CharacterTokenizer()
    complete = [
        {"role": "user", "content": "Write a complete Python game."},
        {
            "role": "assistant",
            "content": "def game():\n" + "    print('playing')\n" * 40,
        },
    ]
    oversized = [
        complete[0],
        {
            "role": "assistant",
            "content": "def game():\n" + "    print('playing')\n" * 200,
        },
    ]

    assert fit_complete_turns(complete, tokenizer, 2048) == complete
    assert fit_complete_turns(oversized, tokenizer, 2048) is None


def test_verified_math_adapter_keeps_only_correct_complete_trace() -> None:
    messages = normalize_source_row(
        {
            "problem": "What is 17 times 23?",
            "solution": "391",
            "generations": [
                "<think>bad</think><final>390</final>",
                "<think>17 * 20 = 340 and 17 * 3 = 51.</think><final>391</final>",
                "<think>unfinished",
            ],
            "correctness_math_verify": [False, True, True],
            "correctness_llama": [False, False, False],
            "finish_reasons": ["stop", "stop", "length"],
        },
        adapter="openr1_math_verified",
    )

    assert messages is not None
    assert messages[-1]["content"].endswith("391")
    assert "390" not in messages[-1]["content"]
    assert "<think>" not in messages[-1]["content"]


def test_verified_math_adapter_prefers_clean_reference_solution() -> None:
    messages = normalize_source_row(
        {
            "problem": "What is 17 times 23?",
            "solution": "Compute 17 × 20 = 340 and 17 × 3 = 51, so the answer is 391.",
            "generations": [
                "<think>" + "A very long verified trace. " * 20 + "</think><final>391</final>"
            ],
            "correctness_math_verify": [True],
            "correctness_llama": [False],
            "finish_reasons": ["stop"],
        },
        adapter="openr1_math_verified",
    )

    assert messages is not None
    assert messages[-1]["content"].startswith("Compute 17 × 20")


def test_execution_filtered_code_adapter_preserves_instruction_and_response() -> None:
    messages = normalize_source_row(
        {
            "instruction": "Write add(a, b).",
            "response": "def add(a, b):\n    return a + b",
        },
        adapter="instruction_response",
    )

    assert messages == [
        {"role": "user", "content": "Write add(a, b)."},
        {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
    ]
