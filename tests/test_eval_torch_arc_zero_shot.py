import json
from pathlib import Path

from scripts.eval_torch_arc_zero_shot import encode_choices, load_arc


class FakeTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        return list(range(1, len(text.split()) + 1))


def test_load_arc_uses_lm_eval_continuation_contract(tmp_path: Path) -> None:
    path = tmp_path / "arc.jsonl"
    row = {
        "doc_id": 7,
        "doc": {
            "id": "example-7",
            "question": "What is correct?",
            "choices": {"label": ["A", "B"], "text": ["First", "Second"]},
            "answerKey": "B",
        },
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    examples = load_arc(path, "arc_easy")
    assert len(examples) == 1
    assert examples[0].context == "Question: What is correct?\nAnswer:"
    assert examples[0].continuations == (" First", " Second")
    assert examples[0].answer == 1


def test_encode_choices_preserves_completion_boundary(tmp_path: Path) -> None:
    path = tmp_path / "arc.jsonl"
    path.write_text(
        json.dumps(
            {
                "doc_id": 0,
                "doc": {
                    "question": "One two three?",
                    "choices": {"label": ["1", "2"], "text": ["Alpha", "Beta"]},
                    "answerKey": "1",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    example = load_arc(path, "arc_challenge")
    choices = encode_choices(FakeTokenizer(), example, max_length=32)
    assert len(choices) == 2
    assert all(choice.completion_start == 5 for choice in choices)
