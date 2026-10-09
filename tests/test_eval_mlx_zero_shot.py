from __future__ import annotations

import json
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest

from complexity.evaluation import zero_shot


def test_arc_easy_matches_lm_eval_zero_shot_prompt_contract(monkeypatch) -> None:
    evaluator = zero_shot
    row = {
        "id": "Mercury_SC_401169",
        "question": "Which object is attracted to a magnet?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["a paper clip", "a rubber band", "a leaf", "a glass"],
        },
        "answerKey": "A",
    }
    monkeypatch.setattr(
        evaluator,
        "load_dataset",
        lambda path, name, split: [row],
    )

    examples = evaluator.load_arc_easy(maximum=None)

    assert examples == [
        evaluator.Example(
            benchmark="arc_easy",
            example_id="Mercury_SC_401169",
            context="Question: Which object is attracted to a magnet?\nAnswer:",
            continuations=(
                " a paper clip",
                " a rubber band",
                " a leaf",
                " a glass",
            ),
            answer=0,
        )
    ]


def test_arc_easy_limit_is_applied_after_formatting(monkeypatch) -> None:
    evaluator = zero_shot
    rows = [
        {
            "id": str(index),
            "question": f"Question {index}?",
            "choices": {"label": ["1", "2"], "text": ["yes", "no"]},
            "answerKey": 2,
        }
        for index in range(3)
    ]
    monkeypatch.setattr(
        evaluator,
        "load_dataset",
        lambda path, name, split: rows,
    )

    examples = evaluator.load_arc_easy(maximum=2)

    assert [example.example_id for example in examples] == ["0", "1"]
    assert [example.answer for example in examples] == [1, 1]


def test_arc_challenge_uses_the_challenge_test_configuration(monkeypatch) -> None:
    evaluator = zero_shot
    calls = []
    row = {
        "id": "challenge-1",
        "question": "Which force pulls objects toward Earth?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["gravity", "friction", "magnetism", "electricity"],
        },
        "answerKey": "A",
    }

    def fake_load_dataset(path, name, split):
        calls.append((path, name, split))
        return [row]

    monkeypatch.setattr(evaluator, "load_dataset", fake_load_dataset)

    examples = evaluator.load_arc_challenge(maximum=None)

    assert calls == [("allenai/ai2_arc", "ARC-Challenge", "test")]
    assert examples[0].benchmark == "arc_challenge"
    assert examples[0].answer == 0
    assert examples[0].context.endswith("\nAnswer:")


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0
    texts = {"context": [1, 2, 3, 4], " short": [5], " long": [6, 1, 2]}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return self.texts[text].copy()


def test_encoding_retains_whole_completion_when_trimming_context() -> None:
    example = zero_shot.Example("test", "one", "context", (" short", " long"), 0)

    encoded = zero_shot.encode_choices(_Tokenizer(), [example], max_length=5)

    assert encoded[0].ids == (1, 2, 3, 4, 5)
    assert encoded[1].ids == (3, 4, 6, 1, 2)
    assert encoded[1].completion_start == 2
    assert encoded[1].completion_length == 3
    with pytest.raises(ValueError, match="scoring boundary"):
        zero_shot.encode_choices(_Tokenizer(), [example], max_length=3)


class _NumpyMLX:
    """Small numeric backend for verifying the MLX causal scoring operation."""

    int32 = np.int32
    float32 = np.float32
    array = staticmethod(np.array)
    arange = staticmethod(np.arange)
    sum = staticmethod(np.sum)
    where = staticmethod(np.where)
    take_along_axis = staticmethod(np.take_along_axis)
    logsumexp = staticmethod(lambda array, axis: np.logaddexp.reduce(array, axis=axis))

    @staticmethod
    def eval(*_arrays):
        pass


def _causal_model(tokens):
    # Every prediction depends only on the prefix, so right padding cannot
    # change a real token's probability.
    prefix = np.cumsum(tokens, axis=1) % 7
    vocabulary = np.arange(7)[None, None, :]
    return -np.abs(prefix[..., None] - vocabulary).astype(np.float32)


def test_scoring_uses_only_shifted_completion_targets_and_excludes_padding(monkeypatch) -> None:
    monkeypatch.setattr(zero_shot, "_mlx", lambda: _NumpyMLX)
    examples = [
        zero_shot.Example("test", "one", "context", (" short", " long"), 1),
        zero_shot.Example("test", "two", "context", (" short",), 0),
    ]
    tokenizer = _Tokenizer()

    batched = zero_shot.score_examples(
        _causal_model, tokenizer, examples, batch_size=3, max_length=8
    )
    unbatched = zero_shot.score_examples(
        _causal_model, tokenizer, examples, batch_size=1, max_length=8
    )

    for choice in zero_shot.encode_choices(tokenizer, examples, max_length=8):
        tokens = np.array([choice.ids], dtype=np.int32)
        logits = _causal_model(tokens)[0]
        expected = sum(
            float(logits[position - 1, tokens[0, position]])
            - float(np.logaddexp.reduce(logits[position - 1]))
            for position in range(choice.completion_start, len(choice.ids))
        )
        actual = batched[choice.example_index][choice.choice_index]
        assert actual["loglikelihood"] == pytest.approx(expected)
        assert actual["loglikelihood_normalized"] == pytest.approx(
            expected / choice.completion_length
        )
        assert actual == pytest.approx(unbatched[choice.example_index][choice.choice_index])


def test_summary_distinguishes_raw_and_normalized_decisions() -> None:
    examples = [
        zero_shot.Example("test", "one", "context", (" short", " long"), 1, "a"),
        zero_shot.Example("test", "two", "context", (" short",), 0, "b"),
    ]
    scores = [
        [
            {"loglikelihood": -2.0, "loglikelihood_normalized": -2.0},
            {"loglikelihood": -3.0, "loglikelihood_normalized": -1.0},
        ],
        [{"loglikelihood": -1.0, "loglikelihood_normalized": -1.0}],
    ]

    summary = zero_shot.summarize(examples, scores)

    assert summary["acc"] == 0.5
    assert summary["acc_norm"] == 1.0
    assert summary["macro_acc"] == 0.5
    assert summary["subjects"]["a"]["acc"] == 0.0
    with pytest.raises(ValueError, match="exactly one score"):
        zero_shot.summarize(examples[:1], scores[1:])
    with pytest.raises(ValueError, match="finite"):
        zero_shot.summarize(
            examples[1:], [[{"loglikelihood": float("nan"), "loglikelihood_normalized": 0}]]
        )


def test_piqa_reads_local_official_files_without_dataset_runtime(tmp_path) -> None:
    (tmp_path / "dev.jsonl").write_text(
        json.dumps({"goal": "Remove a screw", "sol1": " use a screwdriver", "sol2": "use glue"})
        + "\n"
    )
    (tmp_path / "dev-labels.lst").write_text("0\n")

    examples = zero_shot.load_piqa(None, data_dir=tmp_path)

    assert examples == [
        zero_shot.Example("piqa", "0", "Remove a screw", (" use a screwdriver", " use glue"), 0)
    ]


def test_bundle_load_failure_propagates_without_ignoring_missing_weights(monkeypatch, tmp_path):
    fake_utils = ModuleType("mlx_lm.utils")
    calls = []

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        raise ValueError("missing trained weights")

    fake_utils.load = load
    monkeypatch.setitem(sys.modules, "mlx_lm", ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.utils", fake_utils)

    with pytest.raises(ValueError, match="missing trained weights"):
        zero_shot._load_bundle(tmp_path)

    assert calls == [((tmp_path.as_posix(),), {})]


def test_evaluation_helpers_and_cli_help_need_no_optional_runtimes() -> None:
    program = """
import importlib.abc
import runpy
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'datasets', 'torch'}:
            raise RuntimeError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from complexity.evaluation import Example, encode_choices
Example('test', '1', 'context', (' choice',), 0)
sys.argv = ['zero_shot', '--help']
runpy.run_module('complexity.evaluation.zero_shot', run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "--tasks" in result.stdout
    assert "--piqa-data" in result.stdout


def test_cli_runs_bundle_to_report_without_conversion_scripts(monkeypatch, tmp_path, capsys):
    class Model:
        def __call__(self, tokens):
            return _causal_model(tokens)

        def eval(self):
            pass

        def parameters(self):
            return []

    report = tmp_path / "result.json"
    example = zero_shot.Example("arc_easy", "one", "context", (" short", " long"), 0)
    monkeypatch.setattr(zero_shot, "_mlx", lambda: _NumpyMLX)
    monkeypatch.setattr(zero_shot, "_load_bundle", lambda path: (Model(), _Tokenizer()))
    monkeypatch.setitem(zero_shot.LOADERS, "arc_easy", lambda maximum: [example])
    monkeypatch.setattr(
        sys, "argv", ["zero_shot", str(tmp_path), "--tasks", "arc_easy", "--output", str(report)]
    )

    zero_shot.main()

    result = json.loads(report.read_text())
    captured = capsys.readouterr()
    assert json.loads(captured.out) == result
    assert result["complete"] is True
    assert result["benchmarks"]["arc_easy"]["examples"] == 1
    assert result["continuation_tokenization"] == "separate_context_and_continuation"
    assert result["chat_template_applied"] is False
    assert not report.with_name("result.json.tmp").exists()
