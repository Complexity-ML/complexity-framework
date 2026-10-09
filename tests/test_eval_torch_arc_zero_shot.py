import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from complexity.evaluation.zero_shot import (
    Example,
    encode_choices,
    load_arc,
    load_torch_bundle,
    score_examples,
)


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


class _NumericTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    texts = {"context": [1, 2, 3, 4], " short": [5], " long": [6, 1, 2]}

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return self.texts[text].copy()


class _CausalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.ones(1))
        self.inference_flags = []

    def forward(self, tokens):
        self.inference_flags.append((self.training, torch.is_grad_enabled()))
        prefix = tokens.cumsum(dim=1) % 7
        vocabulary = torch.arange(7, device=tokens.device)[None, None, :]
        return {"logits": -(prefix[..., None] - vocabulary).abs().float()}


@pytest.mark.parametrize("batch_size", [1, 3])
def test_torch_scores_continuation_loglikelihood_with_shift_and_padding(batch_size):
    examples = [
        Example("test", "one", "context", (" short", " long"), 1),
        Example("test", "two", "context", (" short",), 0),
    ]
    model = _CausalModel()
    tokenizer = _NumericTokenizer()

    scores = score_examples(
        model,
        tokenizer,
        examples,
        batch_size=batch_size,
        max_length=8,
        backend="torch",
        device="cpu",
    )

    assert model.training is True  # Scoring does not alter the caller's mode.
    assert all(flags == (False, False) for flags in model.inference_flags)
    for choice in encode_choices(tokenizer, examples, max_length=8):
        tokens = torch.tensor([choice.ids])
        probabilities = model(tokens)["logits"][0].log_softmax(dim=-1)
        expected = sum(
            probabilities[position - 1, tokens[0, position]].item()
            for position in range(choice.completion_start, len(choice.ids))
        )
        result = scores[choice.example_index][choice.choice_index]
        assert result["loglikelihood"] == pytest.approx(expected)
        assert result["loglikelihood_normalized"] == pytest.approx(
            expected / choice.completion_length
        )


def _native_bundle(tmp_path):
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from complexity import ComplexityModel, ModelConfig

    model = ComplexityModel(
        ModelConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            intermediate_size=32,
            vocab_size=32,
            num_experts=2,
            top_k=1,
        )
    ).eval()
    model_dir = tmp_path / "model"
    model.save_pretrained(model_dir)
    tokenizer = FastTokenizer(
        WordLevel(
            {
                "[UNK]": 0,
                "Question": 1,
                ":": 2,
                "What": 3,
                "is": 4,
                "correct": 5,
                "?": 6,
                "Answer": 7,
                "First": 8,
                "Second": 9,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return model_dir, tokenizer_path


def test_native_bundle_rejects_missing_learned_weights(tmp_path):
    from safetensors.torch import load_file, save_file

    model_dir, tokenizer_path = _native_bundle(tmp_path)
    path = model_dir / "model.safetensors"
    weights = load_file(path)
    del weights["embed_tokens.weight"]
    save_file(weights, path)

    with pytest.raises(RuntimeError, match="embed_tokens.weight"):
        load_torch_bundle(model_dir, tokenizer_path=tokenizer_path)


def test_native_torch_cli_runs_local_arc_without_mlx_or_dataset_download(tmp_path):
    model_dir, tokenizer_path = _native_bundle(tmp_path)
    samples = tmp_path / "arc.jsonl"
    samples.write_text(
        json.dumps(
            {
                "doc": {
                    "id": "example",
                    "question": "What is correct?",
                    "choices": {"label": ["A", "B"], "text": ["First", "Second"]},
                    "answerKey": "A",
                }
            }
        )
        + "\n"
    )
    report_path = tmp_path / "report.json"
    # Block all external benchmark and MLX dependencies in the actual CLI
    # process. Native checkpoint loading and Torch scoring remain real.
    program = """
import importlib.abc
import runpy
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'mlx', 'mlx_lm', 'datasets'}:
            raise RuntimeError('unexpected optional import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
sys.argv = ['zero_shot', *sys.argv[1:]]
runpy.run_module('complexity.evaluation.zero_shot', run_name='__main__')
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            program,
            str(model_dir),
            "--backend",
            "torch",
            "--device",
            "cpu",
            "--tokenizer",
            str(tokenizer_path),
            "--tasks",
            "arc_easy",
            "--arc-easy-samples",
            str(samples),
            "--output",
            str(report_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report == json.loads(report_path.read_text())
    assert report["complete"] is True
    assert report["backend"] == "torch"
    assert report["device"] == "cpu"
    assert report["dtype"] == "float32"
    assert report["benchmarks"]["arc_easy"]["examples"] == 1
