"""Public commands exercise real native checkpoint loading and causal scoring."""

import hashlib
import json
import subprocess
import sys

import pytest
import torch

from complexity import ComplexityModel, ModelConfig
from complexity.evaluation import zero_shot


@pytest.fixture
def bundles(tmp_path):
    from tokenizers import Tokenizer as FastTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

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
    torch.manual_seed(42)
    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=16,
        shared_intermediate_size=32,
        vocab_size=32,
        num_experts=2,
        top_k=2,
        routing_strategy="token_id_multi_hash",
        max_position_embeddings=64,
        use_custom_kernels=False,
        use_sdpa=False,
    )
    model = ComplexityModel(config).eval()
    context = "Question: What is correct?\nAnswer:"
    ids = torch.tensor([tokenizer.encode(context).ids])
    with torch.inference_mode():
        direction = model(ids)["last_hidden_state"][0, -1].clone()
        # Both candidates contain one token. Swapping just their output
        # embeddings forces opposite choices without altering the context.
        model.embed_tokens.weight[8].copy_(direction)
        model.embed_tokens.weight[9].copy_(-direction)
    a = tmp_path / "a"
    model.config.use_custom_kernels = True
    model.config.use_cggr = "auto"
    model.config.use_sdpa = True
    model.save_pretrained(a)
    with torch.inference_mode():
        model.embed_tokens.weight[8].copy_(-direction)
        model.embed_tokens.weight[9].copy_(direction)
    model.config.use_custom_kernels = False
    model.config.use_cggr = False
    model.config.use_sdpa = False
    b = tmp_path / "b"
    model.save_pretrained(b)

    arc = tmp_path / "arc.jsonl"
    arc.write_text(
        json.dumps(
            {
                "id": "real-fixture",
                "question": "What is correct?",
                "choices": {"label": ["A", "B"], "text": ["First", "Second"]},
                "answerKey": "A",
            }
        )
        + "\n"
    )
    piqa = tmp_path / "piqa"
    piqa.mkdir()
    (piqa / "dev.jsonl").write_text(
        json.dumps(
            {
                "goal": context,
                "sol1": "First",
                "sol2": "Second",
            }
        )
        + "\n"
    )
    (piqa / "dev-labels.lst").write_text("0\n")
    return a, b, tokenizer_path, arc, piqa


def _fingerprints(paths):
    return {
        str(file): hashlib.sha256(file.read_bytes()).hexdigest()
        for path in paths
        for file in path.rglob("*")
        if file.is_file()
    }


def _options(bundles):
    _, _, tokenizer, arc, piqa = bundles
    return dict(
        tokenizer_path=tokenizer,
        tasks="arc_easy,piqa",
        max_length=64,
        batch_size=2,
        arc_easy_samples=arc,
        piqa_data=piqa,
    )


def test_compare_real_checkpoints_shares_data_and_encodings_and_returns_delta(
    bundles,
    monkeypatch,
    tmp_path,
):
    a, b, _, _, _ = bundles
    before = _fingerprints((a, b))
    calls = {"tokenizer": 0, "arc": 0, "piqa": 0, "encoding": 0}
    for key, name in (
        ("tokenizer", "_load_torch_tokenizer"),
        ("arc", "load_arc"),
        ("piqa", "load_piqa"),
        ("encoding", "encode_choices"),
    ):
        original = getattr(zero_shot, name)

        def counted(*args, _key=key, _original=original, **kwargs):
            calls[_key] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(zero_shot, name, counted)
    output = tmp_path / "comparison.json"
    result = zero_shot.compare_choices(a, b, output=output, **_options(bundles))
    assert calls == {"tokenizer": 1, "arc": 1, "piqa": 1, "encoding": 2}
    assert result["complete"] is True
    assert result["max_samples"] is None
    assert result["kernel_policy"] == "portable"
    assert result["kernel_overrides"] == {
        "use_custom_kernels": False,
        "use_cggr": False,
        "use_sdpa": False,
        "rmsnorm": "pytorch_reference",
    }
    for task in ("arc_easy", "piqa"):
        assert result["models"]["A"]["benchmarks"][task]["acc"] == 1
        assert result["models"]["B"]["benchmarks"][task]["acc"] == 0
        assert result["delta_b_minus_a"][task] == {
            "acc": -1,
            "acc_norm": -1,
            "acc_points": -100,
            "acc_norm_points": -100,
        }
        assert result["datasets"][task]["examples"] == 1
    assert json.loads(output.read_text()) == result
    assert not output.with_name("comparison.json.tmp").exists()
    assert _fingerprints((a, b)) == before


def test_portable_loader_normalizes_instances_without_changing_source_config(bundles):
    from complexity.core.normalization.norms import RMSNorm

    a, _, tokenizer, _, _ = bundles
    before = _fingerprints((a,))
    model, _ = zero_shot.load_torch_bundle(a, tokenizer_path=tokenizer, kernel_policy="portable")
    assert model.config.use_custom_kernels is False
    assert model.config.use_cggr is False
    assert model.config.use_sdpa is False
    assert not model.training
    assert not any(parameter.requires_grad for parameter in model.parameters())
    assert model.layers[0].self_attn.use_sdpa is False
    assert model.layers[0].mlp.engine.config.backend.value == "pytorch"
    norms = [module for module in model.modules() if isinstance(module, RMSNorm)]
    assert norms
    assert all(module.forward.__func__ is zero_shot._portable_rmsnorm for module in norms)
    assert RMSNorm.forward is not zero_shot._portable_rmsnorm
    assert _fingerprints((a,)) == before


def test_portable_loader_rejects_attention_that_ignores_its_kernel_policy(bundles):
    a, _, tokenizer, _, _ = bundles
    config = json.loads((a / "config.json").read_text())
    config["attention_type"] = "flash"
    (a / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Portable evaluation supports"):
        zero_shot.load_torch_bundle(a, tokenizer_path=tokenizer, kernel_policy="portable")


def test_compare_does_not_publish_complete_result_when_second_checkpoint_is_broken(
    bundles,
    tmp_path,
):
    from safetensors.torch import load_file, save_file

    a, b, _, _, _ = bundles
    weights = load_file(b / "model.safetensors")
    del weights["embed_tokens.weight"]
    save_file(weights, b / "model.safetensors")
    output = tmp_path / "comparison.json"
    with pytest.raises(RuntimeError, match="embed_tokens.weight"):
        zero_shot.compare_choices(a, b, output=output, **_options(bundles))
    report = json.loads(output.read_text())
    assert report["complete"] is False
    assert report["models"]["A"]["complete"] is True
    assert "B" not in report["models"]
    assert report["delta_b_minus_a"] == {}


@pytest.mark.parametrize(
    "settings",
    [
        {"tasks": "arc_easy,arc_easy"},
        {"batch_size": 0},
        {"max_samples": 0},
        {"kernel_policy": "checkpoint"},
        {"dtype": "float16"},
    ],
)
def test_invalid_comparison_fails_before_dataset_access(bundles, monkeypatch, settings):
    a, b, _, _, _ = bundles
    options = {**_options(bundles), **settings}
    monkeypatch.setattr(
        zero_shot, "_load_tasks", lambda *args, **kwargs: pytest.fail("data loaded")
    )
    with pytest.raises(ValueError):
        zero_shot.compare_choices(a, b, **options)


def test_cli_choices_and_compare_emit_clean_json_with_real_cpu_checkpoints(bundles, tmp_path):
    a, b, tokenizer, arc, piqa = bundles
    common = [
        "--tokenizer",
        str(tokenizer),
        "--tasks",
        "arc_easy,piqa",
        "--max-length",
        "64",
        "--batch-size",
        "2",
        "--arc-easy-samples",
        str(arc),
        "--piqa-data",
        str(piqa),
    ]
    output = tmp_path / "choices.json"
    choices = subprocess.run(
        [
            sys.executable,
            "-m",
            "complexity",
            "eval",
            "choices",
            str(a),
            "--backend",
            "torch",
            "--kernel-policy",
            "portable",
            "--output",
            str(output),
            *common,
        ],
        text=True,
        capture_output=True,
    )
    assert choices.returncode == 0, choices.stderr
    choice_report = json.loads(choices.stdout)
    assert choice_report == json.loads(output.read_text())
    assert choice_report["complete"] is True
    compared = subprocess.run(
        [
            sys.executable,
            "-m",
            "complexity",
            "eval",
            "compare",
            str(a),
            str(b),
            *common,
        ],
        text=True,
        capture_output=True,
    )
    assert compared.returncode == 0, compared.stderr
    report = json.loads(compared.stdout)
    assert report["complete"] is True
    assert report["delta_b_minus_a"]["piqa"]["acc_norm_points"] == -100


def test_eval_help_is_complete_without_importing_torch_mlx_or_datasets():
    program = """
import importlib.abc, sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx', 'mlx_lm', 'datasets'}:
            raise RuntimeError('unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, BlockOptional())
from complexity.evaluation import compare_choices, evaluate_choices
from typer.testing import CliRunner
from complexity.cli.app import app
for command in ('choices', 'compare'):
    result = CliRunner().invoke(app, ['eval', command, '--help'])
    assert result.exit_code == 0, result.output
    for flag in ('--tasks', '--dtype', '--kernel-policy', '--piqa-data', '--max-length'):
        assert flag in result.stdout
assert 'torch' not in sys.modules
print('help-ok')
"""
    result = subprocess.run([sys.executable, "-c", program], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "help-ok"


def test_compare_cli_requires_exactly_two_models():
    for arguments in (("one",), ("one", "two", "three")):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "complexity",
                "eval",
                "compare",
                *arguments,
                "--tokenizer",
                "local.json",
            ],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 2
        assert result.stdout == ""


@pytest.mark.parametrize("target", ["weights", "config", "arc", "tokenizer", "piqa"])
def test_comparison_output_cannot_replace_its_inputs(bundles, monkeypatch, target):
    a, b, tokenizer, arc, piqa = bundles
    output = {
        "weights": a / "model.safetensors",
        "config": b / "config.json",
        "arc": arc,
        "tokenizer": tokenizer,
        "piqa": piqa / "dev.jsonl",
    }[target]
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    monkeypatch.setattr(
        zero_shot, "_load_tasks", lambda *args, **kwargs: pytest.fail("data loaded")
    )
    with pytest.raises(ValueError, match="output"):
        zero_shot.compare_choices(a, b, output=output, **_options(bundles))
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
