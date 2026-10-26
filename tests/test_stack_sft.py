from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from complexity.training.stack_sft import (
    DatasetMix,
    SourceConfig,
    StackSFTConfig,
    StackSFTDatasetBuilder,
    StackSFTRunner,
)


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def test_stack_sft_config_loads_recipe():
    config = StackSFTConfig.from_yaml("complexity/training/stack_sft/recipes/50m_agentic.yaml")
    assert config.name == "sft-50m-plus5b-agentic-stack"
    assert len(config.stages) == 6
    assert config.stages[2].name == "finalization"
    assert any(source.name == "coding_tool_traces" for source in config.stages[4].mix.sources)
    assert all(source.path for stage in config.stages for source in stage.mix.sources)


def test_stack_sft_dataset_builder_reads_file_backed_sources(tmp_path: Path):
    general = write_jsonl(
        tmp_path / "general.jsonl",
        [
            {"prompt": "User:\nHello\n\nAssistant:\n", "completion": "Hello!"},
            {"instruction": "Explain gravity", "output": "Gravity pulls masses together."},
        ],
    )
    code = write_jsonl(
        tmp_path / "code.jsonl",
        [
            {
                "instruction": "Write Python that adds two numbers.",
                "input": "Use a function.",
                "output": "def add(a, b):\n    return a + b",
            }
        ],
    )
    mix = DatasetMix(
        records=12,
        seed=1,
        sources=[
            SourceConfig(name="general", path=str(general), format="jsonl", weight=0.5),
            SourceConfig(name="code", path=str(code), format="jsonl", weight=0.5),
        ],
    )

    out = StackSFTDatasetBuilder().build(mix, tmp_path / "mixed.jsonl")

    lines = out.read_text(encoding="utf-8").splitlines()
    joined = "\n".join(lines)
    assert len(lines) == 12
    assert "Gravity pulls masses together." in joined
    assert "def add(a, b):" in joined


def test_stack_sft_dataset_builder_reads_messages_source(tmp_path: Path):
    messages = write_jsonl(
        tmp_path / "messages.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            }
        ],
    )
    mix = DatasetMix(
        records=3,
        seed=2,
        sources=[SourceConfig(name="messages", path=str(messages), format="jsonl", weight=1.0)],
    )

    out = StackSFTDatasetBuilder().build(mix, tmp_path / "messages_out.jsonl")

    joined = out.read_text(encoding="utf-8")
    assert "User:\\nWhat is 2+2?" in joined
    assert '"completion": "4"' in joined


def test_stack_sft_dataset_builder_preserves_unicode_line_separators(tmp_path: Path):
    source = write_jsonl(
        tmp_path / "unicode_breaks.jsonl",
        [{"prompt": "User:\nA\u2028B\n\nAssistant:\n", "completion": "C"}],
    )
    mix = DatasetMix(
        records=3,
        seed=2,
        sources=[SourceConfig(name="unicode", path=str(source), format="jsonl", weight=1.0)],
    )

    out = StackSFTDatasetBuilder().build(mix, tmp_path / "unicode_out.jsonl")

    for line in out.read_text(encoding="utf-8").split("\n"):
        if line:
            json.loads(line)


def test_stack_sft_rejects_hardcoded_source_mapping():
    with pytest.raises(ValueError, match="hardcoded synthetic"):
        DatasetMix.from_dict({"records": 8, "sources": {"final": 1.0}})


def test_stack_sft_rejects_source_without_path(tmp_path: Path):
    mix = DatasetMix(
        records=1,
        seed=1,
        sources=[SourceConfig(name="bad", kind="chat", weight=1.0)],
    )

    with pytest.raises(ValueError, match="must define a JSONL/parquet path"):
        StackSFTDatasetBuilder().build(mix, tmp_path / "bad.jsonl")


def test_stack_sft_runner_dry_run_is_native_and_does_not_build_data(tmp_path: Path, capsys):
    source = write_jsonl(
        tmp_path / "source.jsonl",
        [{"prompt": "User:\nHi\n\nAssistant:\n", "completion": "Hi!"}],
    )
    config = StackSFTConfig.from_dict(
        {
            "name": "toy-stack",
            "base_checkpoint": "checkpoints/base/step_000001",
            "source_stage": "supervised-finetuning",
            "data_dir": str(tmp_path / "data"),
            "output_dir": str(tmp_path / "checkpoints"),
            "stages": [
                {
                    "name": "finalization",
                    "steps": 3,
                    "lr": 1e-5,
                    "mix": {
                        "records": 8,
                        "sources": [
                            {
                                "name": "file",
                                "path": str(source),
                                "format": "jsonl",
                                "weight": 1.0,
                            }
                        ],
                    },
                }
            ],
        }
    )
    runner = StackSFTRunner(config, dry_run=True)
    checkpoint = runner.run()
    report = json.loads(capsys.readouterr().out)
    assert report["workflow"] == "staged_lora_sft"
    assert report["training_started"] is False
    assert checkpoint.endswith("toy-stack-finalization/step_000003")
    assert not (tmp_path / "data").exists()
    assert not (tmp_path / "checkpoints").exists()


def test_stack_sft_rejects_full_parameter_training(tmp_path: Path):
    with pytest.raises(ValueError, match="positive LoRA rank"):
        StackSFTConfig.from_dict(
            {
                "name": "full-parameter-is-disabled",
                "base_checkpoint": "checkpoint",
                "lora_rank": 0,
                "stages": [
                    {
                        "name": "stage",
                        "steps": 1,
                        "lr": 1e-5,
                        "mix": {
                            "sources": [
                                {
                                    "name": "source",
                                    "path": str(tmp_path / "source.jsonl"),
                                    "weight": 1.0,
                                }
                            ]
                        },
                    }
                ],
            }
        )


@pytest.fixture
def runnable_stack(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    from complexity import ComplexityModel, ModelConfig
    from complexity.export.assets import write_chat_template
    from complexity.inference.chat_template import default_chat_template

    vocabulary = {"</s>": 0, "<unk>": 1, "User": 2, "Assistant": 3, ":": 4, "Hi": 5, "Hello": 6}
    vocabulary.update({f"token{index}": index for index in range(7, 32)})
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer.save(str(tokenizer_dir / "tokenizer.json"))
    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        shared_intermediate_size=16,
        vocab_size=32,
        num_experts=4,
        top_k=2,
        max_position_embeddings=16,
        use_custom_kernels=False,
    )
    torch.manual_seed(19)
    base = tmp_path / "base"
    ComplexityModel(config).save_pretrained(base)
    template = default_chat_template()
    template["eos_token"] = "</s>"
    write_chat_template({"chat_template": template}, base)
    source = write_jsonl(
        tmp_path / "source.jsonl", [{"prompt": "User: Hi Assistant:", "completion": "Hello"}]
    )
    values = {
        "name": "tiny",
        "base_checkpoint": str(base),
        "source_stage": "refinement",
        "tokenizer": str(tokenizer_dir),
        "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "output"),
        "device": "cpu",
        "precision": "fp32",
        "batch_size": 1,
        "seq_len": 16,
        "save_steps": 1,
        "lora_rank": 2,
        "lora_alpha": 4.0,
        "lora_dropout": 0.1,
        "grad_ckpt": True,
        "stages": [
            {
                "name": name,
                "steps": 2,
                "lr": 1e-3,
                "mix": {
                    "records": 3,
                    "sources": [{"name": "source", "path": str(source), "weight": 1.0}],
                },
            }
            for name in ("first", "second")
        ],
    }
    return values, tokenizer, template


def test_native_stack_two_lora_stages_export_loadable_weights(runnable_stack):
    from complexity.training.sft import load_sft_model

    values, _, _ = runnable_stack
    config = StackSFTConfig.from_dict(values)
    initial = load_sft_model(config.base_checkpoint)
    final = StackSFTRunner(config).run()
    restored = load_sft_model(final)
    assert Path(final).name == "step_000002"
    torch.testing.assert_close(
        initial.embed_tokens.weight, restored.embed_tokens.weight, rtol=0, atol=0
    )
    assert any(
        not torch.equal(tensor, restored.state_dict()[name])
        for name, tensor in initial.state_dict().items()
    )
    assert not any("lora_" in key or ".base." in key for key in restored.state_dict())
    marker = json.loads((Path(final) / "stage.json").read_text())
    assert marker["training_stage"] == "supervised-finetuning"
    assert marker["source_stage"] == "supervised-finetuning"
    assert marker["release_eligible"] is False


def test_native_stack_prompt_mask_and_final_eos_are_retained(runnable_stack, tmp_path):
    from complexity.training.stack_sft.stage import SFTPromptCompletionDataset

    values, tokenizer, template = runnable_stack
    path = write_jsonl(
        tmp_path / "masked.jsonl", [{"prompt": "User: Hi Assistant:", "completion": "Hello"}]
    )
    data = SFTPromptCompletionDataset(
        path, tokenizer, sequence_length=16, vocab_size=32, chat_template=template
    )
    active = data[0]["labels"][data[0]["labels"] != -100].tolist()
    assert active == [6, 0]
    assert data[0]["labels"][:4].tolist() == [-100] * 4
    with pytest.raises(ValueError, match="no truncation"):
        SFTPromptCompletionDataset(
            path, tokenizer, sequence_length=2, vocab_size=32, chat_template=template
        )


def test_native_stack_checkpoint_resume_keeps_dropout_rng_and_adapter_updates(
    runnable_stack, tmp_path, monkeypatch
):
    import complexity.training.stack_sft.stage as native

    values, _, _ = runnable_stack
    config = StackSFTConfig.from_dict(values)
    stage = config.stages[0]
    dataset = StackSFTDatasetBuilder().build(stage.mix, tmp_path / "resume.jsonl")
    full = native.run_lora_stage(
        config, stage, config.base_checkpoint, dataset, tmp_path / "continuous"
    )
    save = native._save_run_checkpoint

    def interrupt_after_step_one(*args, **kwargs):
        path = save(*args, **kwargs)
        if kwargs["step"] == 1:
            raise KeyboardInterrupt
        return path

    monkeypatch.setattr(native, "_save_run_checkpoint", interrupt_after_step_one)
    with pytest.raises(KeyboardInterrupt):
        native.run_lora_stage(config, stage, config.base_checkpoint, dataset, tmp_path / "split")
    monkeypatch.setattr(native, "_save_run_checkpoint", save)
    resumed = native.run_lora_stage(
        config,
        stage,
        config.base_checkpoint,
        dataset,
        tmp_path / "split",
        resume=tmp_path / "split" / "latest",
    )
    first = torch.load(Path(full) / "checkpoint.pt", weights_only=True)
    second = torch.load(Path(resumed) / "checkpoint.pt", weights_only=True)
    assert first["scheduler"] == second["scheduler"]
    torch.testing.assert_close(first["rng"]["torch"], second["rng"]["torch"], rtol=0, atol=0)
    for name in first["model"]:
        torch.testing.assert_close(first["model"][name], second["model"][name], rtol=0, atol=0)


def test_stack_rejects_pretraining_source_before_building_data(runnable_stack):
    values, _, _ = runnable_stack
    values["source_stage"] = "pretraining"
    config = StackSFTConfig.from_dict(values)
    with pytest.raises(ValueError, match="direct pretraining -> SFT is forbidden"):
        StackSFTRunner(config).run()
    assert not Path(config.data_dir).exists()


def test_stack_rejects_unknown_stage_before_outputs(runnable_stack):
    values, _, _ = runnable_stack
    config = StackSFTConfig.from_dict(values)
    with pytest.raises(ValueError, match="Unknown stack stage"):
        StackSFTRunner(config).run(start_stage="absent")
    assert not Path(config.data_dir).exists()
