"""Native replacements for the deleted 500M launcher's SFT contracts.

Fifty historical test responsibilities remain covered through package APIs.
V1 same-position fixtures now use strict V2 next-token labels and supervised
EOS; oversized examples fail closed. Runtime reasoning injection is replaced
by explicitly authored native targets. Truncated/reset probe LR schedules
are replaced by the full continuous horizon, and shell/parser assertions by
typed JSON/CLI contracts. Trained route tables must never be regenerated.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from torch.utils.data import DataLoader
from typer.testing import CliRunner

from complexity.cli.app import app
from complexity.core.losses.sft import sft_loss_from_hidden
from complexity.export.native_checkpoint import (
    load_native_checkpoint,
    normalize_native_checkpoint_config,
)
from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    render_inference_prompt,
)
from complexity.training.adapter_optimizer import build_finetuning_optimizer
from complexity.training.curriculum_planning import (
    CurriculumPlanningConfig,
    validate_sft_release_manifest,
)
from complexity.training.finetuning import (
    TEXT_SUPERVISED_FINETUNING,
    validate_full_parameter_finetuning,
    validate_training_stage_transition,
)
from complexity.training.sft import (
    EXPERIMENT,
    SFTTokenShardDataset,
    load_sft_config,
    load_sft_model,
    reference_epoch_batches,
    run_sft,
)
from complexity.training.sft_evaluation import (
    ValidationState,
    complete_epoch_eligible,
    discover_sft_evaluation_partitions,
    validate_evaluation_coverage,
)
from complexity.training.sft_examples import final_completion, thinking_completion
from complexity.training.sft_preparation import (
    encode_agentic_trajectory,
    encode_native_sft_record,
)
from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
    validate_shard_supervision,
)
from complexity.training.sft_weighting import WeightedSFTTokenShardDataset
from complexity.training.stack_sft.config import DatasetMix, SourceConfig
from complexity.training.stack_sft.datasets import StackSFTDatasetBuilder
from complexity.training.stack_sft.stage import SFTPromptCompletionDataset
from tests.test_sft_workflow import artifacts as artifacts
from tests.test_sft_workflow import digest, plan_for, write_partition
from tests.test_token_routed_to_tr_hash_conversion import _legacy_checkpoint


def _write_shard(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    template = default_chat_template()
    template["eos_token"] = "</s>"
    (root / "chat_template.json").write_text(json.dumps(template) + "\n")
    partition = write_partition(
        root,
        "train",
        [([10, 11, 12, 13], [-100, 12, 13, 0]), ([20, 21], [21, 0])],
    )
    examples = [
        {**row, "task": task}
        for row, task in zip(
            [json.loads(line) for line in (partition / "examples.jsonl").read_text().splitlines()],
            ("first_task", "second_task"),
        )
    ]
    (partition / "examples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in examples))
    metadata = json.loads((partition / "sft.idx.json").read_text())
    metadata["files"]["examples.jsonl"] = digest(partition / "examples.jsonl")
    (partition / "sft.idx.json").write_text(json.dumps(metadata))
    return SFTTokenShardDataset(root, sequence_length=5, vocab_size=32)


def _reader(root, seq_len=5):
    return SFTTokenShardDataset(root, sequence_length=seq_len, vocab_size=32)


def _config(artifacts, *, epochs=3, **kwargs):
    checkpoint, data = artifacts
    return {
        "format": "complexity-sft-experiment-v1",
        "checkpoint": str(checkpoint),
        "data": str(data),
        "experiment": EXPERIMENT,
        "batch_size": 1,
        "world_size": 1,
        "sequence_length": 8,
        "epochs": epochs,
        **kwargs,
    }


def _planning(**kwargs):
    return CurriculumPlanningConfig.from_dict(
        {
            "checkpoint": "base",
            "sft_bin": "data",
            "curriculum_config": "curriculum.yaml",
            "through_stage": "last",
            "output_root": "output",
            **kwargs,
        }
    )


class _Tokenizer:
    eos_token_id, pad_token_id = 0, 1

    def __init__(self, mapping=None):
        self.mapping = mapping

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return SimpleNamespace(ids=self.mapping[text] if self.mapping is not None else [3])

    @staticmethod
    def token_to_id(token):
        assert token == "</s>"
        return 0

    @staticmethod
    def get_vocab_size():
        return 32


def _jsonl_data(tmp_path, rows):
    source, prepared = tmp_path / "source.jsonl", tmp_path / "prepared.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    mix = DatasetMix(
        sources=[SourceConfig(name="local", weight=1, path=str(source))],
        records=len(rows),
    )
    StackSFTDatasetBuilder().build(mix, prepared)
    return SFTPromptCompletionDataset(
        prepared,
        _Tokenizer(),
        sequence_length=8,
        vocab_size=32,
        chat_template={**default_chat_template(), "eos_token": "</s>"},
    )


def _sampler_plan(*, items=2, batch_size=1, world_size=1, epochs=3):
    from complexity.training.sequence_packing import resolve_epoch_schedule

    schedule = resolve_epoch_schedule(
        items=items, batch_size_per_rank=batch_size, world_size=world_size, epochs=epochs
    )
    return {
        "packed_items": items,
        "batch_size": batch_size,
        "world_size": world_size,
        "epochs": epochs,
        "steps_per_epoch": schedule.steps_per_epoch,
        "seed": 42,
    }


def test_statistical_eval_requires_declared_ten_percent_coverage():
    report = validate_evaluation_coverage(
        500_000, 50_000, minimum_fraction=0.10, partition_name="validation"
    )
    assert report["fraction"] == 0.10
    with pytest.raises(ValueError, match="only 0.01%.*required minimum is 10.00%"):
        validate_evaluation_coverage(500_000, 28, minimum_fraction=0.10, partition_name="natural")


def test_legacy_text_adaptation_entrypoints_stay_removed():
    legacy_paths = (
        "deploy/supervisor/tr_hash_500m_32k_sft.conf",
        "deploy/supervisor/tr_hash_500m_lora_probe.conf",
        "deploy/supervisor/tr_hash_500m_lora_v19_epoch.conf",
        "deploy/supervisor/tr_hash_500m_sft_v18_full_e3.conf",
        "scripts/vast_probe_sft_500m_32k_tr_lora.sh",
        "scripts/vast_sft_500m_32k_v18_lora_epoch.sh",
        "scripts/vast_sft_500m_32k_v19_lora_epoch.sh",
        "scripts/vast_sft_500m_32k_v2_lora_epoch.sh",
        "scripts/vast_sft_500m_32k_v2_balanced_lora.sh",
        "scripts/vast_sft_500m_32k_v2_lora_3e_packed.sh",
        "scripts/vast_sft_200m_luciole_16way_full_3e.sh",
        "scripts/sft_tr_hash_200m_lora.sh",
    )
    assert all(not Path(path).exists() for path in legacy_paths)


def test_mlx_regression_panel_is_version_neutral_and_covers_greeting():
    panel = json.loads(Path("configs/sft_500m_mlx_panel.json").read_text())
    assert panel["id"] == "sft-500m-mlx-panel-v2"
    assert any(item["prompt"] == "Hello" for item in panel["prompts"])
    assert any(item["prompt"] == "What is 2 + 2?" for item in panel["prompts"])


def test_sft_bin_dataset_reads_and_pads_indexed_examples(tmp_path):
    dataset = _write_shard(tmp_path)
    first = dataset[0]
    assert first["input_ids"].tolist() == [10, 11, 12, 13, 0]
    assert first["labels"].tolist() == [-100, 12, 13, 0, -100]
    assert dataset.chat_template["id"] == CHAT_TEMPLATE_ID
    assert dataset.supervised_tokens == 5  # Both terminal EOS targets are included.


def test_sft_bin_packing_keeps_examples_labels_and_eos_boundary(tmp_path):
    _write_shard(tmp_path)
    dataset = _reader(tmp_path, seq_len=8)
    packed = dataset[0]
    assert len(dataset) == 1
    assert packed["input_ids"].tolist() == [10, 11, 12, 13, 0, 20, 21, 0]
    assert packed["labels"].tolist() == [-100, 12, 13, 0, -100, 21, 0, -100]
    weighted = WeightedSFTTokenShardDataset(dataset, {"first_task": 1.0, "second_task": 1.0})
    assert weighted[0]["loss_weight"].tolist() == [1, 1, 1, 1, 0, 1, 1, 0]


def test_sft_bin_packing_preserves_per_example_loss_weights(tmp_path):
    _write_shard(tmp_path)
    dataset = WeightedSFTTokenShardDataset(
        _reader(tmp_path, seq_len=8), {"first_task": 0.5, "second_task": 2.0}
    )
    assert dataset[0]["loss_weight"].tolist() == [0.5, 0.5, 0.5, 0.5, 0, 2, 2, 0]
    assert dataset[0]["labels"].tolist() == [-100, 12, 13, 0, -100, 21, 0, -100]


def test_sft_bin_merges_semantic_loss_sidecar_without_changing_tokens(tmp_path):
    original = _write_shard(tmp_path)
    before = digest(tmp_path / "train" / "input_ids.bin")
    sidecar = tmp_path / "train" / "loss_metadata.jsonl"
    sidecar.write_text(
        "".join(
            json.dumps(
                {
                    "example_id": str(index),
                    "domain": domain,
                    "mode": "chat",
                    "difficulty": "easy",
                }
            )
            + "\n"
            for index, domain in enumerate(("social_greeting", "addition"))
        )
    )
    dataset = WeightedSFTTokenShardDataset(original, {"first_task": 1.0, "second_task": 1.0})
    assert [row["domain"] for row in dataset.examples] == ["social_greeting", "addition"]
    torch.testing.assert_close(dataset[0]["input_ids"], original[0]["input_ids"])
    assert digest(tmp_path / "train" / "input_ids.bin") == before


def test_full_shard_loss_targets_keep_rows_and_weight_visible_tokens(tmp_path):
    _write_shard(tmp_path)
    curriculum = tmp_path / "curriculum.yaml"
    curriculum.write_text("""version: 1
seed: 42
stages:
  - name: full-shard-weighted
    max_examples: all
    epochs: 1
    lr: 1.0e-6
    balance_by: none
    loss_task_targets:
      first_task: 0.5
      second_task: 0.5
""")
    dataset = WeightedSFTTokenShardDataset.from_curriculum(
        _reader(tmp_path, seq_len=8), curriculum, "full-shard-weighted"
    )
    assert len(dataset.examples) == 2
    assert dataset.loss_target_audit["raw_task_supervised_tokens"] == {
        "first_task": 3,
        "second_task": 2,
    }
    assert dataset.task_loss_weights == pytest.approx(
        {
            "first_task": 5 / 6,
            "second_task": 5 / 4,
        }
    )
    assert dataset.loss_target_audit["weighted_task_shares"] == {
        "first_task": 0.5,
        "second_task": 0.5,
    }
    sample = dataset[0]
    hidden = torch.randn(1, 8, 32, requires_grad=True)
    output_weight = torch.eye(32, requires_grad=True)
    labels, weights = sample["labels"][None], sample["loss_weight"][None]
    loss = sft_loss_from_hidden(
        hidden, output_weight, labels, chunk_tokens=3, example_weights=weights
    )
    visible = labels != -100
    direct = F.cross_entropy(hidden[visible], labels[visible], reduction="none")
    expected = (direct * weights[visible]).sum() / weights[visible].sum()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert hidden.grad.count_nonzero() and output_weight.grad.count_nonzero()
    assert float(weights[labels == 0].sum()) == pytest.approx(5 / 6 + 5 / 4)


def test_weighted_sft_loss_applies_one_weight_to_each_examples_tokens():
    hidden = torch.tensor([[[3.0, 0.0]], [[0.0, 1.0]]])
    labels = torch.tensor([[0], [0]])
    per_example = F.cross_entropy(hidden.reshape(2, 2), labels.flatten(), reduction="none")
    loss = sft_loss_from_hidden(
        hidden,
        torch.eye(2),
        labels,
        chunk_tokens=1,
        example_weights=torch.tensor([0.5, 1.5]),
    )
    assert loss.item() == pytest.approx(((per_example[0] * 0.5 + per_example[1] * 1.5) / 2).item())


def test_weighted_sft_loss_normalizes_by_visible_weight_mass():
    hidden = torch.tensor([[[3.0, 0.0]], [[0.0, 1.0]]])
    labels = torch.tensor([[0], [0]])
    per_example = F.cross_entropy(hidden.reshape(2, 2), labels.flatten(), reduction="none")
    loss = sft_loss_from_hidden(
        hidden,
        torch.eye(2),
        labels,
        chunk_tokens=2,
        example_weights=torch.tensor([1.0, 3.0]),
    )
    assert loss.item() == pytest.approx(((per_example[0] + per_example[1] * 3.0) / 4).item())


def test_weighted_sft_loss_accepts_packed_token_weights():
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 1.0], [2.0, 0.0]]])
    labels = torch.tensor([[0, -100, 0]])
    per_token = F.cross_entropy(
        hidden[:, (0, 2), :].reshape(2, 2), torch.tensor([0, 0]), reduction="none"
    )
    loss = sft_loss_from_hidden(
        hidden,
        torch.eye(2),
        labels,
        chunk_tokens=2,
        example_weights=torch.tensor([[0.5, 0.0, 2.0]]),
    )
    assert loss.item() == pytest.approx(((per_token[0] * 0.5 + per_token[1] * 2) / 2.5).item())


def test_sft_bin_accepts_final_assistant_only_multi_turn_shards(tmp_path):
    dataset = _write_shard(tmp_path)
    assert dataset.metadata["format"] == SHARD_FORMAT_V2
    assert dataset.metadata["assistant_supervision"] == FINAL_ASSISTANT_SUPERVISION
    assert dataset.metadata["history_assistant_turns"] == MASKED_ASSISTANT_HISTORY


def test_final_assistant_only_shards_fail_without_masked_history_contract():
    with pytest.raises(ValueError, match="masked assistant history"):
        validate_shard_supervision(
            {
                "format": SHARD_FORMAT_V2,
                "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
            }
        )


def test_v2_shards_reject_unknown_supervision_modes():
    with pytest.raises(ValueError, match="requires all-assistant-turn or final"):
        validate_shard_supervision(
            {"format": SHARD_FORMAT_V2, "assistant_supervision": "ambiguous"}
        )


def test_release_ready_gate_rejects_failed_or_double_wrapped_reasoning_release(tmp_path):
    failed = {
        "quality_status": "failed",
        "release_quality": {
            "ready": False,
            "reasoning_envelope_version": "card-corpus-v2-think-final-v1",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(failed))
    with pytest.raises(ValueError, match="quality_status must be passed"):
        validate_sft_release_manifest(tmp_path, reasoning_envelope=False)
    passed = {
        "quality_status": "passed",
        "release_quality": {
            "ready": True,
            "reasoning_envelope_version": "card-corpus-v2-think-final-v1",
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(passed))
    assert validate_sft_release_manifest(tmp_path, reasoning_envelope=False) == passed
    with pytest.raises(ValueError, match="already contain reasoning envelopes"):
        validate_sft_release_manifest(tmp_path, reasoning_envelope=True)


def test_authored_native_reasoning_target_preserves_prompt_mask_and_final_answer():
    prompt = "<|user|>Question<|end_of_turn|><|assistant|>"
    completion = thinking_completion("Because", "Answer")
    tokenizer = _Tokenizer({prompt: [10, 11, 50], completion: [20, 21, 12, 13, 30, 31]})
    row = {"prompt": prompt, "completion": completion}
    encoded = encode_native_sft_record(tokenizer, row, seq_len=9)
    assert encoded["input_ids"].tolist() == [10, 11, 50, 20, 21, 12, 13, 30, 31]
    assert encoded["labels"].tolist() == [-100, -100, 20, 21, 12, 13, 30, 31, 0]
    assert row == {"prompt": prompt, "completion": completion}


def test_final_only_projection_uses_only_final_assistant_span():
    earlier, final = final_completion("History"), thinking_completion("Because", "Answer")
    mapping = {
        "<|user|>Question<|end_of_turn|>": [1, 2],
        "<|assistant|>": [3],
        earlier: [4, 5],
        "<|user|>Again<|end_of_turn|>": [6],
        final: [20, 8, 30],
    }
    encoded = encode_agentic_trajectory(
        _Tokenizer(mapping),
        {
            "trajectory": [
                {"role": "user", "content": "Question"},
                {"role": "assistant", "content": earlier},
                {"role": "user", "content": "Again"},
                {"role": "assistant", "content": final},
            ]
        },
        12,
        assistant_supervision=FINAL_ASSISTANT_SUPERVISION,
    )
    active = encoded["labels"] != -100
    assert encoded["labels"][active].tolist() == [20, 8, 30, 0]
    assert encoded["input_ids"][:7].tolist() == [1, 2, 3, 4, 5, 6, 3]


def test_messages_use_the_canonical_chat_template(tmp_path):
    source, output = tmp_path / "messages.jsonl", tmp_path / "formatted.jsonl"
    source.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "What is a hash route?"},
                    {"role": "assistant", "content": "A fixed mapping."},
                ]
            }
        )
        + "\n"
    )
    mix = DatasetMix([SourceConfig(name="messages", weight=1, path=str(source))], records=1)
    StackSFTDatasetBuilder().build(mix, output)
    row = json.loads(output.read_text())
    assert row["prompt"] == render_inference_prompt(
        "What is a hash route?", default_chat_template()
    )
    assert row["completion"] == "A fixed mapping."


def test_sft_bin_rejects_left_truncation_and_preserves_original_targets(tmp_path):
    dataset = _write_shard(tmp_path)
    before = digest(tmp_path / "train" / "labels.bin")
    with pytest.raises(ValueError, match="without truncation"):
        _reader(tmp_path, seq_len=2)
    assert dataset[0]["labels"].tolist() == [-100, 12, 13, 0, -100]
    assert digest(tmp_path / "train" / "labels.bin") == before


def test_sft_bin_eval_iterator_is_finite(tmp_path):
    dataset = _write_shard(tmp_path)
    rows = list(DataLoader(dataset, batch_size=1))
    assert len(rows) == 2
    assert sum(int((row["labels"] != -100).sum()) for row in rows) == 5


def test_early_stopping_waits_for_a_complete_epoch():
    assert not complete_epoch_eligible(312, steps_per_epoch=313, minimum_epochs=1)
    assert complete_epoch_eligible(313, steps_per_epoch=313, minimum_epochs=1)
    state = ValidationState(4.0, misses=3)
    assert not state.should_stop(patience=3, step=312, steps_per_epoch=313)
    assert state.should_stop(patience=3, step=313, steps_per_epoch=313)


def test_early_stopping_can_require_multiple_complete_epochs():
    assert not complete_epoch_eligible(625, steps_per_epoch=313, minimum_epochs=2)
    assert complete_epoch_eligible(626, steps_per_epoch=313, minimum_epochs=2)


def test_step_limited_probe_retains_the_complete_lr_schedule_horizon(artifacts, tmp_path):
    plan = plan_for(artifacts, epochs=100)
    output = tmp_path / "probe"
    result = run_sft(plan, output, device="cpu", steps=1)
    state = torch.load(output / "step_000001" / "checkpoint.pt", weights_only=True)
    assert result["step"] == 1 and result["planned_updates"] == 200
    assert plan["warmup_updates"] == 6
    assert state["scheduler"]["last_epoch"] == 1
    assert state["scheduler"]["_last_lr"][0] == pytest.approx(plan["learning_rate"] * 2 / 6)
    assert result["complete"] is False


def test_sft_bin_resolves_matched_and_natural_eval_partitions(tmp_path):
    _write_shard(tmp_path)
    for partition in ("diagnostic", "eval"):
        (tmp_path / partition).mkdir()
        (tmp_path / partition / "sft.idx.json").write_text("{}")
    partitions = discover_sft_evaluation_partitions(tmp_path)
    assert partitions.selection == tmp_path / "diagnostic"
    assert partitions.natural == tmp_path / "eval"


def test_sft_bin_legacy_eval_remains_the_selection_metric(tmp_path):
    _write_shard(tmp_path)
    target = tmp_path / "eval"
    target.mkdir()
    (target / "sft.idx.json").write_text("{}")
    partitions = discover_sft_evaluation_partitions(tmp_path)
    assert partitions.selection == target and partitions.natural is None


def test_sft_bin_epoch_budget_visits_every_example_exactly_three_times(tmp_path):
    dataset = _write_shard(tmp_path)
    plan = _sampler_plan(items=len(dataset), epochs=3)
    visits = Counter(
        item
        for epoch in range(3)
        for ranks in reference_epoch_batches(plan, epoch)
        for batch in ranks
        for item in batch
    )
    assert visits == {0: 3, 1: 3}


def test_sft_bin_epoch_padding_preserves_every_batch_boundary(tmp_path):
    dataset = _write_shard(tmp_path)
    plan = _sampler_plan(items=len(dataset), epochs=3, batch_size=3)
    epochs = [list(reference_epoch_batches(plan, epoch)) for epoch in range(3)]
    assert sum(len(ranks[0]) for updates in epochs for ranks in updates) == 9
    assert all(len(updates) == 1 and len(updates[0][0]) == 3 for updates in epochs)
    assert all(set(updates[0][0]) == {0, 1} for updates in epochs)


def test_sft_bin_resume_cursor_skips_completed_batches(artifacts, tmp_path):
    plan = plan_for(artifacts, epochs=2)
    output = tmp_path / "resume"
    run_sft(plan, output, device="cpu", steps=1)
    run_sft(plan, output, device="cpu", steps=2, resume=output / "latest")
    records = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert [row["epoch"] for row in records] == [1, 1]


def test_packed_sft_resume_cursor_skips_completed_batches(artifacts, tmp_path):
    plan = plan_for(artifacts, epochs=2, batch_size=2)
    assert plan["steps_per_epoch"] == 1
    output = tmp_path / "packed-resume"
    run_sft(plan, output, device="cpu", steps=1)
    run_sft(plan, output, device="cpu", steps=2, resume=output / "latest")
    records = [json.loads(row) for row in (output / "metrics.jsonl").read_text().splitlines()]
    assert [row["step"] for row in records] == [1, 2]
    assert [row["epoch"] for row in records] == [1, 2]


def test_epoch_padding_gives_sparse_distributed_ranks_equal_batch_counts():
    plan = _sampler_plan(items=3, world_size=2, batch_size=2)
    first, second = next(reference_epoch_batches(plan, 0))
    assert len(first) == len(second) == 2
    assert set(first + second) == {0, 1, 2}
    sparse = _sampler_plan(items=1, world_size=4, batch_size=2)
    assert next(reference_epoch_batches(sparse, 0)) == [[0, 0]] * 4


def test_sft_config_rejects_two_dataset_sources(artifacts, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(artifacts, jsonl="second-source.jsonl")))
    with pytest.raises(ValueError, match="Unknown SFT configuration keys.*jsonl"):
        load_sft_config(path)
    result = CliRunner().invoke(app, ["train", "sft", str(path)])
    assert result.exit_code == 2 and "jsonl" in result.stderr


def test_sft_jsonl_eval_iterator_is_finite(tmp_path):
    dataset = _jsonl_data(tmp_path, [{"instruction": "Answer.", "output": "Done."}])
    assert len(dataset.examples) == 1
    assert len(list(DataLoader(dataset, batch_size=1))) == 1
    assert int((dataset[0]["labels"] != -100).sum()) == 2


def test_sft_jsonl_packs_complete_examples_by_default(tmp_path):
    dataset = _jsonl_data(
        tmp_path,
        [
            {"instruction": "First", "output": "One"},
            {"instruction": "Second", "output": "Two"},
        ],
    )
    assert dataset.packing_plan.packed_items == 1
    rows = list(DataLoader(dataset, batch_size=1))
    assert len(rows) == 1 and rows[0]["input_ids"].shape == (1, 8)
    assert int(torch.count_nonzero(rows[0]["labels"] != -100)) == 4
    assert rows[0]["labels"].tolist() == [[3, 0, -100, 3, 0, -100, -100, -100]]


def test_only_deterministic_buffers_are_tolerated_when_loading_exported_weights(
    artifacts, tmp_path
):
    checkpoint, _ = artifacts
    state = load_file(str(checkpoint / "model.safetensors"))
    rope = [name for name in state if name.endswith(".inv_freq")]
    assert rope
    for name in rope:
        state.pop(name)
    save_file(state, str(checkpoint / "model.safetensors"))
    model = load_native_checkpoint(checkpoint)
    assert torch.equal(model.embed_tokens.weight, state["embed_tokens.weight"])
    legacy, config = _legacy_checkpoint()
    for name in [name for name in legacy if name.endswith("topk_token_to_expert")]:
        legacy.pop(name)
    historical = tmp_path / "missing-trained-routes"
    historical.mkdir()
    (historical / "config.json").write_text(json.dumps(config))
    save_file(legacy, str(historical / "model.safetensors"))
    with pytest.raises(ValueError, match="lacks trained route tables"):
        load_native_checkpoint(historical)


def test_sft_loads_huggingface_safetensors_export(artifacts):
    checkpoint, _ = artifacts
    model = load_native_checkpoint(checkpoint)
    expected = load_file(str(checkpoint / "model.safetensors"))
    assert model.config.hidden_size == 16
    torch.testing.assert_close(
        model.embed_tokens.weight, expected["embed_tokens.weight"], rtol=0, atol=0
    )


def test_checkpoint_config_restores_huggingface_tr_hash_aliases():
    config = normalize_native_checkpoint_config(
        {
            "hidden_size": 640,
            "num_hidden_layers": 10,
            "num_attention_heads": 10,
            "num_key_value_heads": 2,
            "vocab_size": 32_000,
            "mlp_type": "tr_hash_engine",
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "shared_intermediate_size": 1_824,
        },
        {"layers.0.mlp.engine.expert_gate": torch.empty(4, 640, 456, device="meta")},
    )
    assert config.top_k == 2
    assert config.intermediate_size == 1_824


def test_sft_canonicalizes_legacy_token_routed_hf_export(tmp_path):
    state, config = _legacy_checkpoint(seed=5, num_hidden_layers=1)
    (tmp_path / "config.json").write_text(json.dumps(config))
    save_file(state, str(tmp_path / "model.safetensors"))
    model = load_native_checkpoint(tmp_path)
    assert model.config.mlp_type == "tr_hash_engine"
    torch.testing.assert_close(
        model.layers[0].mlp.engine.expert_gate, state["layers.0.mlp.gate_proj_w"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        model.layers[0].mlp.engine.route_table,
        state["layers.0.mlp.topk_token_to_expert"],
        rtol=0,
        atol=0,
    )


def test_sft_config_accepts_explicit_local_evaluation_source(artifacts, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(artifacts, execution={"eval_data": "held-out"})))
    plan, execution = load_sft_config(path)
    assert plan["data"] == str(artifacts[1])
    assert execution["eval_data"] == str(tmp_path / "held-out")


def test_sft_early_stopping_state_resets_only_on_real_improvement():
    improved, state = ValidationState(4.0, 2).observe(3.98, min_delta=0.01)
    assert improved is True and state.best_loss == 3.98 and state.misses == 0
    improved, state = state.observe(3.975, min_delta=0.01)
    assert improved is False and state.best_loss == 3.98 and state.misses == 1


def test_initial_validation_is_the_stage_checkpoint_baseline():
    state = ValidationState.from_baseline(3.356103)
    improved, state = state.observe(3.390597, min_delta=0.0)
    assert improved is False and state.best_loss == 3.356103 and state.misses == 1


def test_curriculum_config_exposes_conservative_adaptation_controls():
    config = _planning()
    assert config.lora_rank == 16 and config.lora_alpha == 16
    assert config.expert_lr_multiplier == pytest.approx(0.25)
    assert config.require_release_ready is True and config.pack_sequences is True
    state = ValidationState(4.0, 3)
    assert state.should_stop(patience=3, step=313, steps_per_epoch=313)


@pytest.mark.parametrize("rank", [0, -1])
def test_sft_config_rejects_nonpositive_lora_rank(rank):
    with pytest.raises(ValueError, match="positive LoRA rank"):
        _planning(lora_rank=rank)


def test_native_cli_selects_adaptation_mode_and_defaults_to_preflight(artifacts, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(artifacts)))
    result = CliRunner().invoke(app, ["train", "sft", str(path)])
    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["training_started"] is False and plan["release_eligible"] is False
    assert plan["experiment"] == EXPERIMENT
    assert plan["expert_lr_scale"] == 1.0
    # Full-parameter ablation and preservation-first LoRA have separate APIs.
    assert _planning().lora_rank > 0


def test_full_parameter_model_and_optimizer_cover_every_parameter(artifacts):
    checkpoint, _ = artifacts
    model = load_sft_model(checkpoint)
    assert all(parameter.requires_grad for parameter in model.parameters())
    optimizer = build_finetuning_optimizer(
        model, learning_rate=2e-5, expert_lr_scale=1, weight_decay=0.1, betas=(0.9, 0.95)
    )
    parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(parameters) == len({id(parameter) for parameter in parameters})
    assert {id(parameter) for parameter in parameters} == {
        id(parameter) for parameter in model.parameters()
    }


def test_sft_runtime_requires_checkpoint_stage_provenance():
    with pytest.raises(ValueError, match="requires source_stage.*refinement"):
        validate_full_parameter_finetuning(TEXT_SUPERVISED_FINETUNING)
    validate_full_parameter_finetuning(TEXT_SUPERVISED_FINETUNING, source_stage="refinement")


def test_sft_release_contract_rejects_direct_pretraining_source():
    with pytest.raises(ValueError, match="direct pretraining -> SFT is forbidden"):
        validate_training_stage_transition("text", "pretraining", "supervised-finetuning")


def test_sft_config_supports_a_finite_epoch_budget(artifacts, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(artifacts, epochs=3)))
    plan, _ = load_sft_config(path)
    assert plan["epochs"] == 3 and plan["optimizer_updates"] == 6


def test_sft_config_supports_automatic_complete_epoch_schedule(artifacts, tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config(artifacts)))
    plan, _ = load_sft_config(path)
    assert plan["packed_items"] == 2 and plan["packing"] == "ec721a2_next_fit_one_ignored_eos"
    assert plan["steps_per_epoch"] == 2 and plan["optimizer_updates"] == 6
    assert list(reference_epoch_batches(plan, 2))


def test_sft_native_state_supports_exact_resume(artifacts, tmp_path):
    plan = plan_for(artifacts)
    full, split = tmp_path / "full", tmp_path / "split"
    run_sft(plan, full, device="cpu", steps=2)
    run_sft(plan, split, device="cpu", steps=1)
    run_sft(plan, split, device="cpu", steps=2, resume=split / "latest")
    expected = torch.load(full / "step_000002" / "checkpoint.pt", weights_only=True)
    actual = torch.load(split / "step_000002" / "checkpoint.pt", weights_only=True)
    assert actual["step"] == 2 and actual["scheduler"] == expected["scheduler"]
    assert actual["history"] == expected["history"]
    torch.testing.assert_close(actual["rng"]["torch"], expected["rng"]["torch"], rtol=0, atol=0)
    for name, tensor in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], tensor, rtol=0, atol=0)
    for key, state in expected["optimizer"]["state"].items():
        for name, tensor in state.items():
            torch.testing.assert_close(
                actual["optimizer"]["state"][key][name], tensor, rtol=0, atol=0
            )


def test_sft_resume_rejects_a_packing_configuration_change(artifacts, tmp_path):
    plan = plan_for(artifacts)
    output = tmp_path / "run"
    run_sft(plan, output, device="cpu", steps=1)
    changed = copy.deepcopy(plan)
    changed["packing"] = "invented_without_eos"
    with pytest.raises(ValueError, match="identity"):
        run_sft(changed, output, device="cpu", steps=2, resume=output / "latest")
