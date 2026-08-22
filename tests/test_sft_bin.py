from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import save_file as save_safetensors

import scripts.sft_500m_32k_tr as sft_module
from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    render_inference_prompt,
)
from complexity.training.sft_shard import (
    FINAL_ASSISTANT_SUPERVISION,
    MASKED_ASSISTANT_HISTORY,
    SHARD_FORMAT_V2,
    validate_shard_supervision,
)
from scripts.sft_500m_32k_tr import (
    SFTBinDataset,
    SFTJsonlDataset,
    apply_reasoning_envelope,
    build_parser,
    configure_sft_parameters,
    early_stopping_is_eligible,
    format_record,
    load_checkpoint_state,
    load_model_state_compat,
    lr_schedule_horizon,
    pad_epoch_items,
    resolve_sft_bin_evaluation_partitions,
    sft_loss_from_hidden,
    update_early_stopping,
    validate_evaluation_sample_fraction,
    validate_resume_state,
    validate_sft_release_manifest,
    validation_baseline,
)


def test_statistical_eval_requires_ten_percent_coverage() -> None:
    validate_evaluation_sample_fraction(
        500_000,
        50_000,
        minimum_fraction=0.10,
        partition_name="validation",
    )
    with pytest.raises(ValueError, match="only 0.01%.*required minimum is 10.00%"):
        validate_evaluation_sample_fraction(
            500_000,
            28,
            minimum_fraction=0.10,
            partition_name="natural",
        )


def test_legacy_text_adaptation_entrypoints_stay_removed() -> None:
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


def test_mlx_regression_panel_is_version_neutral_and_covers_greeting() -> None:
    panel = json.loads(Path("configs/sft_500m_mlx_panel.json").read_text())

    assert panel["id"] == "sft-500m-mlx-panel-v2"
    assert any(item["prompt"] == "Hello" for item in panel["prompts"])
    assert any(item["prompt"] == "What is 2 + 2?" for item in panel["prompts"])


def _write_shard(root: Path) -> None:
    train = root / "train"
    train.mkdir(parents=True)
    (root / "chat_template.json").write_text(
        json.dumps(default_chat_template()) + "\n",
        encoding="utf-8",
    )
    input_ids = np.asarray([10, 11, 12, 13, 20, 21], dtype="<u4")
    labels = np.asarray([-100, -100, 12, 13, -100, 21], dtype="<i4")
    input_ids.tofile(train / "input_ids.bin")
    labels.tofile(train / "labels.bin")
    examples = [
        {
            "example_id": "first",
            "task": "test",
            "offset": 0,
            "num_tokens": 4,
            "supervised_tokens": 2,
        },
        {
            "example_id": "second",
            "task": "test",
            "offset": 4,
            "num_tokens": 2,
            "supervised_tokens": 1,
        },
    ]
    with (train / "examples.jsonl").open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example) + "\n")
    (train / "sft.idx.json").write_text(
        json.dumps(
            {
                "format": "complexity-sft-token-shard-v1",
                "chat_template_id": CHAT_TEMPLATE_ID,
                "examples": 2,
                "num_tokens": 6,
                "supervised_tokens": 3,
                "eos_token_id": 199999,
            }
        )
    )


def test_sft_bin_dataset_reads_and_pads_indexed_examples(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(tmp_path, seq_len=5, seed=42, rank=0, world_size=1)
    first = dataset._tensor_example(dataset.examples[0])
    assert first["input_ids"].tolist() == [10, 11, 12, 13, 199999]
    assert first["labels"].tolist() == [-100, -100, 12, 13, -100]
    assert dataset.chat_template["id"] == CHAT_TEMPLATE_ID
    assert first["loss_weight"].item() == pytest.approx(1.0)


def test_sft_bin_packing_keeps_examples_labels_and_eos_boundary(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(
        tmp_path,
        seq_len=8,
        seed=42,
        rank=0,
        world_size=1,
        pack_sequences=True,
    )

    assert dataset.training_items == 1
    packed = dataset._tensor_pack(dataset.packed_examples[0])
    assert packed["input_ids"].tolist() == [
        10,
        11,
        12,
        13,
        199999,
        20,
        21,
        199999,
    ]
    assert packed["labels"].tolist() == [
        -100,
        -100,
        12,
        13,
        -100,
        -100,
        21,
        -100,
    ]
    assert packed["loss_weight"].tolist() == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0]
    )


def test_sft_bin_packing_preserves_per_example_loss_weights(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(
        tmp_path,
        seq_len=8,
        seed=42,
        rank=0,
        world_size=1,
        pack_sequences=True,
    )
    dataset.examples[0]["task"] = "first_task"
    dataset.examples[1]["task"] = "second_task"
    dataset.task_loss_weights = {"first_task": 0.5, "second_task": 2.0}

    packed = dataset._tensor_pack(dataset.packed_examples[0])

    assert packed["loss_weight"].tolist() == pytest.approx(
        [0.5, 0.5, 0.5, 0.5, 0.0, 2.0, 2.0, 0.0]
    )


def test_sft_bin_merges_semantic_loss_sidecar_without_changing_tokens(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path)
    sidecar = tmp_path / "train" / "loss_metadata.jsonl"
    sidecar.write_text(
        "\n".join(
            (
                json.dumps(
                    {
                        "example_id": "first",
                        "domain": "social_greeting",
                        "mode": "chat",
                        "difficulty": "easy",
                    }
                ),
                json.dumps(
                    {
                        "example_id": "second",
                        "domain": "addition",
                        "mode": "chat",
                        "difficulty": "easy",
                    }
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = SFTBinDataset(tmp_path, seq_len=5, seed=42, rank=0, world_size=1)

    assert [example["domain"] for example in dataset.examples] == [
        "social_greeting",
        "addition",
    ]
    assert dataset._tensor_example(dataset.examples[0])["input_ids"].tolist() == [
        10,
        11,
        12,
        13,
        199999,
    ]


def test_full_shard_loss_targets_keep_rows_and_weight_visible_tokens(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path)
    examples_path = tmp_path / "train" / "examples.jsonl"
    examples = [json.loads(line) for line in examples_path.read_text().splitlines()]
    examples[0]["task"] = "long_task"
    examples[1]["task"] = "short_task"
    examples_path.write_text(
        "".join(json.dumps(example) + "\n" for example in examples),
        encoding="utf-8",
    )
    curriculum = tmp_path / "curriculum.yaml"
    curriculum.write_text(
        """version: 1
seed: 42
stages:
  - name: full-shard-weighted
    max_examples: all
    epochs: 1
    lr: 1.0e-6
    balance_by: none
    loss_task_targets:
      long_task: 0.5
      short_task: 0.5
""",
        encoding="utf-8",
    )

    dataset = SFTBinDataset(
        tmp_path,
        seq_len=5,
        seed=42,
        rank=0,
        world_size=1,
        curriculum_config=curriculum,
        curriculum_stage="full-shard-weighted",
    )

    assert len(dataset.examples) == 2
    assert dataset.loss_target_audit["task_visible_supervised_tokens"] == {
        "long_task": 2,
        "short_task": 1,
    }
    assert dataset.task_loss_weights == pytest.approx(
        {"long_task": 0.75, "short_task": 1.5}
    )
    rendered_weights = {
        example["task"]: dataset._tensor_example(example)["loss_weight"].item()
        for example in dataset.examples
    }
    assert rendered_weights == pytest.approx(
        {"long_task": 0.75, "short_task": 1.5}
    )


def test_weighted_sft_loss_applies_one_weight_to_each_examples_tokens() -> None:
    hidden = torch.tensor([[[3.0, 0.0]], [[0.0, 1.0]]])
    output_weight = torch.eye(2)
    labels = torch.tensor([[0], [0]])
    example_weights = torch.tensor([0.5, 1.5])
    per_example = torch.nn.functional.cross_entropy(
        hidden.reshape(2, 2),
        output_weight.new_tensor([0, 0], dtype=torch.long),
        reduction="none",
    )

    loss = sft_loss_from_hidden(
        hidden,
        output_weight,
        labels,
        chunk_tokens=1,
        example_weights=example_weights,
    )

    assert loss.item() == pytest.approx(
        ((per_example[0] * 0.5 + per_example[1] * 1.5) / 2).item()
    )


def test_weighted_sft_loss_normalizes_by_visible_weight_mass() -> None:
    hidden = torch.tensor([[[3.0, 0.0]], [[0.0, 1.0]]])
    output_weight = torch.eye(2)
    labels = torch.tensor([[0], [0]])
    example_weights = torch.tensor([1.0, 3.0])
    per_example = torch.nn.functional.cross_entropy(
        hidden.reshape(2, 2),
        output_weight.new_tensor([0, 0], dtype=torch.long),
        reduction="none",
    )

    loss = sft_loss_from_hidden(
        hidden,
        output_weight,
        labels,
        chunk_tokens=2,
        example_weights=example_weights,
    )

    assert loss.item() == pytest.approx(
        ((per_example[0] + per_example[1] * 3.0) / 4.0).item()
    )


def test_weighted_sft_loss_accepts_packed_token_weights() -> None:
    hidden = torch.tensor([[[3.0, 0.0], [0.0, 1.0], [2.0, 0.0]]])
    output_weight = torch.eye(2)
    labels = torch.tensor([[0, -100, 0]])
    token_weights = torch.tensor([[0.5, 0.0, 2.0]])
    per_token = torch.nn.functional.cross_entropy(
        hidden[:, (0, 2), :].reshape(2, 2),
        output_weight.new_tensor([0, 0], dtype=torch.long),
        reduction="none",
    )

    loss = sft_loss_from_hidden(
        hidden,
        output_weight,
        labels,
        chunk_tokens=2,
        example_weights=token_weights,
    )

    assert loss.item() == pytest.approx(
        ((per_token[0] * 0.5 + per_token[1] * 2.0) / 2.5).item()
    )


def test_sft_bin_accepts_final_assistant_only_multi_turn_shards(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path)
    index_path = tmp_path / "train" / "sft.idx.json"
    metadata = json.loads(index_path.read_text())
    metadata.update(
        {
            "format": SHARD_FORMAT_V2,
            "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
            "history_assistant_turns": MASKED_ASSISTANT_HISTORY,
        }
    )
    index_path.write_text(json.dumps(metadata))

    dataset = SFTBinDataset(tmp_path, seq_len=5, seed=42, rank=0, world_size=1)

    assert dataset.metadata["assistant_supervision"] == FINAL_ASSISTANT_SUPERVISION


def test_final_assistant_only_shards_fail_without_masked_history_contract() -> None:
    metadata = {
        "format": SHARD_FORMAT_V2,
        "assistant_supervision": FINAL_ASSISTANT_SUPERVISION,
    }

    with pytest.raises(ValueError, match="masked assistant history"):
        validate_shard_supervision(metadata)


def test_v2_shards_reject_unknown_supervision_modes() -> None:
    with pytest.raises(ValueError, match="requires all-assistant-turn or final"):
        validate_shard_supervision(
            {"format": SHARD_FORMAT_V2, "assistant_supervision": "ambiguous"}
        )


def test_release_ready_gate_rejects_failed_or_double_wrapped_reasoning_release(
    tmp_path: Path,
) -> None:
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
    assert validate_sft_release_manifest(
        tmp_path, reasoning_envelope=False
    ) == passed
    with pytest.raises(ValueError, match="already contain reasoning envelopes"):
        validate_sft_release_manifest(tmp_path, reasoning_envelope=True)


def test_reasoning_envelope_preserves_prompt_mask_and_final_answer() -> None:
    inputs = np.asarray([10, 11, 50, 12, 13], dtype=np.int64)
    labels = np.asarray([-100, -100, 12, 13, 99], dtype=np.int64)

    rebuilt_inputs, rebuilt_labels = apply_reasoning_envelope(
        inputs,
        labels,
        prefix_ids=[20, 21],
        suffix_ids=[30, 31],
        eos_token_id=99,
        seq_len=9,
    )

    assert rebuilt_inputs.tolist() == [10, 11, 50, 20, 21, 12, 13, 30, 31]
    assert rebuilt_labels.tolist() == [-100, -100, 20, 21, 12, 13, 30, 31, 99]


def test_reasoning_envelope_uses_only_final_assistant_span() -> None:
    inputs = np.asarray([1, 2, 3, 4, 5, 6], dtype=np.int64)
    labels = np.asarray([-100, 3, 99, -100, 6, 99], dtype=np.int64)

    _, rebuilt_labels = apply_reasoning_envelope(
        inputs,
        labels,
        prefix_ids=[20],
        suffix_ids=[30],
        eos_token_id=99,
        seq_len=8,
    )

    assert rebuilt_labels[rebuilt_labels != -100].tolist() == [20, 6, 30, 99]


def test_messages_use_the_canonical_chat_template() -> None:
    template = default_chat_template()
    prompt, completion = format_record(
        {
            "messages": [
                {"role": "user", "content": "What is a hash route?"},
                {"role": "assistant", "content": "A fixed mapping."},
            ]
        },
        template,
    )
    assert prompt == render_inference_prompt("What is a hash route?", template)
    assert completion == "A fixed mapping."


def test_sft_bin_dataset_truncates_from_left_to_keep_response(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(tmp_path, seq_len=2, seed=42, rank=0, world_size=1)
    first = dataset._tensor_example(dataset.examples[0])
    assert first["input_ids"].tolist() == [12, 13]
    assert first["labels"].tolist() == [12, 13]


def test_sft_bin_eval_iterator_is_finite(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(
        tmp_path,
        seq_len=5,
        seed=42,
        rank=0,
        world_size=1,
        repeat=False,
    )
    assert len(list(dataset)) == 2


def test_early_stopping_waits_for_a_complete_epoch() -> None:
    assert not early_stopping_is_eligible(
        312,
        steps_per_epoch=313,
        minimum_epochs=1,
    )
    assert early_stopping_is_eligible(
        313,
        steps_per_epoch=313,
        minimum_epochs=1,
    )


def test_early_stopping_can_require_multiple_complete_epochs() -> None:
    assert not early_stopping_is_eligible(
        625,
        steps_per_epoch=313,
        minimum_epochs=2,
    )
    assert early_stopping_is_eligible(
        626,
        steps_per_epoch=313,
        minimum_epochs=2,
    )


def test_step_limited_probe_uses_its_actual_lr_schedule_horizon() -> None:
    assert lr_schedule_horizon(500, 2_962) == 500
    assert lr_schedule_horizon(5_000, 2_962) == 2_962
    assert lr_schedule_horizon(5_000, 2_962, reset_each_epoch=False) == 5_000


def test_sft_bin_resolves_matched_and_natural_eval_partitions(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path)
    train = tmp_path / "train"
    for partition in ("diagnostic", "eval"):
        target = tmp_path / partition
        target.mkdir()
        for name in (
            "input_ids.bin",
            "labels.bin",
            "examples.jsonl",
            "sft.idx.json",
        ):
            (target / name).write_bytes((train / name).read_bytes())

    matched, natural = resolve_sft_bin_evaluation_partitions(tmp_path)

    assert matched == tmp_path / "diagnostic"
    assert natural == tmp_path / "eval"


def test_sft_bin_legacy_eval_remains_the_selection_metric(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    train = tmp_path / "train"
    target = tmp_path / "eval"
    target.mkdir()
    for name in (
        "input_ids.bin",
        "labels.bin",
        "examples.jsonl",
        "sft.idx.json",
    ):
        (target / name).write_bytes((train / name).read_bytes())

    matched, natural = resolve_sft_bin_evaluation_partitions(tmp_path)

    assert matched == target
    assert natural is None


def test_sft_bin_epoch_budget_visits_every_example_exactly_three_times(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(
        tmp_path,
        seq_len=5,
        seed=42,
        rank=0,
        world_size=1,
        epochs=3,
    )
    assert len(list(dataset)) == 6


def test_sft_bin_epoch_padding_preserves_every_batch_boundary(
    tmp_path: Path,
) -> None:
    _write_shard(tmp_path)
    dataset = SFTBinDataset(
        tmp_path,
        seq_len=5,
        seed=42,
        rank=0,
        world_size=1,
        epochs=3,
        epoch_batch_size=3,
    )

    # Two examples become one complete three-example batch per epoch. The
    # epochs are not concatenated into only two batches.
    assert len(list(dataset)) == 9


def test_sft_bin_resume_cursor_skips_completed_batches(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    complete = SFTBinDataset(
        tmp_path,
        seq_len=5,
        seed=42,
        rank=0,
        world_size=1,
        epochs=2,
        epoch_batch_size=1,
    )
    resumed = SFTBinDataset(
        tmp_path,
        seq_len=5,
        seed=42,
        rank=0,
        world_size=1,
        epochs=2,
        epoch_batch_size=1,
        start_step=1,
    )

    complete_ids = [item["input_ids"].tolist() for item in complete]
    resumed_ids = [item["input_ids"].tolist() for item in resumed]

    assert resumed_ids == complete_ids[1:]


def test_packed_sft_resume_cursor_skips_completed_batches(tmp_path: Path) -> None:
    _write_shard(tmp_path)
    common = {
        "seq_len": 5,
        "seed": 42,
        "rank": 0,
        "world_size": 1,
        "epochs": 2,
        "epoch_batch_size": 1,
        "pack_sequences": True,
    }
    complete = SFTBinDataset(tmp_path, **common)
    resumed = SFTBinDataset(tmp_path, start_step=1, **common)

    complete_ids = [item["input_ids"].tolist() for item in complete]
    resumed_ids = [item["input_ids"].tolist() for item in resumed]

    assert resumed_ids == complete_ids[1:]


def test_epoch_padding_gives_sparse_distributed_ranks_equal_batch_counts() -> None:
    all_items = [0, 1, 2]
    first_rank = pad_epoch_items(
        [0, 2],
        all_items=all_items,
        rank=0,
        world_size=2,
        batch_size=2,
    )
    second_rank = pad_epoch_items(
        [1],
        all_items=all_items,
        rank=1,
        world_size=2,
        batch_size=2,
    )

    assert len(first_rank) == len(second_rank) == 2


def test_sft_parser_rejects_two_dataset_sources() -> None:
    parser = build_parser()
    try:
        parser.parse_args(
            [
                "--checkpoint",
                "checkpoint",
                "--jsonl",
                "data.jsonl",
                "--sft-bin",
                "data-bin",
            ]
        )
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("JSONL and SFT bin inputs must be mutually exclusive")


def test_sft_jsonl_eval_iterator_is_finite(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        json.dumps({"instruction": "Answer.", "output": "Done."}) + "\n",
        encoding="utf-8",
    )
    dataset = SFTJsonlDataset(
        str(path),
        "unused-tokenizer",
        seq_len=32,
        seed=42,
        rank=0,
        world_size=1,
        repeat=False,
    )

    # The tokenizer is loaded lazily, so the finite-dataset contract can be
    # tested independently through the stored records and flag.
    assert len(dataset.records) == 1
    assert dataset.repeat is False


def test_sft_jsonl_packs_complete_examples_by_default(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 0

        @staticmethod
        def encode(text: str, add_special_tokens: bool = False) -> list[int]:
            del text, add_special_tokens
            return [3]

    monkeypatch.setattr(sft_module.Tokenizer, "load", lambda _path: FakeTokenizer())
    path = tmp_path / "train.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"instruction": "First", "output": "One"}),
                json.dumps({"instruction": "Second", "output": "Two"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = SFTJsonlDataset(
        str(path),
        "fake-tokenizer",
        seq_len=8,
        seed=42,
        rank=0,
        world_size=1,
        repeat=False,
    )

    assert dataset.pack_sequences is True
    assert dataset.training_items == 1
    rows = list(dataset)
    assert len(rows) == 1
    assert rows[0]["input_ids"].shape == (8,)
    assert int(torch.count_nonzero(rows[0]["labels"] != -100)) == 4


def test_derived_route_buffers_are_tolerated_when_loading_exported_weights() -> None:
    class HistoricalRouteModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.register_buffer("topk_token_to_expert", torch.zeros(2, 4))
            self.register_buffer("pair_hash_route_codes", torch.zeros(2, 4))
            self.register_buffer("pair_hash_expert_pairs", torch.zeros(2, 4))

    model = HistoricalRouteModel()
    load_model_state_compat(model, {"weight": torch.tensor([2.0])})
    assert model.weight.item() == 2.0


def test_sft_loads_huggingface_safetensors_export(tmp_path: Path) -> None:
    config = {
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 8,
        "vocab_size": 32,
    }
    (tmp_path / "config.json").write_text(
        json.dumps(config),
        encoding="utf-8",
    )
    save_safetensors(
        {"weight": torch.arange(4, dtype=torch.float32)},
        str(tmp_path / "model.safetensors"),
    )

    checkpoint_dir, state = load_checkpoint_state(tmp_path)

    assert checkpoint_dir == tmp_path
    assert state["config"] == config
    assert state["export_format"] == "huggingface_safetensors"
    assert state["model"]["weight"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_sft_canonicalizes_legacy_token_routed_hf_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_config = {
        "mlp_type": "token_routed",
        "hidden_size": 16,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": 8,
        "vocab_size": 32,
    }
    (tmp_path / "config.json").write_text(json.dumps(legacy_config))
    save_safetensors(
        {"legacy.weight": torch.arange(4, dtype=torch.float32)},
        str(tmp_path / "model.safetensors"),
    )
    calls = []

    class ConvertedConfig:
        def to_dict(self):
            return {**legacy_config, "mlp_type": "tr_hash_engine"}

    class ConvertedModel:
        config = ConvertedConfig()

        def state_dict(self):
            return {"engine.weight": torch.arange(4, dtype=torch.float32)}

    def fake_convert(state_dict, config):
        calls.append((state_dict, config))
        return ConvertedModel()

    monkeypatch.setattr(
        "scripts.sft_500m_32k_tr.convert_token_routed_checkpoint",
        fake_convert,
    )

    _, state = load_checkpoint_state(tmp_path)

    assert len(calls) == 1
    assert calls[0][1]["mlp_type"] == "token_routed"
    assert state["config"]["mlp_type"] == "tr_hash_engine"
    assert list(state["model"]) == ["engine.weight"]


def test_sft_parser_accepts_jsonl_evaluation_source() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--jsonl",
            "train.jsonl",
            "--eval-jsonl",
            "eval.jsonl",
        ]
    )
    assert args.jsonl == "train.jsonl"
    assert args.eval_jsonl == "eval.jsonl"


def test_sft_early_stopping_state_resets_only_on_real_improvement() -> None:
    improved, best, misses = update_early_stopping(
        4.0,
        2,
        3.98,
        min_delta=0.01,
    )
    assert improved is True
    assert best == 3.98
    assert misses == 0

    improved, best, misses = update_early_stopping(
        best,
        misses,
        3.975,
        min_delta=0.01,
    )
    assert improved is False
    assert best == 3.98
    assert misses == 1


def test_initial_validation_is_the_stage_checkpoint_baseline() -> None:
    best = validation_baseline(3.356103)
    improved, best, misses = update_early_stopping(
        best,
        0,
        3.390597,
        min_delta=0.0,
    )

    assert improved is False
    assert best == 3.356103
    assert misses == 1


def test_sft_parser_exposes_conservative_training_controls() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--save-best",
            "--early-stopping-patience",
            "3",
        ]
    )
    assert args.lr == pytest.approx(5e-6)
    assert args.lora_rank == 16
    assert args.lora_alpha == 16
    assert args.expert_lr_multiplier == pytest.approx(0.25)
    assert args.eval_at_start is True
    assert args.save_best is True
    assert args.early_stopping_patience == 3


@pytest.mark.parametrize("rank", ["0", "-1"])
def test_sft_parser_rejects_nonpositive_lora_rank(rank: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--checkpoint", "checkpoint", "--lora-rank", rank]
        )


def test_every_500m_shell_launcher_selects_adaptation_mode_explicitly() -> None:
    launchers = [
        path
        for path in Path("scripts").glob("*.sh")
        if "-m scripts.sft_500m_32k_tr" in path.read_text(encoding="utf-8")
    ]
    assert launchers
    for launcher in launchers:
        source = launcher.read_text(encoding="utf-8")
        assert "--lora-rank" in source or "--full-parameter" in source, launcher


def test_full_parameter_mode_unfreezes_every_parameter() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    model.requires_grad_(False)
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--source-stage",
            "refinement",
            "--full-parameter",
        ]
    )

    stats = configure_sft_parameters(args, model)

    assert stats["mode"] == "full-parameter"
    assert stats["trainable"] == stats["total"]
    assert stats["frozen"] == 0
    assert stats["token_io_frozen"] is False
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_sft_runtime_requires_checkpoint_stage_provenance() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    args = build_parser().parse_args(["--checkpoint", "checkpoint"])

    with pytest.raises(ValueError, match="refinement"):
        configure_sft_parameters(args, model)


def test_sft_parser_rejects_direct_pretraining_source() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["--checkpoint", "checkpoint", "--source-stage", "pretraining"]
        )


def test_sft_parser_supports_a_finite_epoch_budget() -> None:
    args = build_parser().parse_args(["--checkpoint", "checkpoint", "--epochs", "3"])
    assert args.epochs == 3


def test_sft_parser_supports_automatic_epoch_schedule() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "checkpoint",
            "--steps",
            "0",
            "--epochs",
            "3",
            "--pack-sequences",
            "--save-every-epoch",
            "--eval-every-epoch",
        ]
    )

    assert args.steps == 0
    assert args.epochs == 3
    assert args.pack_sequences is True
    assert args.save_every_epoch is True
    assert args.eval_every_epoch is True


def test_sft_parser_and_state_support_exact_resume() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "base/checkpoint.pt",
            "--resume",
            "sft/step_000100/checkpoint.pt",
            "--steps",
            "200",
        ]
    )
    saved_args = {
        name: getattr(args, name)
        for name in (
            "jsonl",
            "sft_bin",
            "curriculum_config",
            "curriculum_stage",
            "epochs",
            "batch_size",
            "seq_len",
            "lr",
            "weight_decay",
            "beta1",
            "beta2",
            "warmup_ratio",
            "bf16",
            "use_custom_kernels",
            "grad_ckpt",
            "loss_chunk_tokens",
            "sft_fp32_loss",
            "seed",
        )
    }
    validate_resume_state(
        args,
        {
            "step": 100,
            "optimizer": {},
            "scheduler": {},
            "world_size": 4,
            "args": saved_args,
        },
        world_size=4,
    )

    assert args.resume.endswith("checkpoint.pt")


def test_sft_resume_rejects_a_packing_configuration_change() -> None:
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "base/checkpoint.pt",
            "--resume",
            "sft/step_000100/checkpoint.pt",
            "--steps",
            "0",
            "--epochs",
            "3",
            "--pack-sequences",
            "--save-every-epoch",
            "--eval-every-epoch",
        ]
    )

    with pytest.raises(ValueError, match="pack_sequences"):
        validate_resume_state(
            args,
            {
                "step": 100,
                "optimizer": {},
                "scheduler": {},
                "world_size": 4,
                "args": {
                    "pack_sequences": False,
                    "save_every_epoch": True,
                    "eval_every_epoch": True,
                },
            },
            world_size=4,
        )
