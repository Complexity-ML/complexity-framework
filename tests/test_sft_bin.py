from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file as save_safetensors

from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    render_inference_prompt,
)
from scripts.sft_500m_32k_tr import (
    SFTBinDataset,
    SFTJsonlDataset,
    build_parser,
    configure_trainable_parameters,
    early_stopping_is_eligible,
    format_record,
    load_checkpoint_state,
    load_model_state_compat,
    pad_epoch_items,
    resolve_sft_bin_evaluation_partitions,
    update_early_stopping,
    validate_resume_state,
    validation_baseline,
)


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


def test_sft_can_freeze_token_input_and_output_parameters() -> None:
    class ToyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = torch.nn.Embedding(16, 8)
            self.layers = torch.nn.Linear(8, 8)
            self.lm_head = torch.nn.Linear(8, 16, bias=False)

    model = ToyModel()
    stats = configure_trainable_parameters(model, freeze_token_io=True)

    assert model.embed_tokens.weight.requires_grad is False
    assert model.lm_head.weight.requires_grad is False
    assert model.layers.weight.requires_grad is True
    assert stats["token_io_frozen"] is True
    assert stats["frozen"] == 16 * 8 * 2
    assert stats["trainable"] > 0


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
            "--freeze-token-io",
            "--save-best",
            "--early-stopping-patience",
            "3",
        ]
    )
    assert args.freeze_token_io is True
    assert args.eval_at_start is True
    assert args.save_best is True
    assert args.early_stopping_patience == 3


def test_sft_parser_supports_a_finite_epoch_budget() -> None:
    args = build_parser().parse_args(["--checkpoint", "checkpoint", "--epochs", "3"])
    assert args.epochs == 3


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
            "freeze_token_io",
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
