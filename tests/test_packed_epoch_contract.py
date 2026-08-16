from __future__ import annotations

from pathlib import Path

import pytest
import torch

from complexity.dataset.pipeline import DataPipeline
from complexity.training.packing import (
    FRAMEWORK_PACKING_CONTRACTS,
    TOKEN_PACK_PIPELINES,
    PackingKind,
    resolve_packed_epoch_schedule,
    resolve_token_pack_schedule,
    validate_framework_packing_contracts,
    validate_token_pack_pipeline,
)
from complexity.training.runner import FineWebStreamingDataset


def test_packed_epoch_contract_preserves_source_exposure() -> None:
    schedule = resolve_packed_epoch_schedule(
        full_steps=1000,
        exposure_factors=(4.0, 4.0, 1.0),
    )

    assert schedule.steps == (250, 250, 1000)
    assert schedule.total_steps == 1500
    assert schedule.unpacked_total_steps == 3000
    schedule.assert_source_exposure()


def test_token_pack_schedule_preserves_a_true_1t_without_epochs() -> None:
    schedule = resolve_token_pack_schedule(
        target_tokens=1_000_000_000_000,
        tokens_per_step=1_048_576,
        token_packs=50,
    )

    assert len(schedule.boundaries) == 50
    assert schedule.boundaries[-1] == schedule.total_steps
    assert schedule.actual_tokens >= 1_000_000_000_000
    assert schedule.actual_tokens - 1_000_000_000_000 < schedule.tokens_per_step
    assert max(schedule.pack_step_counts) - min(schedule.pack_step_counts) <= 1


def test_token_packs_are_scoped_only_to_text_pretraining() -> None:
    assert TOKEN_PACK_PIPELINES == {"text-pretraining"}
    validate_token_pack_pipeline("text-pretraining")

    for pipeline in FRAMEWORK_PACKING_CONTRACTS:
        if pipeline == "text-pretraining":
            continue
        with pytest.raises(ValueError, match="restricted to text-pretraining"):
            validate_token_pack_pipeline(pipeline)


def test_token_pack_schedule_rejects_non_pretraining_pipeline() -> None:
    with pytest.raises(ValueError, match="restricted to text-pretraining"):
        resolve_token_pack_schedule(
            target_tokens=1_000_000,
            tokens_per_step=2048,
            token_packs=10,
            pipeline="supervised-finetuning",
        )


def test_packed_epoch_contract_uses_realized_fractional_exposure() -> None:
    schedule = resolve_packed_epoch_schedule(
        full_steps=925,
        exposure_factors=(3.727,),
    )

    assert schedule.steps == (249,)
    assert schedule.steps[0] * 3.727 >= 925


def test_packed_epoch_contract_can_be_explicitly_disabled() -> None:
    schedule = resolve_packed_epoch_schedule(
        full_steps=1000,
        exposure_factors=(4.0, 2.0),
        enabled=False,
    )

    assert schedule.steps == (1000, 1000)
    assert schedule.exposure_factors == (1.0, 1.0)


@pytest.mark.parametrize("factors", ((), (0.0,), (float("nan"),)))
def test_packed_epoch_contract_rejects_invalid_exposure(factors) -> None:
    with pytest.raises(ValueError):
        resolve_packed_epoch_schedule(full_steps=100, exposure_factors=factors)


def test_every_framework_training_family_has_an_explicit_packing_contract() -> None:
    assert set(FRAMEWORK_PACKING_CONTRACTS) == {
        "text-pretraining",
        "supervised-finetuning",
        "vision-supervised-finetuning",
        "detector-pretraining",
        "vision-pretraining",
        "audio-pretraining",
        "video-pretraining",
        "vision-language-training",
        "sensor-fusion-training",
    }
    validate_framework_packing_contracts()
    for contract in FRAMEWORK_PACKING_CONTRACTS.values():
        assert bool(contract.reason.strip())
        assert contract.required is (contract.kind is not PackingKind.FIXED_SHAPE)


class _Tokenizer:
    eos_token_id = 0

    @staticmethod
    def encode(text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [int(token) for token in text.split()]


def test_text_pretraining_stream_is_packed_across_document_boundaries() -> None:
    dataset = FineWebStreamingDataset.__new__(FineWebStreamingDataset)
    dataset.tokenizer = _Tokenizer()
    dataset.seq_len = 4
    dataset.dataset = ({"text": "1 2"}, {"text": "3 4 5"}, {"text": "6 7"})

    batches = list(dataset)

    assert dataset.packing_contract == "text-pretraining"
    assert len(batches) == 2
    assert torch.equal(batches[0]["input_ids"], torch.tensor([1, 2, 0, 3]))
    assert torch.equal(batches[0]["labels"], torch.tensor([2, 0, 3, 4]))
    assert torch.equal(batches[1]["input_ids"], torch.tensor([4, 5, 0, 6]))
    assert torch.equal(batches[1]["labels"], torch.tensor([5, 0, 6, 7]))


def test_offline_pretraining_packing_preserves_long_documents_and_all_tokens() -> None:
    pipeline = DataPipeline.__new__(DataPipeline)
    source = [[1, 2], [3, 4, 5, 6, 7, 8], [9]]

    packed = pipeline._pack(source, seq_len=4)

    assert packed == [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    assert [token for sequence in packed for token in sequence] == [
        token for sequence in source for token in sequence
    ]


def test_production_launchers_cannot_hardcode_packing_off() -> None:
    forbidden = (
        "MOSAIC_PACKED_EPOCH=0",
        "MOSAIC_PACKED_EPOCH:-0",
        "PACK_SEQUENCES=0",
        "PACK_SEQUENCES:-0",
        "--no-mosaic-packed-epoch",
        "--no-pack-sequences",
    )
    offenders: list[str] = []
    for path in Path("scripts").glob("*"):
        if path.suffix not in {".py", ".sh"}:
            continue
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append(f"{path}:{token}")
    assert not offenders, "packing disabled in production launcher: " + ", ".join(offenders)


def test_detector_pretraining_launchers_cannot_default_mosaic_to_zero() -> None:
    forbidden = ("${MOSAIC:-0}", "${MOSAIC:-0.0}", "--mosaic 0", '--mosaic "0"')
    offenders: list[str] = []
    for path in Path("scripts").glob("*train_detector*.sh"):
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                offenders.append(f"{path}:{token}")
    assert not offenders, "Mosaic disabled in detector pretraining launcher: " + ", ".join(
        offenders
    )
