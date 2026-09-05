from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from complexity.tokenizer import Tokenizer
from scripts.prepare_tr_hash_gsm8k_sft import (
    clean_answer,
    encode_record,
    validate_tokenizer,
    write_artifact,
)

AGENTIC_TOKENIZER = Path(
    os.environ.get(
        "TR_HASH_AGENTIC_TOKENIZER",
        "/Users/boris/.cache/huggingface/hub/"
        "models--AETHORIA-AI--TR-HASH-Tokenizer-32K-Agentic/snapshots/"
        "2fcbc2c5359ded0244ca14531f1b3806eebac55e",
    )
)


def test_clean_answer_removes_gsm8k_verifier_markup() -> None:
    answer = "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 24"
    assert clean_answer(answer) == "Natalia sold 48/2 = 24 clips.\nFinal answer: 24"


@pytest.mark.skipif(not AGENTIC_TOKENIZER.exists(), reason="pinned Agentic tokenizer unavailable")
def test_native_agentic_projection_masks_prompt_and_supervises_answer() -> None:
    tokenizer = Tokenizer.load(str(AGENTIC_TOKENIZER))
    validate_tokenizer(tokenizer)
    prompt, completion, inputs, labels, supervised = encode_record(
        tokenizer,
        "What is 2 + 3?",
        "Add the numbers: 2 + 3 = <<2+3=5>>5.\n#### 5",
        sequence_length=2048,
    )
    assert prompt.startswith("<|user|>")
    assert prompt.endswith("<|end_of_turn|><|assistant|>")
    assert completion.endswith("Final answer: 5")
    assert int(inputs.max()) < 32_000
    assert supervised == int(np.count_nonzero(labels != -100))
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    assert np.all(labels[: len(prompt_ids) - 1] == -100)
    assert int(labels[-1]) == tokenizer.eos_token_id


@pytest.mark.skipif(not AGENTIC_TOKENIZER.exists(), reason="pinned Agentic tokenizer unavailable")
def test_fixture_split_contract_never_serializes_test(tmp_path: Path) -> None:
    train = pa.Table.from_pylist([{"question": "Train question?", "answer": "Work.\n#### 1"}])
    test = pa.Table.from_pylist([{"question": "Secret test?", "answer": "Work.\n#### 2"}])
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    pq.write_table(train, train_path)
    pq.write_table(test, test_path)
    output = tmp_path / "artifact"
    info = write_artifact(
        train_parquet=train_path,
        test_parquet=test_path,
        tokenizer_path=AGENTIC_TOKENIZER,
        output=output,
        sequence_length=2048,
        expected_train=1,
        expected_test=1,
    )
    serialized = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
    )
    assert "Train question?" in serialized
    assert "Secret test?" not in serialized
    assert not (output / "test").exists()
    assert info["splits"]["test"]["payload_in_artifact"] is False
    contract = json.loads(
        Path("configs/tr_hash_agentic_100m_gsm8k_sft.json").read_text(encoding="utf-8")
    )
    assert contract["dataset"]["training_split"] == "train"
    assert contract["dataset"]["evaluation_split"] == "test"
    assert contract["dataset"]["test_is_never_tokenized_for_training"] is True
    assert contract["publication"]["enabled"] is False


def test_launchers_keep_training_and_test_paths_separate() -> None:
    training = Path("scripts/run_tr_hash_agentic_100m_gsm8k_sft.sh").read_text()
    evaluation = Path("scripts/eval_tr_hash_gsm8k_supervised_epochs.sh").read_text()
    assert "TEST_PARQUET" not in training
    assert "--sft-bin" in training
    assert "--epochs 3" in training
    assert "--save-every-epoch" in training
    assert "--experiment-label supervised_gsm8k_sft" in evaluation
    assert "--split test" in evaluation


def test_asset_materializer_pins_commit_and_subfolder() -> None:
    source = Path("scripts/materialize_tr_hash_gsm8k_assets.sh").read_text()
    assert 'MODEL_REVISION="99fee390916154ec9d8c0f049c3a890f81414e20"' in source
    assert 'MODEL_SUBFOLDER="token_pack_014_213622"' in source
    assert "optimizer_rank" not in source
    assert "GSM8K_TRAIN_PARQUET" in source
    assert "GSM8K_TEST_PARQUET" in source
    assert 'dataset/chat_template.json" "$ASSET_ROOT/tokenizer/chat_template.json' in source
