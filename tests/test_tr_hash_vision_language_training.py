import io
import json
import sys
import tarfile

import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from complexity.generative.vision_language import (
    TRHashImageTextToText,
    TRHashVisionLanguageConfig,
    VisionLanguageTarDataset,
    collate_vision_language,
)
from complexity.generative.vision_language.data import DEFAULT_PROMPT
from complexity.generative.vision_language.training import (
    freeze_decoder_for_vision_only_sft,
    stage_is_sft,
    tokenize_prompt_response,
)


def _tiny_vocab_tokenizer() -> Tokenizer:
    vocab = {
        "<pad>": 0,
        "</s>": 1,
        "<unk>": 2,
        "describe": 3,
        "this": 4,
        "image": 5,
        "a": 6,
        "cat": 7,
        "dog": 8,
        "what": 9,
        "color": 10,
        "is": 11,
        "it": 12,
        "red": 13,
    }
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _tiny_config() -> TRHashVisionLanguageConfig:
    return TRHashVisionLanguageConfig(
        image_size=32,
        patch_size=16,
        vision_hidden_size=16,
        vision_layers=1,
        vision_heads=4,
        vision_num_experts=4,
        vision_top_k=2,
        vision_shared_width=16,
        vision_expert_width=4,
        num_visual_tokens=4,
        vocab_size=101,
        hidden_size=32,
        decoder_layers=1,
        attention_heads=4,
        key_value_heads=2,
        max_position_embeddings=64,
        num_experts=4,
        top_k=2,
        shared_width=32,
        routed_width=32,
    )


def _write_shard(path, samples) -> None:
    with tarfile.open(path, "w") as archive:
        for sample_id, prompt, response in samples:
            image = Image.new("RGB", (12, 10), color=(10, 20, 30))
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="WEBP")
            metadata = {"prompt": prompt} if prompt is not None else {}
            members = {
                f"{sample_id}.webp": image_bytes.getvalue(),
                f"{sample_id}.txt": response.encode(),
                f"{sample_id}.json": json.dumps(metadata).encode(),
            }
            for name, payload in members.items():
                info = tarfile.TarInfo(name)
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))


def test_dataset_defaults_to_the_documented_prompt_when_shard_has_none(tmp_path):
    shard = tmp_path / "train-00000.tar"
    _write_shard(shard, [("sample-1", None, "A calico cat on a windowsill.")])

    rows = list(VisionLanguageTarDataset([shard], image_size=16))
    batch = collate_vision_language(rows)

    assert rows[0]["prompt"] == DEFAULT_PROMPT
    assert rows[0]["response"] == "A calico cat on a windowsill."
    assert batch["pixel_values"].shape == (1, 3, 16, 16)
    assert batch["prompts"] == [DEFAULT_PROMPT]


def test_dataset_uses_the_shard_prompt_when_present(tmp_path):
    shard = tmp_path / "train-00000.tar"
    _write_shard(shard, [("sample-1", "What color is the car?", "It is red.")])

    rows = list(VisionLanguageTarDataset([shard], image_size=16))

    assert rows[0]["prompt"] == "What color is the car?"
    assert rows[0]["response"] == "It is red."


def test_tokenize_prompt_response_masks_prompt_and_pads_with_ignore_index():
    tokenizer = _tiny_vocab_tokenizer()
    input_ids, labels = tokenize_prompt_response(
        tokenizer,
        prompts=["describe this image"],
        responses=["a cat"],
        max_length=32,
        device=torch.device("cpu"),
    )

    # prompt(3) + response(2) + appended </s> = 6 tokens: [3,4,5, 6,7,1]
    assert input_ids.tolist() == [[3, 4, 5, 6, 7, 1]]
    assert labels.tolist() == [[-100, -100, -100, 6, 7, 1]]


def test_tokenize_prompt_response_pads_shorter_sequences_and_masks_padding():
    tokenizer = _tiny_vocab_tokenizer()
    input_ids, labels = tokenize_prompt_response(
        tokenizer,
        prompts=["describe this image", "what color is it"],
        responses=["a cat", "a dog"],
        max_length=32,
        device=torch.device("cpu"),
    )

    # row 0: prompt(3) + response(2) + </s> = 6 tokens, padded to width 7.
    # row 1: prompt(4) + response(2) + </s> = 7 tokens (the longer row).
    assert input_ids.shape == labels.shape == (2, 7)
    assert input_ids[0].tolist() == [3, 4, 5, 6, 7, 1, 0]
    assert labels[0].tolist() == [-100, -100, -100, 6, 7, 1, -100]
    assert input_ids[1].tolist() == [9, 10, 11, 12, 6, 8, 1]
    assert labels[1].tolist() == [-100, -100, -100, -100, 6, 8, 1]


def test_stage_is_sft_stays_in_pretrain_when_disabled():
    for step in (0, 500, 999):
        assert stage_is_sft(step, total_steps=1000, sft_steps=0) is False


def test_stage_is_sft_switches_over_for_the_final_window():
    """Regression guard: the VLM trainer's "our vision trick" -- the same
    noisy-alignment -> clean-conversational-SFT switch used for TR-HASH
    Vision v8 and the text-to-image trainer -- moves from --shards to
    --sft-shards for exactly the final --sft-steps steps, not before."""
    assert stage_is_sft(699, total_steps=1000, sft_steps=300) is False
    assert stage_is_sft(700, total_steps=1000, sft_steps=300) is True
    assert stage_is_sft(999, total_steps=1000, sft_steps=300) is True


def test_sft_steps_requires_sft_shards_and_vice_versa(monkeypatch, tmp_path):
    from complexity.generative.vision_language import training

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cf-vlm-train",
            "--shards",
            str(tmp_path / "*.tar"),
            "--output",
            str(tmp_path / "out"),
            "--max-steps",
            "10",
            "--sft-steps",
            "5",
        ],
    )
    with pytest.raises(ValueError, match="--sft-steps requires --sft-shards"):
        training.main()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cf-vlm-train",
            "--shards",
            str(tmp_path / "*.tar"),
            "--output",
            str(tmp_path / "out"),
            "--max-steps",
            "10",
            "--sft-shards",
            str(tmp_path / "sft" / "*.tar"),
        ],
    )
    with pytest.raises(ValueError, match="--sft-shards requires --sft-steps"):
        training.main()


def test_freeze_decoder_for_vision_only_sft_leaves_only_vision_trainable():
    """Regression guard: the framework bans full-parameter SFT of language
    weights everywhere except vision-shaped pipelines (complexity/training/
    finetuning.py). The VLM's stage-2 SFT corpus is image-grounded QA/dialogue
    -- a language-instruction shape -- so it must never leave the decoder
    trainable, LoRA or otherwise. Only the vision tower/resampler/projection
    may update."""
    config = _tiny_config()
    model = TRHashImageTextToText(config)
    assert all(p.requires_grad for p in model.decoder.parameters())

    stats = freeze_decoder_for_vision_only_sft(model)

    assert not any(p.requires_grad for p in model.decoder.parameters())
    assert all(p.requires_grad for p in model.vision_tower.parameters())
    assert all(p.requires_grad for p in model.resampler.parameters())
    assert all(p.requires_grad for p in model.visual_projection.parameters())
    assert stats["trainable"] < stats["total"]
    assert stats["trainable"] + stats["frozen"] == stats["total"]


def test_image_text_to_text_sft_is_exempt_from_the_full_parameter_ban():
    from complexity.training.finetuning import (
        IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING,
        validate_full_parameter_finetuning,
    )

    validate_full_parameter_finetuning(IMAGE_TEXT_TO_TEXT_SUPERVISED_FINETUNING)


def test_end_to_end_batch_from_shard_to_loss_backward(tmp_path):
    """The new basic VLM training script's core wiring: read a shard, collate,
    tokenize prompt/response with masking, forward through the real model,
    and backpropagate -- proving the pieces actually fit together."""
    shard = tmp_path / "train-00000.tar"
    _write_shard(
        shard,
        [
            ("sample-1", None, "a cat"),
            ("sample-2", "what color is it", "a dog"),
        ],
    )
    config = _tiny_config()
    rows = list(VisionLanguageTarDataset([shard], image_size=config.image_size))
    batch = collate_vision_language(rows)
    assert len(rows) == 2

    tokenizer = _tiny_vocab_tokenizer()
    input_ids, labels = tokenize_prompt_response(
        tokenizer, batch["prompts"], batch["responses"], max_length=32, device=torch.device("cpu")
    )

    model = TRHashImageTextToText(config)
    output = model(batch["pixel_values"], input_ids, labels=labels)
    assert torch.isfinite(output["loss"])
    output["loss"].backward()
    assert any(p.grad is not None for p in model.parameters())
