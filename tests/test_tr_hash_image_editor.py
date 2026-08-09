import io
import json
import tarfile

import torch
from PIL import Image

from complexity.generative.image import (
    AtlasImageEditTarDataset,
    TRHashImageConfig,
    TRHashImageEditConfig,
    TRHashImageEditor,
    TRHashTextToImage,
    collate_atlas_image_edits,
)
from complexity.generative.image.editing import sample_edit_condition_dropout


def _tiny_config(**overrides) -> TRHashImageEditConfig:
    values = dict(
        image_size=32,
        vae_downsample_factor=4,
        latent_channels=4,
        latent_patch_size=2,
        vocab_size=101,
        max_text_length=16,
        text_hidden_size=32,
        text_layers=1,
        text_heads=4,
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        num_experts=4,
        top_k=2,
        shared_width=96,
        expert_width=16,
        time_buckets=8,
        caption_dropout=0.0,
        source_dropout=0.0,
    )
    values.update(overrides)
    return TRHashImageEditConfig(**values)


def _base_config(config: TRHashImageEditConfig) -> TRHashImageConfig:
    return TRHashImageConfig(
        **{name: getattr(config, name) for name in TRHashImageConfig.__dataclass_fields__}
    )


def test_default_editor_remains_close_to_the_200m_target():
    config = TRHashImageEditConfig()
    assert 190_000_000 <= config.estimated_parameter_count() <= 220_000_000
    assert config.source_dropout == 0.05


def test_editor_forward_loss_and_source_gates_receive_gradients():
    torch.manual_seed(17)
    model = TRHashImageEditor(_tiny_config())
    source = torch.randn(2, 4, 8, 8)
    target = torch.randn(2, 4, 8, 8)
    instruction_ids = torch.randint(0, 101, (2, 10))
    instruction_mask = torch.ones_like(instruction_ids, dtype=torch.bool)
    timesteps = torch.tensor([0.2, 0.8])

    prediction = model(target, source, timesteps, instruction_ids, instruction_mask)
    assert prediction.shape == target.shape
    loss = model.flow_matching_loss(target, source, instruction_ids, instruction_mask)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.source_input_gate.grad is not None
    assert model.source_block_gates.grad is not None
    assert model.blocks[0].mlp.expert_down.grad is not None


def test_text_to_image_weights_initialize_all_shared_editor_tensors():
    config = _tiny_config()
    base = TRHashTextToImage(_base_config(config))
    editor = TRHashImageEditor(config)

    missing = editor.load_text_to_image_state_dict(base.state_dict())

    assert missing
    assert all(key.startswith("source_") for key in missing)
    assert torch.equal(editor.latent_in.weight, base.latent_in.weight)
    assert torch.count_nonzero(editor.source_input_gate) == 0


def test_editor_sampling_supports_separate_image_and_text_guidance():
    model = TRHashImageEditor(_tiny_config()).eval()
    source = torch.randn(1, 4, 8, 8)
    instruction_ids = torch.randint(0, 101, (1, 8))
    instruction_mask = torch.ones_like(instruction_ids, dtype=torch.bool)
    output = model.edit(
        source,
        instruction_ids,
        instruction_mask,
        steps=2,
        image_guidance_scale=1.25,
        text_guidance_scale=2.5,
        generator=torch.Generator().manual_seed(5),
    )
    assert output.shape == source.shape
    assert torch.isfinite(output).all()


def test_source_dropout_always_selects_the_joint_unconditional_branch():
    dropped_text, dropped_source = sample_edit_condition_dropout(
        10_000,
        caption_dropout=0.1,
        source_dropout=0.2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(31),
    )
    assert dropped_source.any()
    assert (~dropped_source & dropped_text).any()
    assert not (dropped_source & ~dropped_text).any()


def test_edit_dataset_reads_explicit_source_instruction_target_triplets(tmp_path):
    shard = tmp_path / "train-00000.tar"
    source = Image.new("RGB", (12, 10), color=(10, 20, 30))
    target = Image.new("RGB", (12, 10), color=(50, 60, 70))

    def encoded(image):
        payload = io.BytesIO()
        image.save(payload, format="WEBP")
        return payload.getvalue()

    members = {
        "pair-1.source.webp": encoded(source),
        "pair-1.target.webp": encoded(target),
        "pair-1.txt": b"Make the scene warmer while preserving its geometry.",
        "pair-1.json": json.dumps({"license": "CC0", "edit": "warm"}).encode(),
    }
    with tarfile.open(shard, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    rows = list(AtlasImageEditTarDataset([shard], image_size=16))
    batch = collate_atlas_image_edits(rows)
    assert len(rows) == 1
    assert rows[0]["instruction"].startswith("Make the scene warmer")
    assert rows[0]["metadata"]["edit"] == "warm"
    assert batch["source_pixel_values"].shape == (1, 3, 16, 16)
    assert batch["target_pixel_values"].shape == (1, 3, 16, 16)
    assert not torch.equal(
        batch["source_pixel_values"],
        batch["target_pixel_values"],
    )
