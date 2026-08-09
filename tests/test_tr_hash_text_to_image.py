import io
import json
import tarfile

import torch
from PIL import Image

from complexity.generative.image import (
    AtlasImageTarDataset,
    TRHashImageConfig,
    TRHashTextToImage,
    collate_atlas_images,
)


def _tiny_config() -> TRHashImageConfig:
    return TRHashImageConfig(
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
    )


def test_default_model_is_a_real_roughly_200m_configuration():
    config = TRHashImageConfig()
    assert 190_000_000 <= config.estimated_parameter_count() <= 220_000_000
    assert config.image_token_count == 256
    assert config.route_vocab_size == 256 * config.time_buckets


def test_spatial_time_routes_are_stable_bounded_and_change_with_time():
    model = TRHashTextToImage(_tiny_config())
    timesteps = torch.tensor([0.1, 0.9])
    first = model.build_image_route_ids(timesteps)
    second = model.build_image_route_ids(timesteps)

    assert torch.equal(first, second)
    assert first.shape == (2, 16)
    assert int(first.min()) >= 0
    assert int(first.max()) < model.config.route_vocab_size
    assert not torch.equal(first[0], first[1])


def test_latent_forward_and_flow_loss_have_the_expected_contract():
    torch.manual_seed(7)
    model = TRHashTextToImage(_tiny_config())
    latents = torch.randn(2, 4, 8, 8)
    caption_ids = torch.randint(0, 101, (2, 12))
    caption_mask = torch.ones_like(caption_ids, dtype=torch.bool)
    timesteps = torch.tensor([0.25, 0.75])

    prediction = model(latents, timesteps, caption_ids, caption_mask)
    assert prediction.shape == latents.shape

    loss = model.flow_matching_loss(latents, caption_ids, caption_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].mlp.expert_down.grad is not None


def test_euler_sampler_supports_classifier_free_guidance():
    model = TRHashTextToImage(_tiny_config()).eval()
    caption_ids = torch.randint(0, 101, (1, 8))
    caption_mask = torch.ones_like(caption_ids, dtype=torch.bool)
    output = model.sample(
        caption_ids,
        caption_mask,
        steps=2,
        guidance_scale=2.0,
        generator=torch.Generator().manual_seed(11),
    )
    assert output.shape == (1, 4, 8, 8)
    assert torch.isfinite(output).all()


def test_atlas_tar_dataset_reads_image_caption_and_metadata(tmp_path):
    shard = tmp_path / "train-00000.tar"
    image = Image.new("RGB", (12, 10), color=(10, 20, 30))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="WEBP")
    members = {
        "sample-1.webp": image_bytes.getvalue(),
        "sample-1.txt": b"A small blue study.",
        "sample-1.json": json.dumps({"license": "CC0", "museum": "Example"}).encode(),
    }
    with tarfile.open(shard, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))

    rows = list(AtlasImageTarDataset([shard], image_size=16))
    batch = collate_atlas_images(rows)
    assert len(rows) == 1
    assert rows[0]["caption"] == "A small blue study."
    assert rows[0]["metadata"]["license"] == "CC0"
    assert batch["pixel_values"].shape == (1, 3, 16, 16)
    assert batch["captions"] == ["A small blue study."]
