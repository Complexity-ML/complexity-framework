import torch

from complexity.generative.audio import TRHashAudioConfig, TRHashTextToSpeech


def _tiny_config() -> TRHashAudioConfig:
    return TRHashAudioConfig(
        n_mels=8,
        frame_patch_size=4,
        max_audio_frames=32,
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


def test_config_derived_shapes_are_consistent():
    config = _tiny_config()
    assert config.audio_token_count == 8
    assert config.route_vocab_size == 8 * config.time_buckets
    assert config.latent_patch_features == config.n_mels * config.frame_patch_size


def test_time_routes_are_stable_bounded_and_change_with_time():
    model = TRHashTextToSpeech(_tiny_config())
    timesteps = torch.tensor([0.1, 0.9])
    first = model.build_audio_route_ids(timesteps)
    second = model.build_audio_route_ids(timesteps)

    assert torch.equal(first, second)
    assert first.shape == (2, 8)
    assert int(first.min()) >= 0
    assert int(first.max()) < model.config.route_vocab_size
    assert not torch.equal(first[0], first[1])


def test_mel_forward_and_flow_loss_have_the_expected_contract():
    torch.manual_seed(7)
    model = TRHashTextToSpeech(_tiny_config())
    mel = torch.randn(2, 8, 32)
    caption_ids = torch.randint(0, 101, (2, 12))
    caption_mask = torch.ones_like(caption_ids, dtype=torch.bool)
    timesteps = torch.tensor([0.25, 0.75])

    prediction = model(mel, timesteps, caption_ids, caption_mask)
    assert prediction.shape == mel.shape

    loss = model.flow_matching_loss(mel, caption_ids, caption_mask)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert model.blocks[0].mlp.expert_down.grad is not None


def test_euler_sampler_supports_classifier_free_guidance():
    model = TRHashTextToSpeech(_tiny_config()).eval()
    caption_ids = torch.randint(0, 101, (1, 8))
    caption_mask = torch.ones_like(caption_ids, dtype=torch.bool)
    output = model.sample(
        caption_ids,
        caption_mask,
        steps=2,
        guidance_scale=2.0,
        generator=torch.Generator().manual_seed(11),
    )
    assert output.shape == (1, 8, 32)
    assert torch.isfinite(output).all()
