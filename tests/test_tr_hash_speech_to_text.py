import torch

from complexity.generative.audio import TRHashSpeechToText, TRHashSpeechToTextConfig


def _tiny_config() -> TRHashSpeechToTextConfig:
    return TRHashSpeechToTextConfig(
        sample_rate=16_000,
        n_mels=16,
        n_fft=128,
        hop_length=64,
        audio_hidden_size=32,
        audio_layers=1,
        audio_heads=4,
        num_audio_tokens=4,
        vocab_size=101,
        hidden_size=32,
        decoder_layers=2,
        attention_heads=4,
        key_value_heads=2,
        max_position_embeddings=64,
        num_experts=4,
        top_k=2,
        shared_width=64,
        routed_width=64,
    )


def test_audio_and_text_form_one_causal_sequence_and_predict_text_only():
    torch.manual_seed(17)
    config = _tiny_config()
    model = TRHashSpeechToText(config)
    waveform = torch.randn(2, 4_000)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()
    labels[:, :3] = -100

    output = model(waveform, input_ids, labels=labels)
    assert output["audio_tokens"].shape == (2, config.num_audio_tokens, config.hidden_size)
    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])

    output["loss"].backward()
    assert model.audio_projection[1].weight.grad is not None
    assert model.decoder.layers[0].mlp.engine.expert_down.grad is not None


def test_audio_encoder_actually_routes_through_multiple_experts():
    """The audio encoder must not be a plain dense pass-through — frames
    route through real TR-Hash MoE experts, same as the decoder."""
    config = _tiny_config()
    model = TRHashSpeechToText(config)
    assert config.audio_num_experts > 1
    for block in model.audio_encoder.blocks:
        assert torch.unique(block.mlp.route_table).numel() > 1

    waveform = torch.randn(2, 4_000)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(waveform, input_ids, labels=input_ids.clone())
    output["loss"].backward()
    assert model.audio_encoder.blocks[0].mlp.expert_gate.grad is not None
    assert model.audio_encoder.blocks[0].mlp.expert_gate.grad.abs().sum() > 0


def test_audio_routes_are_stable_and_text_routes_remain_token_ids():
    config = _tiny_config()
    model = TRHashSpeechToText(config).eval()
    waveform = torch.randn(1, 4_000)
    input_ids = torch.tensor([[3, 5, 8, 13]])

    _, first = model.prepare_multimodal_inputs(waveform, input_ids)
    _, second = model.prepare_multimodal_inputs(waveform, input_ids)
    assert torch.equal(first, second)
    assert torch.equal(first[:, config.num_audio_tokens :], input_ids)
    assert int(first.min()) >= 0
    assert int(first.max()) < config.vocab_size


def test_different_waveforms_produce_different_audio_prefixes():
    config = _tiny_config()
    model = TRHashSpeechToText(config).eval()
    input_ids = torch.tensor([[3, 5, 8, 13]])

    torch.manual_seed(1)
    embeddings_a, _ = model.prepare_multimodal_inputs(torch.randn(1, 4_000), input_ids)
    torch.manual_seed(2)
    embeddings_b, _ = model.prepare_multimodal_inputs(torch.randn(1, 4_000), input_ids)
    audio_tokens = config.num_audio_tokens
    assert not torch.allclose(
        embeddings_a[:, :audio_tokens], embeddings_b[:, :audio_tokens]
    )
