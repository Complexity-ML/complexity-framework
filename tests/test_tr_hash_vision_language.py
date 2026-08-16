import torch

from complexity.generative.vision_language import (
    TRHashImageTextToText,
    TRHashVisionLanguageConfig,
)
from complexity.models import ComplexityModel


def _tiny_config() -> TRHashVisionLanguageConfig:
    return TRHashVisionLanguageConfig(
        image_size=32,
        patch_size=8,
        vision_hidden_size=32,
        vision_layers=1,
        vision_heads=4,
        num_visual_tokens=4,
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


def test_complexity_model_accepts_external_embeddings_with_explicit_routes():
    config = _tiny_config().decoder_config()
    model = ComplexityModel(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (2, 7))
    expected = model(input_ids)["logits"]
    actual = model(
        inputs_embeds=model.embed_tokens(input_ids),
        routing_ids=input_ids,
    )["logits"]
    torch.testing.assert_close(actual, expected)


def test_image_and_text_form_one_causal_sequence_and_predict_text_only():
    torch.manual_seed(17)
    config = _tiny_config()
    model = TRHashImageTextToText(config)
    pixels = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    labels = input_ids.clone()
    labels[:, :3] = -100

    output = model(pixels, input_ids, labels=labels)
    assert output["visual_tokens"].shape == (2, config.num_visual_tokens, config.hidden_size)
    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["loss"].ndim == 0
    assert torch.isfinite(output["loss"])

    output["loss"].backward()
    assert model.visual_projection[1].weight.grad is not None
    assert model.decoder.layers[0].mlp.engine.expert_down.grad is not None


def test_vision_tower_actually_routes_through_multiple_experts():
    """The vision tower must not be a plain dense (num_experts=1) pass-through
    — patches route through real TR-Hash MoE experts, same as the decoder."""
    config = _tiny_config()
    model = TRHashImageTextToText(config)
    assert config.vision_num_experts > 1
    for block in model.vision_tower.blocks:
        assert torch.unique(block.mlp.route_table).numel() > 1

    pixels = torch.randn(2, 3, 32, 32)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(pixels, input_ids, labels=input_ids.clone())
    output["loss"].backward()
    assert model.vision_tower.blocks[0].mlp.expert_gate.grad is not None
    assert model.vision_tower.blocks[0].mlp.expert_gate.grad.abs().sum() > 0


def test_visual_routes_are_stable_and_text_routes_remain_token_ids():
    config = _tiny_config()
    model = TRHashImageTextToText(config).eval()
    pixels = torch.randn(1, 3, 32, 32)
    input_ids = torch.tensor([[3, 5, 8, 13]])

    _, first = model.prepare_multimodal_inputs(pixels, input_ids)
    _, second = model.prepare_multimodal_inputs(pixels, input_ids)
    assert torch.equal(first, second)
    assert torch.equal(first[:, config.num_visual_tokens :], input_ids)
    assert int(first.min()) >= 0
    assert int(first.max()) < config.vocab_size
