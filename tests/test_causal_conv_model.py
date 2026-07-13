import torch


def test_causal_conv_mixer_is_strictly_causal_and_has_no_qkv():
    from complexity.core.attention import AttentionConfig, CausalConvMixer

    mixer = CausalConvMixer(
        AttentionConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            causal_conv_kernel_size=3,
            causal_conv_dilation=2,
        )
    )
    prefix = torch.randn(2, 5, 16)
    future_a = torch.randn(2, 3, 16)
    future_b = torch.randn(2, 3, 16)

    output_a, cache_a = mixer(torch.cat([prefix, future_a], dim=1))
    output_b, cache_b = mixer(torch.cat([prefix, future_b], dim=1))

    assert output_a.shape == (2, 8, 16)
    assert torch.allclose(output_a[:, :5], output_b[:, :5], atol=1e-6)
    assert cache_a is None
    assert cache_b is None
    assert not hasattr(mixer, "q_proj")
    assert not hasattr(mixer, "k_proj")
    assert not hasattr(mixer, "v_proj")


def test_model_builds_causal_conv_layers_with_dilation_cycle():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    model = ComplexityModel(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=48,
            vocab_size=64,
            attention_type="causal_conv",
            mlp_type="lexical_object_micro_expert",
            lexical_object_rank=4,
            tie_lexical_object_embeddings=True,
            micro_num_experts=4,
            micro_expert_width=2,
            causal_conv_kernel_size=3,
            causal_conv_dilation_cycle=3,
        )
    )

    assert [layer.self_attn.dilation for layer in model.layers] == [1, 2, 4, 1]
    output = model(torch.randint(0, 64, (2, 8)))
    assert output["logits"].shape == (2, 8, 64)

    forbidden_parameter_fragments = (
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv",
        "query",
        "key_proj",
        "value_proj",
    )
    parameter_names = [name.lower() for name, _ in model.named_parameters()]
    assert not any(
        fragment in name
        for name in parameter_names
        for fragment in forbidden_parameter_fragments
    )
    assert not any(isinstance(module, torch.nn.MultiheadAttention) for module in model.modules())


def test_causal_conv_cached_decode_matches_full_sequence_logits():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    torch.manual_seed(7)
    model = ComplexityModel(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=48,
            vocab_size=64,
            attention_type="causal_conv",
            mlp_type="lexical_object_micro_expert",
            lexical_object_rank=4,
            tie_lexical_object_embeddings=True,
            micro_num_experts=4,
            micro_expert_width=2,
            causal_conv_kernel_size=3,
            causal_conv_dilation_cycle=3,
        )
    ).eval()
    input_ids = torch.randint(0, 64, (2, 9))

    with torch.no_grad():
        full_logits = model(input_ids)["logits"]
        cache = None
        cache_pointers = None
        step_logits = []
        for index in range(input_ids.shape[1]):
            output = model(
                input_ids[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output["past_key_values"]
            current_pointers = [state.data_ptr() for state in cache]
            if cache_pointers is None:
                cache_pointers = current_pointers
            else:
                assert current_pointers == cache_pointers
            step_logits.append(output["logits"])

    incremental_logits = torch.cat(step_logits, dim=1)
    assert torch.allclose(full_logits, incremental_logits, atol=1e-5, rtol=1e-5)
    assert [state.shape[1] for state in cache] == [2, 4, 8, 2]


def test_causal_state_conv_has_persistent_fixed_state_and_exact_decode():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    torch.manual_seed(11)
    model = ComplexityModel(
        ModelConfig(
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=40,
            vocab_size=64,
            attention_type="causal_state_conv",
            mlp_type="lexical_object_micro_expert",
            lexical_object_rank=4,
            tie_lexical_object_embeddings=True,
            micro_num_experts=4,
            micro_expert_width=2,
            causal_conv_kernel_size=3,
            causal_conv_dilation_cycle=3,
            causal_state_rank=4,
        )
    ).eval()
    input_ids = torch.randint(0, 64, (2, 9))

    with torch.no_grad():
        full = model(input_ids)["logits"]
        cache = None
        pieces = []
        pointers = None
        for index in range(input_ids.shape[1]):
            output = model(
                input_ids[:, index : index + 1],
                past_key_values=cache,
                use_cache=True,
            )
            cache = output["past_key_values"]
            current = [(conv.data_ptr(), state.data_ptr()) for conv, state in cache]
            pointers = current if pointers is None else pointers
            assert current == pointers
            pieces.append(output["logits"])

    assert torch.allclose(full, torch.cat(pieces, dim=1), atol=1e-5, rtol=1e-5)
    assert all(state.shape == (2, 32) for _, state in cache)
    assert not any(
        part in name
        for name, _ in model.named_parameters()
        for part in ("q_proj", "k_proj", "v_proj")
    )

    model.train()
    model(input_ids)["logits"].square().mean().backward()
    assert model.layers[0].self_attn.state_decay_down.weight.grad is not None
    assert model.layers[0].self_attn.state_decay_up.weight.grad is not None


def test_lexical_attention_layers_preserve_shared_wvr_context_positions() -> None:
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel
    from complexity.core.attention.lexical_wrv import LexicalWRVAttention
    from complexity.core.attention.causal_fast_weight_conv import CausalFastWeightConvMixer

    config = ModelConfig(
        vocab_size=64, hidden_size=32, num_hidden_layers=10,
        num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
        attention_type="causal_fast_weight_conv",
        mlp_type="lexical_object_micro_expert",
        lexical_object_rank=16,
        tie_lexical_object_embeddings=True,
        causal_stable_delta=True, causal_state_rank=8,
        lexical_attention_layer_indices=(4, 9), max_position_embeddings=64,
    )
    model = ComplexityModel(config)
    assert isinstance(model.layers[4].self_attn, LexicalWRVAttention)
    assert isinstance(model.layers[9].self_attn, LexicalWRVAttention)
    assert model.layers[4].self_attn.lexical_token_scale is model.layers[0].mlp.token_scale
    assert model.layers[9].self_attn.lexical_token_scale is model.layers[0].mlp.token_scale
    conv_layers = [model.layers[i].self_attn for i in range(10) if i not in {4, 9}]
    assert all(isinstance(layer, CausalFastWeightConvMixer) for layer in conv_layers)
    assert model.layers[0].self_attn.context_enabled
    assert model.layers[5].self_attn.context_enabled
    assert model.layers[0].self_attn.shared_context is model.layers[5].self_attn.shared_context
    assert all(
        not model.layers[i].self_attn.context_enabled
        for i in (1, 2, 3, 6, 7, 8)
    )
