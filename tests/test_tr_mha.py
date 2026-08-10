import torch


def _small_config(
    attention_type: str = "tr_mha",
    *,
    tr_mha_targets: str = "qv",
):
    from complexity.config import ModelConfig

    return ModelConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        attention_type=attention_type,
        intermediate_size=48,
        mlp_type="tr_hash_engine",
        tr_mha_num_experts=4,
        tr_mha_adapter_rank=4,
        tr_mha_top_k=2,
        tr_mha_adapter_gate_init=0.1,
        tr_mha_id_primary_logit=2.0,
        tr_mha_id_secondary_logit=1.0,
        tr_mha_id_other_logit=-2.0,
        tr_mha_verifier_gate_init=0.1,
        tr_mha_verifier_temperature=1.0,
        tr_mha_targets=tr_mha_targets,
        max_position_embeddings=64,
    )


def test_tr_mha_uses_one_kv_head_per_query_head() -> None:
    from complexity.core.attention import TokenRoutedMultiHeadAttention
    from complexity.models import ComplexityModel

    model = ComplexityModel(_small_config())
    assert all(
        isinstance(layer.self_attn, TokenRoutedMultiHeadAttention)
        for layer in model.layers
    )
    assert all(
        layer.self_attn.num_heads == layer.self_attn.num_kv_heads == 4
        for layer in model.layers
    )
    assert all(layer.self_attn.num_kv_groups == 1 for layer in model.layers)
    assert all(layer.mlp.__class__.__name__ == "TRHashEngineMLP" for layer in model.layers)

    input_ids = torch.randint(0, 64, (2, 8))
    output = model(input_ids)
    assert output["logits"].shape == (2, 8, 64)


def test_id_prior_is_balanced_and_context_can_verify_it() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(7)
    attention = ComplexityModel(_small_config()).layers[0].self_attn
    token_ids = torch.arange(64).view(1, -1)
    hidden_a = torch.randn(1, 64, 32)
    hidden_b = hidden_a + 3.0 * torch.randn_like(hidden_a)

    probabilities_a, indices_a, weights_a, gate_a = (
        attention.routing_distribution(hidden_a, token_ids)
    )
    probabilities_b, _, _, _ = attention.routing_distribution(
        hidden_b, token_ids
    )

    assert probabilities_a.shape == (1, 64, 4)
    assert indices_a.shape == (1, 64, 2)
    assert weights_a.shape == (1, 64, 2)
    assert gate_a.shape == (1, 64, 1)
    assert torch.allclose(
        probabilities_a.sum(dim=-1), torch.ones(1, 64), atol=1e-6
    )
    primary_counts = torch.bincount(
        indices_a[..., 0].reshape(-1), minlength=4
    )
    assert primary_counts.tolist() == [16, 16, 16, 16]
    assert not torch.allclose(probabilities_a, probabilities_b)


def test_tr_mha_changes_q_and_v_but_not_shared_k() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(9)
    attention = ComplexityModel(_small_config()).layers[0].self_attn
    hidden = torch.randn(2, 7, 32)
    token_ids = torch.randint(0, 64, (2, 7))

    with torch.no_grad():
        original_gate = attention.adapter_output_gate.clone()
        attention.adapter_output_gate.zero_()
        k_base, q_base, v_base = attention._project_kqv(
            hidden, token_ids=token_ids
        )
        attention.adapter_output_gate.copy_(original_gate)
        k_routed, q_routed, v_routed = attention._project_kqv(
            hidden, token_ids=token_ids
        )

    assert torch.equal(k_base, k_routed)
    assert not torch.allclose(q_base, q_routed)
    assert not torch.allclose(v_base, v_routed)


def test_tr_mha_cache_matches_full_causal_attention() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(13)
    attention = ComplexityModel(_small_config()).layers[0].self_attn.eval()
    hidden = torch.randn(1, 8, 32)
    token_ids = torch.randint(0, 64, (1, 8))

    full, _ = attention(hidden, token_ids=token_ids)
    cached_outputs = []
    cache = None
    for index in range(hidden.shape[1]):
        output, cache = attention(
            hidden[:, index : index + 1],
            token_ids=token_ids[:, index : index + 1],
            past_key_value=cache,
            use_cache=True,
        )
        cached_outputs.append(output)
    cached = torch.cat(cached_outputs, dim=1)

    assert torch.allclose(full, cached, atol=2e-5, rtol=2e-4)


def test_tr_mha_router_and_adapters_receive_gradients() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(11)
    model = ComplexityModel(_small_config())
    input_ids = torch.randint(0, 64, (2, 12))
    logits = model(input_ids)["logits"]
    logits[:, :-1].float().square().mean().backward()

    for layer in model.layers:
        attention = layer.self_attn
        assert attention.context_router_weight.grad is not None
        assert attention.context_router_weight.grad.abs().sum() > 0
        assert attention.q_adapter_up.grad is not None
        assert attention.q_adapter_up.grad.abs().sum() > 0
        assert attention.v_adapter_up.grad is not None
        assert attention.v_adapter_up.grad.abs().sum() > 0


def test_tr_mha_v2_is_exactly_neutral_at_initialization() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(17)
    attention = ComplexityModel(
        _small_config("tr_mha_v2")
    ).layers[0].self_attn
    hidden = torch.randn(2, 7, 32)
    token_ids = torch.randint(0, 64, (2, 7))

    assert torch.count_nonzero(attention.qv_adapter_up) == 0
    with torch.no_grad():
        k_neutral, q_neutral, v_neutral = attention._project_kqv(
            hidden, token_ids=token_ids
        )
        attention.qv_adapter_up.normal_(mean=0.0, std=0.02)
        k_routed, q_routed, v_routed = attention._project_kqv(
            hidden, token_ids=token_ids
        )

    assert torch.equal(k_neutral, k_routed)
    assert not torch.allclose(q_neutral, q_routed)
    assert not torch.allclose(v_neutral, v_routed)


def test_tr_mha_v2_keeps_fixed_candidates_and_verifies_their_weights() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(19)
    attention = ComplexityModel(
        _small_config("tr_mha_v2")
    ).layers[0].self_attn
    token_ids = torch.arange(32).view(1, -1)
    hidden_a = torch.randn(1, 32, 32)
    hidden_b = hidden_a + torch.randn_like(hidden_a)
    with torch.no_grad():
        attention.context_router_weight.normal_(mean=0.0, std=0.5)
        attention.verifier_gate_bias.fill_(4.0)

    _, indices_a, weights_a, _ = attention.routing_distribution(
        hidden_a, token_ids
    )
    _, indices_b, weights_b, _ = attention.routing_distribution(
        hidden_b, token_ids
    )

    assert torch.equal(indices_a, indices_b)
    assert torch.all(indices_a[..., 0] != indices_a[..., 1])
    assert not torch.allclose(weights_a, weights_b)
    assert torch.allclose(
        weights_a.sum(dim=-1), torch.ones(1, 32), atol=1e-6
    )


def test_tr_mha_v2_neutral_branch_can_start_learning() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(23)
    model = ComplexityModel(_small_config("tr_mha_v2"))
    input_ids = torch.randint(0, 64, (2, 12))
    logits = model(input_ids)["logits"]
    logits[:, :-1].float().square().mean().backward()

    for layer in model.layers:
        attention = layer.self_attn
        assert attention.qv_adapter_up.grad is not None
        assert attention.qv_adapter_up.grad.abs().sum() > 0


def test_tr_mha_v2_v_only_preserves_q_and_k() -> None:
    from complexity.models import ComplexityModel

    torch.manual_seed(29)
    attention = ComplexityModel(
        _small_config("tr_mha_v2", tr_mha_targets="v")
    ).layers[0].self_attn
    hidden = torch.randn(2, 7, 32)
    token_ids = torch.randint(0, 64, (2, 7))

    with torch.no_grad():
        k_neutral, q_neutral, v_neutral = attention._project_kqv(
            hidden, token_ids=token_ids
        )
        attention.qv_adapter_up.normal_(mean=0.0, std=0.02)
        k_routed, q_routed, v_routed = attention._project_kqv(
            hidden, token_ids=token_ids
        )

    assert attention.route_targets == "v"
    assert attention.qv_adapter_up.shape[-1] == attention.hidden_size
    assert torch.equal(k_neutral, k_routed)
    assert torch.equal(q_neutral, q_routed)
    assert not torch.allclose(v_neutral, v_routed)

