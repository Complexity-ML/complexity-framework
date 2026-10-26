"""Sequence-aware diagnostic contracts for the canonical composed routing law."""

import json

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from complexity.config import ModelConfig
from complexity.core.mlp.base import MLPConfig
from complexity.core.mlp.tr_hash_engine import TRHashEngineMLP
from complexity.models import ComplexityModel
from complexity.profiling.routing import (
    audit_full_probe_routing,
    collect_routing_probe,
    load_tr_hash_model,
    parse_layers,
    routed_contributions,
    stratified_token_sample,
)


def _small_mlp(layer_index=1, intermediate_size=8) -> TRHashEngineMLP:
    torch.manual_seed(7)
    return TRHashEngineMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=intermediate_size,
            num_experts=4,
            vocab_size=64,
            top_k=2,
            layer_idx=layer_index,
            use_custom_kernels=False,
        )
    )


def _inputs():
    hidden = torch.randn(2, 16, 8, generator=torch.Generator().manual_seed(11))
    return hidden, torch.arange(32).reshape(2, 16)


def _oracle(token_ids, layer_index, previous_token_ids=None):
    result = np.empty((2, *token_ids.shape), dtype=np.int64)
    for row, tokens in enumerate(token_ids.tolist()):
        p2, p1 = [0, 0] if previous_token_ids is None else previous_token_ids[row].tolist()
        for position, token in enumerate(tokens):
            context = (p1 + 5 * p2) % 9
            if layer_index == 1:
                primary = ((token + 1) % 4) ^ (2 * ((context % 3) % 2))
                stride = 1 + (token // 4 + 2 * (context // 3) + 1) % 3
            else:
                primary = (token + layer_index) % 4
                stride = 1 + (token // 4 + (1 + layer_index % 2) * (context % 3) + layer_index) % 3
            result[:, row, position] = primary, (primary + stride) % 4
            p2, p1 = p1, token
    return result.reshape(2, -1)


def test_parse_layers_preserves_order_and_removes_duplicates():
    assert parse_layers("0, 5,11,5,23") == (0, 5, 11, 23)


@pytest.mark.parametrize("layer_index", (0, 1, 2, 3))
def test_probe_preserves_sequence_boundaries_and_matches_two_lag_oracle(layer_index):
    hidden, token_ids = _inputs()
    mlp = _small_mlp(layer_index)
    probe = collect_routing_probe(mlp, hidden, token_ids, "cpu", batch_size=7)
    np.testing.assert_array_equal(probe.routes, _oracle(token_ids, layer_index))
    np.testing.assert_array_equal(probe.token_ids, token_ids)
    assert probe.reconstruction_max_abs_error < 1e-9
    np.testing.assert_allclose(
        probe.shared + probe.contributions.sum(axis=0), probe.actual_output, atol=1e-9, rtol=1e-5
    )
    flattened_routes = _oracle(token_ids.reshape(1, -1), layer_index)
    assert not np.array_equal(probe.routes[:, 16], flattened_routes[:, 16])


def test_stratified_sample_represents_each_observed_primary_expert():
    hidden, token_ids = _inputs()
    probe = collect_routing_probe(_small_mlp(), hidden, token_ids, "cpu")
    selected = stratified_token_sample(probe, count=16, rng=np.random.default_rng(42))
    primary = probe.routes[0, selected]
    assert len(selected) == len(np.unique(selected)) == 16
    assert sorted(np.unique(primary).tolist()) == [0, 1, 2, 3]


def test_sampled_contributions_keep_context_from_before_selection():
    hidden, token_ids = _inputs()
    mlp = _small_mlp()
    probe = collect_routing_probe(mlp, hidden, token_ids, "cpu")
    selected = np.array([3, 7, 18], dtype=np.int64)
    packed, norms = routed_contributions(probe, selected)
    vectors, experts, route_ranks, source_indices = packed
    expected = _oracle(token_ids, 1)[:, selected].reshape(-1)
    resampled_ids = token_ids.flatten()[selected].unsqueeze(0)
    assert not np.array_equal(expected, _oracle(resampled_ids, 1).reshape(-1))
    assert vectors.shape == (6, 8)
    np.testing.assert_array_equal(experts, expected)
    np.testing.assert_array_equal(route_ranks, np.repeat([0, 1], 3))
    np.testing.assert_array_equal(source_indices, np.tile(selected, 2))
    assert np.all(norms > 0)
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    reconstructed = (vectors * norms[:, None]).reshape(2, 3, 8).sum(axis=0)
    np.testing.assert_allclose(
        reconstructed + probe.shared[selected], probe.actual_output[selected], atol=1e-9, rtol=1e-5
    )


def test_full_probe_audit_uses_unsampled_routes_and_finite_norm_ratios():
    hidden, token_ids = _inputs()
    residual = torch.randn(hidden.shape, generator=torch.Generator().manual_seed(12))
    probe = collect_routing_probe(_small_mlp(), hidden, token_ids, "cpu", batch_size=7)
    before = audit_full_probe_routing(probe, residual)
    selected = stratified_token_sample(probe, 5, np.random.default_rng(3))
    routed_contributions(probe, selected)
    audit = audit_full_probe_routing(probe, residual)
    assert audit == before
    expected_routes = _oracle(token_ids, 1)
    expected_counts = [np.bincount(route, minlength=4).tolist() for route in expected_routes]
    assert audit["tokens"] == 32
    assert audit["sequence_shape"] == [2, 16]
    assert audit["unique_token_ids"] == 32
    assert audit["sampling"] == "all_collected_probe_tokens"
    assert audit["route_counts"] == expected_counts
    assert sum(audit["combined_route_counts"]) == 64
    np.testing.assert_allclose(np.asarray(audit["route_fractions"]).sum(axis=1), 1)
    assert sum(audit["combined_route_fractions"]) == pytest.approx(1)
    assert audit["tokens_with_repeated_expert_across_routes"] == 0
    assert audit["all_experts_observed_per_route"]
    for metric in (
        "routed_branch_norm",
        "residual_stream_norm",
        "mlp_input_norm",
        "routed_to_residual_norm_ratio",
        "routed_to_mlp_input_norm_ratio",
    ):
        assert all(np.isfinite(value) for value in audit[metric].values())
        assert audit[metric]["max"] >= audit[metric]["p99"] >= audit[metric]["p50"]


def test_projection_batch_size_never_resets_context_and_chunks_use_real_history():
    hidden, token_ids = _inputs()
    mlp = _small_mlp()
    full = collect_routing_probe(mlp, hidden, token_ids, "cpu", batch_size=1)
    batched = collect_routing_probe(mlp, hidden, token_ids, "cpu", batch_size=7)
    np.testing.assert_array_equal(batched.routes, full.routes)
    np.testing.assert_allclose(batched.contributions, full.contributions, atol=1e-10, rtol=1e-5)
    history = token_ids[:, 3:5]
    chunk = collect_routing_probe(
        mlp, hidden[:, 5:], token_ids[:, 5:], "cpu", previous_token_ids=history, batch_size=3
    )
    np.testing.assert_array_equal(
        chunk.routes, full.routes.reshape(2, 2, 16)[:, :, 5:].reshape(2, -1)
    )
    np.testing.assert_array_equal(chunk.routes, _oracle(token_ids[:, 5:], 1, history))
    np.testing.assert_allclose(
        chunk.actual_output,
        full.actual_output.reshape(2, 16, 8)[:, 5:].reshape(-1, 8),
        atol=1e-10,
        rtol=1e-5,
    )


def test_probe_rejects_wrong_decomposition_instead_of_returning_false_vectors(monkeypatch):
    hidden, token_ids = _inputs()
    mlp = _small_mlp()
    actual_forward = mlp.forward
    monkeypatch.setattr(
        mlp, "forward", lambda *args, **kwargs: actual_forward(*args, **kwargs) + 0.01
    )
    with pytest.raises(ValueError, match="do not reconstruct the real forward"):
        collect_routing_probe(mlp, hidden, token_ids, "cpu")


def test_zero_contributions_have_finite_zero_vectors_and_norms():
    hidden, token_ids = _inputs()
    mlp = _small_mlp()
    with torch.no_grad():
        mlp.engine.expert_down.zero_()
    probe = collect_routing_probe(mlp, hidden, token_ids, "cpu")
    packed, norms = routed_contributions(probe, np.arange(32))
    assert np.isfinite(packed[0]).all()
    assert np.count_nonzero(packed[0]) == np.count_nonzero(norms) == 0
    audit = audit_full_probe_routing(probe, torch.zeros_like(hidden))
    assert audit["routed_branch_norm"]["max"] == 0
    assert audit["routed_to_residual_norm_ratio"]["max"] == 0


def test_load_tr_hash_model_accepts_current_engine_bundle(tmp_path):
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        mlp_type="tr_hash_engine",
        routing_strategy="token_id_composed_modulo",
        num_experts=4,
        top_k=2,
        tie_word_embeddings=False,
        use_custom_kernels=False,
    )
    source = ComplexityModel(config).eval()
    (tmp_path / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    save_file(
        {key: value.detach().contiguous() for key, value in source.state_dict().items()},
        tmp_path / "model.safetensors",
    )
    loaded = load_tr_hash_model(tmp_path, torch.device("cpu"))
    assert len(loaded.layers) == 1
    assert isinstance(loaded.layers[0].mlp, TRHashEngineMLP)
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert not loaded.training
    for name, value in source.state_dict().items():
        assert torch.equal(loaded.state_dict()[name], value)
    token_ids = torch.arange(12).reshape(2, 6)
    with torch.no_grad():
        torch.testing.assert_close(
            loaded(token_ids)["logits"], source(token_ids)["logits"], atol=0, rtol=0
        )


def test_sampling_rejects_a_count_that_cannot_represent_every_expert():
    hidden, token_ids = _inputs()
    probe = collect_routing_probe(_small_mlp(), hidden, token_ids, "cpu")
    with pytest.raises(ValueError, match="every observed primary expert"):
        stratified_token_sample(probe, 2, np.random.default_rng(0))


def test_primary_weight_variation_is_rejected_and_forward_stays_unchanged():
    hidden, token_ids = _inputs()
    mlp = _small_mlp()
    before = mlp(hidden, token_ids).detach().clone()
    with pytest.raises(ValueError, match="fixed top_k_primary_weight of 0.5"):
        mlp.set_top_k_primary_weight(0.9)
    after = mlp(hidden, token_ids).detach()
    assert torch.equal(before, after)
    probe = collect_routing_probe(mlp, hidden, token_ids, "cpu")
    np.testing.assert_allclose(probe.actual_output, before.reshape(-1, 8), atol=0, rtol=0)


def test_diagnostic_rejects_ambiguous_flat_inputs_and_reduced_runtime_capacity():
    hidden, token_ids = _inputs()
    mlp = _small_mlp()
    with pytest.raises(ValueError, match="hidden \\[batch, sequence, hidden\\]"):
        collect_routing_probe(mlp, hidden.reshape(-1, 8), token_ids.flatten(), "cpu")
    mlp.set_active_expert_width(1)
    with pytest.raises(ValueError, match="full configured expert capacity"):
        collect_routing_probe(mlp, hidden, token_ids, "cpu")


@pytest.mark.parametrize("bad_input", ("nan", "float_ids", "bad_history", "bad_residual"))
def test_probe_rejects_invalid_data(bad_input):
    hidden, token_ids = _inputs()
    mlp = _small_mlp()
    if bad_input == "nan":
        hidden[0, 0, 0] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            collect_routing_probe(mlp, hidden, token_ids, "cpu")
    elif bad_input == "float_ids":
        with pytest.raises(ValueError, match="torch.long"):
            collect_routing_probe(mlp, hidden, token_ids.float(), "cpu")
    elif bad_input == "bad_history":
        with pytest.raises(ValueError, match="previous_token_ids"):
            collect_routing_probe(
                mlp,
                hidden,
                token_ids,
                "cpu",
                previous_token_ids=torch.zeros(2, 3, dtype=torch.long),
            )
    else:
        probe = collect_routing_probe(mlp, hidden, token_ids, "cpu")
        with pytest.raises(ValueError, match="original sequence shape"):
            audit_full_probe_routing(probe, hidden.flatten())
