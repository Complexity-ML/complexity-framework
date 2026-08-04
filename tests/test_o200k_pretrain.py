"""Regression tests for the o200k pretraining runner utilities."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch


def test_chunked_hidden_loss_matches_full_loss():
    from complexity.core.losses import causal_lm_loss, causal_lm_loss_from_hidden

    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.randint(0, 13, (2, 5))

    full_loss, _ = causal_lm_loss(torch.nn.functional.linear(hidden, weight), labels)
    chunked_loss, _ = causal_lm_loss_from_hidden(
        hidden,
        weight,
        labels,
        chunk_tokens=3,
        checkpoint_chunks=False,
    )

    assert torch.allclose(full_loss, chunked_loss, atol=1e-6)


def test_open_ended_streaming_cli_contract():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args(["--dataset", "fineweb", "--steps", "0"])

    assert args.steps == 0
    assert args.warmup_steps is None


def test_chunked_hidden_loss_can_skip_metric_sync():
    from complexity.core.losses import causal_lm_loss_from_hidden

    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.randint(0, 13, (2, 5))

    loss, metrics = causal_lm_loss_from_hidden(
        hidden,
        weight,
        labels,
        chunk_tokens=3,
        checkpoint_chunks=False,
        sync_metrics=False,
    )

    assert torch.isfinite(loss)
    assert math.isnan(metrics.ce)


def test_full_hidden_loss_can_skip_metric_sync():
    from complexity.core.losses import causal_lm_loss_from_hidden

    torch.manual_seed(0)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.randint(0, 13, (2, 5))

    loss, metrics = causal_lm_loss_from_hidden(
        hidden,
        weight,
        labels,
        chunk_tokens=0,
        sync_metrics=False,
    )

    assert torch.isfinite(loss)
    assert math.isnan(metrics.ce)


def test_reduce_average_tensor_defers_to_single_item_sync():
    from complexity.training.o200k.runtime import reduce_average_tensor

    value = torch.tensor(3.5)

    assert reduce_average_tensor(value, distributed=False) == pytest.approx(3.5)


def _paired_init_test_configs(*, top_k=2, routed_output_scale=1.0):
    from complexity.config import ModelConfig
    from complexity.training.o200k.paired_init import dense_reference_config

    routed = ModelConfig(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        attention_type="gqa",
        mlp_type="token_routed",
        intermediate_size=16,
        shared_intermediate_size=48,
        shared_expert=True,
        num_experts=4,
        top_k=top_k,
        top_k_primary_weight=1.0 / top_k,
        routing_strategy="modulo_balanced_secondary",
        routed_output_scale=routed_output_scale,
    )
    return routed, dense_reference_config(routed)


def test_same_seed_alone_does_not_pair_deeper_backbone_weights():
    from complexity.models import ComplexityModel

    routed_config, dense_config = _paired_init_test_configs()
    torch.manual_seed(42)
    dense = ComplexityModel(dense_config)
    torch.manual_seed(42)
    routed = ComplexityModel(routed_config)

    assert torch.equal(
        dense.layers[0].self_attn.q_proj.weight,
        routed.layers[0].self_attn.q_proj.weight,
    )
    assert not torch.equal(
        dense.layers[1].self_attn.q_proj.weight,
        routed.layers[1].self_attn.q_proj.weight,
    )


def test_paired_dense_initialization_copies_common_backbone_exactly():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.paired_init import (
        initialize_token_routed_from_dense_reference,
    )

    routed_config, dense_config = _paired_init_test_configs()
    torch.manual_seed(42)
    dense = ComplexityModel(dense_config)
    torch.manual_seed(42)
    routed = ComplexityModel(routed_config)

    stats = initialize_token_routed_from_dense_reference(routed, dense)

    assert stats["split_layers"] == 3
    for dense_layer, routed_layer in zip(
        dense.layers, routed.layers, strict=True
    ):
        assert torch.equal(
            dense_layer.self_attn.q_proj.weight,
            routed_layer.self_attn.q_proj.weight,
        )


def test_paired_top4_routed_model_is_functionally_dense_at_initialization():
    from complexity.models import ComplexityModel
    from complexity.training.o200k.paired_init import (
        initialize_token_routed_from_dense_reference,
    )

    routed_config, dense_config = _paired_init_test_configs(
        top_k=4,
        routed_output_scale=4.0,
    )
    torch.manual_seed(7)
    dense = ComplexityModel(dense_config).eval()
    torch.manual_seed(7)
    routed = ComplexityModel(routed_config).eval()
    initialize_token_routed_from_dense_reference(routed, dense)
    token_ids = torch.randint(0, routed_config.vocab_size, (2, 9))

    with torch.no_grad():
        dense_hidden = dense(token_ids, return_logits=False)[
            "last_hidden_state"
        ]
        routed_hidden = routed(token_ids, return_logits=False)[
            "last_hidden_state"
        ]

    assert torch.allclose(dense_hidden, routed_hidden, atol=2e-6, rtol=2e-5)


def test_topk_primary_weight_schedule_ramps_toward_final():
    from complexity.training.o200k.runtime import scheduled_topk_primary_weight

    assert scheduled_topk_primary_weight(0, 100, 0.5, 0.9, 0.5) == pytest.approx(0.5)
    assert scheduled_topk_primary_weight(50, 100, 0.5, 0.9, 0.5) == pytest.approx(0.9)
    assert scheduled_topk_primary_weight(100, 100, 0.5, 0.9, 0.5) == pytest.approx(0.9)

    mid = scheduled_topk_primary_weight(25, 100, 0.5, 0.9, 0.5)
    assert 0.5 < mid < 0.9


def test_apply_topk_primary_weight_updates_token_routed_layers():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP
    from complexity.training.o200k.runtime import apply_topk_primary_weight

    mlp = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=16,
            shared_expert=False,
            top_k=2,
            top_k_primary_weight=0.5,
        )
    )

    assert apply_topk_primary_weight(mlp, 0.85) == 1
    assert mlp._primary_weight == pytest.approx(0.85)


def test_liger_fused_ce_availability_is_exposed(monkeypatch):
    from complexity.core.losses import fused_ce

    monkeypatch.setattr(fused_ce, "_liger_available", lambda: True)

    assert fused_ce.has_liger_fused_linear_ce() is True


def test_profile_param_counts_are_stable():
    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import PROFILES, make_config

    common = {
        "vocab_size": 200019,
        "use_mu_guidance": False,
        "learn_shared_routed_gates": True,
        "shared_gate_init": 1.0,
        "routed_gate_init": 0.1,
        "top_k": 2,
        "top_k_primary_weight": 0.5,
        "mu_clamp": False,
        "mu_norm": False,
        "mu_alpha_init": 1.0,
        "mu_init_value": 0.0,
        "mu_context_min": -2.0,
        "mu_context_max": 2.0,
        "use_custom_kernels": "auto",
        "moe_telemetry": False,
    }

    expected = {
        "50m": (200019, 51.9),
        "100m": (200019, 99.7),
        "200m_o200k": (200019, 200.1),
        "300m": (200019, 300.8),
        "1b": (200019, 1030.8),
        "8b": (200019, 8201.5),
    }
    for name, profile in PROFILES.items():
        vocab_size, expected_millions = expected[name]
        args = SimpleNamespace(**{**common, "vocab_size": vocab_size}, **profile)
        with torch.device("meta"):
            model = ComplexityModel(make_config(args))
        assert model.num_parameters() / 1e6 == pytest.approx(expected_millions, abs=0.1)


def test_random_dataset_infers_vocab_from_tokenizer(monkeypatch):
    from complexity.training import o200k_pretrain

    class FakeTokenizer:
        vocab_size = 200019

    monkeypatch.setattr(o200k_pretrain.Tokenizer, "load", lambda path: FakeTokenizer())
    args = SimpleNamespace(vocab_size=None, dataset="random", tokenizer="./tokenizer-o200k")

    assert o200k_pretrain.infer_vocab_size(args) == 200019


def test_vocab_size_override_wins(monkeypatch):
    from complexity.training import o200k_pretrain

    def fail_if_called(path):
        raise AssertionError("Tokenizer.load should not be called when --vocab-size is set")

    monkeypatch.setattr(o200k_pretrain.Tokenizer, "load", fail_if_called)
    args = SimpleNamespace(vocab_size=32000, dataset="random", tokenizer="./tokenizer-o200k")

    assert o200k_pretrain.infer_vocab_size(args) == 32000


def test_o200k_parser_enables_grad_checkpointing_by_default():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args([])

    assert args.grad_ckpt is True


def test_o200k_parser_can_disable_grad_checkpointing():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args(["--no-grad-ckpt"])

    assert args.grad_ckpt is False


def test_o200k_parser_supports_paired_dense_initialization():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args(["--paired-dense-init"])

    assert args.paired_dense_init is True


def test_o200k_parser_disables_grad_clipping_by_default():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args([])

    assert args.max_grad_norm == 0.0


def test_o200k_parser_uses_auto_loss_backend_by_default():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args([])

    assert args.loss_backend == "auto"


def test_o200k_parser_supports_token_shards():
    from complexity.training.o200k_pretrain import build_parser

    args = build_parser().parse_args(["--dataset", "tokens", "--tokens-path", "data/tokens"])

    assert args.dataset == "tokens"
    assert args.tokens_path == "data/tokens"


def test_token_shard_dataset_and_frequencies(tmp_path):
    from complexity.data.token_shards import (
        TokenShardDataset,
        load_token_shard,
        token_shard_frequencies,
        write_token_shard,
    )

    write_token_shard(tmp_path, range(100), vocab_size=128, tokenizer="dummy")
    tokens, metadata = load_token_shard(tmp_path)

    assert tokens.shape == (100,)
    assert metadata["num_tokens"] == 100
    assert metadata["dtype"] == "<u2"
    assert len(metadata["sha256"]) == 64

    freqs = token_shard_frequencies(tmp_path, vocab_size=128)
    assert freqs.dtype == torch.int64
    assert freqs.sum().item() == 100
    assert freqs[42].item() == 1

    train_freqs = token_shard_frequencies(
        tmp_path,
        vocab_size=128,
        eval_ratio=0.2,
        seq_len=8,
    )
    assert train_freqs.sum().item() == 90
    assert train_freqs[89].item() == 1
    assert train_freqs[90].item() == 0

    ds = TokenShardDataset(tmp_path, seq_len=8, seed=0, eval_ratio=0.2)
    batch = next(iter(ds))
    assert batch["input_ids"].shape == (8,)
    assert batch["labels"].shape == (8,)


def test_token_frequency_accumulator_is_exact_above_float32_limit():
    from complexity.data.token_shards import _accumulate_frequency_chunk

    frequencies = torch.tensor([2**24, 0], dtype=torch.int64)
    _accumulate_frequency_chunk(frequencies, torch.tensor([0, 1, 1]))

    assert frequencies.tolist() == [2**24 + 1, 2]


def test_text_routing_frequencies_exclude_the_eval_tail(monkeypatch):
    from complexity.training.o200k import data

    tokens = list(range(10_000))
    monkeypatch.setattr(data, "load_text_tokens", lambda *args: tokens)

    freqs = data.text_token_frequencies(
        "ignored.txt",
        "ignored-tokenizer",
        vocab_size=10_000,
        eval_ratio=0.1,
    )

    # Local text reserves at least 2,048 tokens, capped at 20% here.
    assert freqs.dtype == torch.int64
    assert freqs.sum().item() == 8_000
    assert freqs[7_999].item() == 1
    assert freqs[8_000].item() == 0


def test_token_routed_topk_uses_precomputed_zipf_routes():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(0)
    cfg = MLPConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        vocab_size=64,
        shared_expert=False,
        top_k=2,
        top_k_primary_weight=0.5,
    )
    mlp = TokenRoutedMLP(cfg)
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 64, (2, 5))

    out_fast = mlp(hidden, token_ids=token_ids)

    flat_x = hidden.reshape(-1, hidden.size(-1))
    route_ids = mlp.topk_token_to_expert[:, token_ids.clamp(0, mlp.vocab_size - 1)]
    gate_w = mlp.gate_proj_w
    up_w = mlp.up_proj_w
    down_w = mlp.down_proj_w
    out_ref = 0.5 * mlp._dispatch_once(
        flat_x, route_ids[0].reshape(-1), gate_w, up_w, down_w, False, 16
    )
    out_ref = out_ref + 0.5 * mlp._dispatch_once(
        flat_x,
        route_ids[1].reshape(-1),
        gate_w,
        up_w,
        down_w,
        False,
        16,
    )

    assert torch.allclose(out_fast.reshape_as(out_ref), out_ref, atol=1e-6)


def test_equal_top2_pair_fusion_matches_legacy_outputs_and_gradients():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(7)
    cfg = MLPConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        vocab_size=64,
        shared_expert=False,
        routing_strategy="token_id_balanced_hash",
        top_k=2,
        top_k_primary_weight=0.5,
        use_cggr=False,
    )
    fused = TokenRoutedMLP(cfg)
    legacy = TokenRoutedMLP(cfg)
    combined = TokenRoutedMLP(cfg)
    legacy.load_state_dict(fused.state_dict())
    combined.load_state_dict(fused.state_dict())

    hidden_fused = torch.randn(3, 5, 16, requires_grad=True)
    hidden_legacy = hidden_fused.detach().clone().requires_grad_(True)
    hidden_combined = hidden_fused.detach().clone().requires_grad_(True)
    token_ids = torch.randint(0, 64, (3, 5))
    probe = torch.randn(15, 16)

    out_fused = fused(hidden_fused, token_ids=token_ids).reshape(15, 16)
    routes = legacy.topk_token_to_expert[:, token_ids].reshape(2, -1)
    flat_legacy = hidden_legacy.reshape(15, 16)
    out_legacy = 0.5 * legacy._dispatch_once(
        flat_legacy,
        routes[0],
        legacy.gate_proj_w,
        legacy.up_proj_w,
        legacy.down_proj_w,
        False,
        16,
    )
    out_legacy = out_legacy + 0.5 * legacy._dispatch_once(
        flat_legacy,
        routes[1],
        legacy.gate_proj_w,
        legacy.up_proj_w,
        legacy.down_proj_w,
        False,
        16,
    )
    combined_routes = combined.topk_token_to_expert[:, token_ids].reshape(
        2, -1
    )
    out_combined = combined._dispatch_equal_top2_pair_cggr(
        hidden_combined.reshape(15, 16),
        combined_routes,
        combined.gate_proj_w,
        combined.up_proj_w,
        combined.down_proj_w,
        16,
        use_cggr=False,
    )

    assert fused.last_dispatch_path == "top2_pair_fused"
    assert torch.allclose(out_fused, out_legacy, atol=1e-6, rtol=1e-5)
    assert torch.allclose(out_combined, out_legacy, atol=1e-6, rtol=1e-5)

    (out_fused * probe).sum().backward()
    (out_legacy * probe).sum().backward()
    (out_combined * probe).sum().backward()
    assert torch.allclose(
        hidden_fused.grad,
        hidden_legacy.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.allclose(
        hidden_combined.grad,
        hidden_legacy.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    legacy_params = dict(legacy.named_parameters())
    combined_params = dict(combined.named_parameters())
    for name, parameter in fused.named_parameters():
        assert parameter.grad is not None
        assert legacy_params[name].grad is not None
        assert combined_params[name].grad is not None
        assert torch.allclose(
            parameter.grad,
            legacy_params[name].grad,
            atol=1e-5,
            rtol=1e-5,
        ), name
        assert torch.allclose(
            combined_params[name].grad,
            legacy_params[name].grad,
            atol=1e-5,
            rtol=1e-5,
        ), name


def test_compact_pair_hash_planner_matches_precomputed_routes():
    from complexity.core.mlp.token_routed import (
        _encode_pair_coverage_hash_routes,
        _create_pair_coverage_hash_metadata,
        _create_pair_coverage_hash_routes,
    )
    from complexity_cuda.triton_token_routed import (
        pair_coverage_hash_expert_ids,
        sort_pair_hash_by_expert,
    )

    vocab_size = 4096
    token_ids = torch.tensor(
        [[-1, 0, 1, 17, 511], [2048, 4094, 4095, 4096, 9999]]
    )
    _, expert_pairs, _ = _create_pair_coverage_hash_metadata(vocab_size, 4)
    for layer_idx in range(10):
        routes = _create_pair_coverage_hash_routes(
            vocab_size,
            4,
            layer_idx,
        )
        route_codes = _encode_pair_coverage_hash_routes(
            routes,
            expert_pairs,
        )
        decoded = pair_coverage_hash_expert_ids(
            token_ids,
            route_codes,
            expert_pairs,
            vocab_size=vocab_size,
        )
        expected = routes[:, token_ids.clamp(0, vocab_size - 1)]
        assert torch.equal(decoded.long(), expected)
        sorted_indices, expert_offsets, expert_counts = (
            sort_pair_hash_by_expert(
                token_ids,
                route_codes,
                expert_pairs,
                vocab_size=vocab_size,
                num_experts=4,
            )
        )
        flat_experts = expected.reshape(-1)
        expected_experts, _ = torch.sort(flat_experts, stable=True)
        actual_experts = flat_experts[sorted_indices]
        assert torch.equal(actual_experts, expected_experts)
        assert int(expert_counts.sum()) == 2 * token_ids.numel()
        assert torch.equal(
            torch.diff(expert_offsets),
            expert_counts,
        )
        (
            inverse_sorted_indices,
            inverse_indices,
            inverse_offsets,
            inverse_counts,
        ) = sort_pair_hash_by_expert(
            token_ids,
            route_codes,
            expert_pairs,
            vocab_size=vocab_size,
            num_experts=4,
            return_inverse=True,
        )
        assert torch.equal(inverse_sorted_indices, sorted_indices)
        assert torch.equal(inverse_offsets, expert_offsets)
        assert torch.equal(inverse_counts, expert_counts)
        assert torch.equal(
            inverse_sorted_indices[inverse_indices],
            torch.arange(
                inverse_sorted_indices.numel(),
                dtype=inverse_sorted_indices.dtype,
            ),
        )

    compact_bytes = (
        route_codes.numel() * route_codes.element_size()
        + expert_pairs.numel() * expert_pairs.element_size()
    )
    full_routes_bytes = (
        _create_pair_coverage_hash_routes(vocab_size, 4, 0).numel() * 8
    )
    assert compact_bytes < full_routes_bytes


@pytest.mark.parametrize(
    "routing_strategy",
    [
        "token_id_balanced_hash",
        "token_id_pair_coverage_hash",
    ],
)
def test_hash_counting_dispatch_matches_pair_dispatch_and_gradients(
    routing_strategy,
):
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(23)
    cfg = MLPConfig(
        hidden_size=12,
        intermediate_size=32,
        num_experts=4,
        vocab_size=128,
        shared_expert=False,
        routing_strategy=routing_strategy,
        top_k=2,
        top_k_primary_weight=0.5,
        use_cggr=False,
    )
    reference = TokenRoutedMLP(cfg)
    hashed = TokenRoutedMLP(cfg)
    hashed.load_state_dict(reference.state_dict())
    token_ids = torch.randint(0, cfg.vocab_size, (21,))
    reference_x = torch.randn(21, 12, requires_grad=True)
    hashed_x = reference_x.detach().clone().requires_grad_(True)
    routes = reference.topk_token_to_expert[:, token_ids]

    reference_out = reference._dispatch_equal_top2_pair_cggr(
        reference_x,
        routes,
        reference.gate_proj_w,
        reference.up_proj_w,
        reference.down_proj_w,
        12,
        use_cggr=False,
    )
    hashed_out = hashed._dispatch_equal_top2_hash_cggr(
        hashed_x,
        token_ids,
        hashed.gate_proj_w,
        hashed.up_proj_w,
        hashed.down_proj_w,
        12,
        use_cggr=False,
    )
    assert torch.allclose(
        hashed_out,
        reference_out,
        atol=1e-6,
        rtol=1e-5,
    )

    probe = torch.randn_like(reference_out)
    (reference_out * probe).sum().backward()
    (hashed_out * probe).sum().backward()
    assert torch.allclose(
        hashed_x.grad,
        reference_x.grad,
        atol=1e-6,
        rtol=1e-5,
    )
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in hashed.named_parameters():
        assert parameter.grad is not None
        assert torch.allclose(
            parameter.grad,
            reference_parameters[name].grad,
            atol=1e-5,
            rtol=1e-5,
        ), name


def test_hash_pair_gates_preserve_equal_mix_then_learn_without_rerouting():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(29)
    common = dict(
        hidden_size=12,
        intermediate_size=32,
        num_experts=4,
        vocab_size=128,
        shared_expert=False,
        routing_strategy="token_id_balanced_hash",
        top_k=2,
        top_k_primary_weight=0.5,
        use_cggr=False,
    )
    reference = TokenRoutedMLP(MLPConfig(**common))
    gated = TokenRoutedMLP(
        MLPConfig(
            **common,
            learn_hash_pair_gates=True,
            hash_pair_gate_init=0.5,
        )
    )
    missing, unexpected = gated.load_state_dict(
        reference.state_dict(),
        strict=False,
    )
    assert missing == ["hash_pair_gate_logits"]
    assert unexpected == []
    assert gated.hash_pair_gate_logits.numel() == 6
    assert sum(p.numel() for p in gated.parameters()) == (
        sum(p.numel() for p in reference.parameters()) + 6
    )

    token_ids = torch.randint(0, 128, (3, 7))
    hidden = torch.randn(3, 7, 12)
    original_routes = gated.topk_token_to_expert[:, token_ids].clone()
    reference_out = reference(hidden, token_ids=token_ids)
    gated_out = gated(hidden, token_ids=token_ids)
    assert torch.allclose(gated_out, reference_out, atol=1e-6, rtol=1e-5)

    gated_out.square().mean().backward()
    assert gated.hash_pair_gate_logits.grad is not None
    assert gated.hash_pair_gate_logits.grad.abs().sum() > 0

    with torch.no_grad():
        gated.hash_pair_gate_logits[0] = 1.0
    changed_out = gated(hidden, token_ids=token_ids)
    assert not torch.allclose(changed_out, reference_out)
    assert torch.equal(
        gated.topk_token_to_expert[:, token_ids],
        original_routes,
    )

def test_hash_channel_modulation_is_neutral_then_learns_without_rerouting():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(31)
    common = dict(
        hidden_size=12,
        intermediate_size=32,
        num_experts=4,
        vocab_size=128,
        shared_expert=False,
        routing_strategy="token_id_balanced_hash",
        top_k=2,
        top_k_primary_weight=0.5,
        use_cggr=False,
    )
    reference = TokenRoutedMLP(MLPConfig(**common))
    hashed = TokenRoutedMLP(
        MLPConfig(
            **common,
            learn_hash_channel_modulation=True,
            hash_channel_scale_init=0.0,
            layer_idx=3,
        )
    )
    missing, unexpected = hashed.load_state_dict(
        reference.state_dict(),
        strict=False,
    )
    assert missing == ["hash_channel_scale"]
    assert unexpected == []
    assert hashed.hash_channel_scale.shape == (4, 8)

    token_ids = torch.randint(0, 128, (3, 7))
    hidden = torch.randn(3, 7, 12)
    original_routes = hashed.topk_token_to_expert[:, token_ids].clone()
    reference_out = reference(hidden, token_ids=token_ids)
    hashed_out = hashed(hidden, token_ids=token_ids)
    assert torch.allclose(hashed_out, reference_out, atol=1e-6, rtol=1e-5)

    hashed_out.square().mean().backward()
    assert hashed.hash_channel_scale.grad is not None
    assert hashed.hash_channel_scale.grad.abs().sum() > 0

    with torch.no_grad():
        hashed.hash_channel_scale[0, 0] = 0.25
    changed_out = hashed(hidden, token_ids=token_ids)
    assert not torch.allclose(changed_out, reference_out)
    assert torch.equal(
        hashed.topk_token_to_expert[:, token_ids],
        original_routes,
    )


def test_unequal_top2_weights_keep_the_generic_dispatch():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    mlp = TokenRoutedMLP(
        MLPConfig(
            hidden_size=8,
            intermediate_size=16,
            num_experts=4,
            vocab_size=32,
            shared_expert=False,
            routing_strategy="token_id_balanced_hash",
            top_k=2,
            top_k_primary_weight=0.6,
            use_cggr=False,
        )
    )
    mlp(
        torch.randn(2, 4, 8),
        token_ids=torch.randint(0, 32, (2, 4)),
    )

    assert mlp.last_dispatch_path == "masked_dense"


def test_token_routed_topk_aux_routes_are_balanced_and_distinct():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        vocab_size=16,
        shared_expert=False,
        top_k=2,
        token_frequencies=torch.ones(16),
    )
    mlp = TokenRoutedMLP(cfg)

    routes = mlp.topk_token_to_expert.cpu()
    assert torch.all(routes[0] != routes[1])
    for route_idx in range(2):
        counts = torch.bincount(routes[route_idx], minlength=4)
        assert counts.tolist() == [4, 4, 4, 4]


def test_token_routed_masked_dispatch_matches_token_reference():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP, sort_tokens_by_expert

    torch.manual_seed(0)
    cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=4,
        vocab_size=64,
        shared_expert=False,
        top_k=1,
    )
    mlp = TokenRoutedMLP(cfg)
    flat_x = torch.randn(11, 8)
    expert_ids = torch.tensor([0, 3, 1, 0, 2, 3, 1, 2, 0, 3, 2])
    sorted_x, sorted_idx, expert_offsets, expert_counts = sort_tokens_by_expert(
        flat_x, expert_ids, mlp.num_experts
    )

    out = mlp._dispatch_sorted(
        flat_x,
        sorted_x,
        sorted_idx,
        expert_offsets,
        expert_counts,
        mlp.gate_proj_w,
        mlp.up_proj_w,
        mlp.down_proj_w,
        use_cggr=False,
        H=8,
    )

    ref = torch.empty_like(flat_x)
    for i, expert in enumerate(expert_ids.tolist()):
        x = flat_x[i]
        gate = x @ mlp.gate_proj_w[expert]
        up = x @ mlp.up_proj_w[expert]
        ref[i] = (torch.nn.functional.silu(gate) * up) @ mlp.down_proj_w[expert]

    assert torch.allclose(out, ref, atol=1e-6)


def test_cggr_dispatch_auto_selects_triton_when_available(monkeypatch):
    import complexity.core.mlp.token_routed as token_routed

    monkeypatch.setattr(token_routed, "supports_custom_triton", lambda policy: True)

    use_cggr, reasons = token_routed.cggr_dispatch_decision(
        cggr_policy="auto",
        kernel_policy="auto",
        is_cuda=True,
        has_cggr=True,
        has_autograd=True,
    )

    assert use_cggr is True
    assert reasons == []


def test_cggr_dispatch_falls_back_when_auto_is_not_supported(monkeypatch):
    import complexity.core.mlp.token_routed as token_routed

    monkeypatch.setattr(token_routed, "supports_custom_triton", lambda policy: False)

    use_cggr, reasons = token_routed.cggr_dispatch_decision(
        cggr_policy="auto",
        kernel_policy="auto",
        is_cuda=True,
        has_cggr=True,
        has_autograd=True,
    )

    assert use_cggr is False
    assert "supports_custom_triton(policy='auto')=False" in reasons


def test_o200k_config_defaults_cggr_to_auto():
    from complexity.training.o200k_pretrain import make_config

    args = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        shared_intermediate_size=64,
        vocab_size=128,
        use_mu_guidance=False,
        learn_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.1,
        top_k=2,
        top_k_primary_weight=0.5,
        shared_expert_chunk_tokens=0,
        static_expert_capacity=False,
        routing_strategy="zipf",
        mu_clamp=False,
        mu_norm=False,
        mu_alpha_init=1.0,
        mu_init_value=0.0,
        mu_context_min=-2.0,
        mu_context_max=2.0,
        use_custom_kernels="auto",
        moe_telemetry=False,
    )

    assert make_config(args).use_cggr == "auto"


def test_shared_expert_chunking_matches_dense_path():
    from complexity.core.mlp.base import MLPConfig
    from complexity.core.mlp.token_routed import TokenRoutedMLP

    torch.manual_seed(0)
    cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=2,
        vocab_size=32,
        shared_expert=True,
        shared_intermediate_size=24,
        shared_expert_chunk_tokens=0,
    )
    dense = TokenRoutedMLP(cfg)

    chunked_cfg = MLPConfig(
        hidden_size=8,
        intermediate_size=16,
        num_experts=2,
        vocab_size=32,
        shared_expert=True,
        shared_intermediate_size=24,
        shared_expert_chunk_tokens=5,
    )
    chunked = TokenRoutedMLP(chunked_cfg)
    chunked.load_state_dict(dense.state_dict(), strict=False)

    hidden = torch.randn(3, 4, 8, requires_grad=True)
    token_ids = torch.randint(0, 32, (3, 4))

    out_dense = dense(hidden, token_ids=token_ids)
    out_chunked = chunked(hidden, token_ids=token_ids)

    assert torch.allclose(out_chunked, out_dense, atol=1e-6)


def test_latest_checkpoint_resolution(tmp_path):
    from complexity.training.o200k_pretrain import resolve_checkpoint_path

    root = tmp_path / "ckpts"
    step_1 = root / "step_000001"
    step_2 = root / "step_000002"
    step_1.mkdir(parents=True)
    step_2.mkdir()
    (step_1 / "checkpoint.pt").write_bytes(b"1")
    (step_2 / "checkpoint.pt").write_bytes(b"2")
    (root / "latest").write_text("step_000001\n")

    assert resolve_checkpoint_path(str(root / "latest")) == step_1


def test_local_checkpoint_save_latest_and_rotation(tmp_path):
    from complexity.utils.local_checkpoint import load_local_checkpoint, save_local_checkpoint

    root = tmp_path / "ckpts"
    save_local_checkpoint(root, step=1, state={"step": 1, "value": torch.tensor([1])}, total_limit=2)
    save_local_checkpoint(root, step=2, state={"step": 2, "value": torch.tensor([2])}, total_limit=2)
    latest_dir = save_local_checkpoint(root, step=3, state={"step": 3, "value": torch.tensor([3])}, total_limit=2)

    assert latest_dir == root / "step_000003"
    assert sorted(path.name for path in root.glob("step_*")) == ["step_000002", "step_000003"]
    assert (root / "latest").read_text(encoding="utf-8").strip() == "step_000003"

    resolved, state = load_local_checkpoint(root / "latest")
    assert resolved == latest_dir
    assert state["step"] == 3
    assert state["value"].item() == 3


def test_tr_diagnostics_reports_gates_rms_and_grads():
    from complexity.core.mlp import MLPConfig, TokenRoutedMLP
    from complexity.training.moe_telemetry import global_tr_diagnostics

    torch.manual_seed(0)
    config = MLPConfig(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        vocab_size=64,
        shared_expert=True,
        shared_intermediate_size=32,
        use_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.1,
        top_k=2,
        top_k_primary_weight=0.5,
        collect_moe_telemetry=True,
    )
    mlp = TokenRoutedMLP(config)
    hidden = torch.randn(2, 5, 16)
    token_ids = torch.randint(0, 64, (2, 5))

    loss = mlp(hidden, token_ids=token_ids).pow(2).mean()
    loss.backward()
    diagnostics = global_tr_diagnostics(mlp, num_experts=4)

    assert diagnostics["shared_gate"] == pytest.approx(1.0)
    assert diagnostics["routed_gate"] == pytest.approx(0.1)
    assert diagnostics["shared_rms"] > 0
    assert diagnostics["routed_rms"] > 0
    assert diagnostics["shared_grad_norm"] > 0
    assert diagnostics["routed_grad_norm"] > 0
    assert all(diagnostics[f"expert_{idx}_grad_norm"] > 0 for idx in range(4))


def test_muon_tr_optimizer_builds_for_o200k_runner():
    from types import SimpleNamespace

    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import build_optimizer, make_config

    args = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        shared_intermediate_size=64,
        vocab_size=128,
        use_mu_guidance=False,
        learn_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.1,
        top_k=2,
        top_k_primary_weight=0.5,
        static_expert_capacity=False,
        routing_strategy="zipf",
        mu_clamp=False,
        mu_norm=False,
        mu_alpha_init=1.0,
        mu_init_value=0.0,
        mu_context_min=-2.0,
        mu_context_max=2.0,
        optimizer="muon_tr",
        lr=3e-4,
        weight_decay=0.1,
        muon_lr=0.01,
        muon_scope="expert",
        expert_lr_scale=1.5,
        shared_lr_scale=1.0,
        expert_weight_decay=0.005,
        shared_weight_decay=0.01,
        muon_ns_steps=5,
        muon_adaptive_ns=False,
        muon_max_lr_ratio=2.0,
        muon_lr_warmup_steps=50,
        muon_skip_ns_warmup_steps=0,
        muon_token_count_scaling=False,
        muon_max_update_rms=1.0,
    )
    model = ComplexityModel(make_config(args))

    optimizer, stats = build_optimizer(args, model)

    assert hasattr(optimizer, "update_token_counts")
    assert stats["muon_expert_params"] > 0
    assert stats["muon_shared_params"] == 0
    assert stats["adamw_params"] > 0


def test_adamw_optimizer_uses_foreach_for_o200k_runner():
    from types import SimpleNamespace

    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import build_optimizer, make_config

    args = SimpleNamespace(
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=32,
        shared_intermediate_size=64,
        vocab_size=128,
        use_mu_guidance=False,
        learn_shared_routed_gates=True,
        shared_gate_init=1.0,
        routed_gate_init=0.1,
        top_k=2,
        top_k_primary_weight=0.5,
        static_expert_capacity=False,
        routing_strategy="zipf",
        mu_clamp=False,
        mu_norm=False,
        mu_alpha_init=1.0,
        mu_init_value=0.0,
        mu_context_min=-2.0,
        mu_context_max=2.0,
        optimizer="adamw",
        lr=3e-4,
        expert_lr_scale=2.0,
        shared_lr_scale=0.75,
        weight_decay=0.1,
        shared_expert_chunk_tokens=0,
        use_custom_kernels="auto",
        moe_telemetry=False,
    )
    model = ComplexityModel(make_config(args))

    optimizer, stats = build_optimizer(args, model)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert stats["adamw_params"] > 0
    assert stats["adamw_impl"] in {"foreach", "default"}
    assert stats["adamw_expert_params"] > 0
    assert stats["adamw_shared_params"] > 0
    assert stats["expert_lr_scale"] == 2.0
    assert stats["shared_lr_scale"] == 0.75
    group_lrs = {
        group["group_name"]: group["lr"]
        for group in optimizer.param_groups
    }
    assert group_lrs["base"] == pytest.approx(3e-4)
    assert group_lrs["shared"] == pytest.approx(2.25e-4)
    assert group_lrs["expert"] == pytest.approx(6e-4)


def test_token_routed_summary_reports_stored_and_active_width():
    from types import SimpleNamespace

    from complexity.training.o200k_pretrain import token_routed_config_summary

    args = SimpleNamespace(
        attention_type="gqa",
        num_attention_heads=8,
        num_key_value_heads=2,
        hidden_size=384,
        num_hidden_layers=10,
        shared_intermediate_size=1392,
        intermediate_size=256,
        top_k=2,
        expert_initialization="gpt_normal",
        routing_strategy="modulo_balanced_secondary",
        top_k_primary_weight=0.5,
        top_k_primary_weight_final=0.5,
        grad_ckpt=False,
        learn_shared_routed_gates=False,
        shared_output_scale=1.0,
        routed_output_scale=1.8,
        shared_output_scale_first_layer=None,
        shared_output_scale_last_layer=None,
        routed_output_scale_first_layer=None,
        routed_output_scale_last_layer=None,
        expert_diversity_lambda=0.0,
        expert_lr_scale=2.0,
    )

    summary = token_routed_config_summary(args)

    assert "stored_width=1648" in summary
    assert "active_width=1520 (92.2%)" in summary
    assert "expert_lr_scale=2" in summary


def test_batch_expert_counts_counts_current_batch():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import batch_expert_counts

    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=16,
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=False,
    )
    model = ComplexityModel(config)
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    counts = batch_expert_counts(model, input_ids, num_experts=4, distributed=False)

    assert counts.sum().item() == input_ids.numel()
    assert counts.shape == (4,)


def test_batch_expert_counts_counts_all_topk_routes():
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel
    from complexity.training.o200k_pretrain import batch_expert_counts

    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=16,
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=False,
        top_k=2,
    )
    model = ComplexityModel(config)
    input_ids = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]])

    counts = batch_expert_counts(model, input_ids, num_experts=4, distributed=False)

    assert counts.sum().item() == input_ids.numel() * 2
    assert counts.shape == (4,)


def test_lsh_threshold_mode_propagates_to_token_routed_mlp():
    from complexity.config import ModelConfig
    from complexity.core.mlp import TokenRoutedMLP
    from complexity.models import ComplexityModel

    config = ModelConfig(
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        intermediate_size=16,
        vocab_size=16,
        mlp_type="token_routed",
        num_experts=4,
        routing_strategy="lsh_hidden",
        lsh_routing=True,
        lsh_threshold_mode="zero",
        shared_expert=False,
    )

    model = ComplexityModel(config)
    mlps = [module for module in model.modules() if isinstance(module, TokenRoutedMLP)]

    assert mlps
    assert all(module.config.lsh_threshold_mode == "zero" for module in mlps)


def test_plan_run_math():
    from complexity.training.plan_run import parse_tokens

    assert parse_tokens("30B") == 30_000_000_000
    assert parse_tokens("1.5M") == 1_500_000
