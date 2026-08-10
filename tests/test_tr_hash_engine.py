import pytest
import torch
import torch.nn.functional as F

from complexity.tr_hash import (
    AttentionBackbone,
    GraphBucket,
    GraphBucketPlanner,
    TRHashBackend,
    TRHashEngine,
    TRHashEngineConfig,
    TRHashPhase,
    TRHashPrecision,
    TRHashStrategy,
    build_route_table,
    compile_top2_pair_metadata,
    decode_top2_pair_metadata,
    select_backend,
    supports_fused_cuda,
)


@pytest.mark.parametrize("num_experts", [2, 4, 8, 16])
@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_supported_expert_and_topk_matrix(num_experts, top_k):
    if top_k > num_experts:
        with pytest.raises(ValueError, match="top_k cannot exceed"):
            TRHashEngineConfig(
                hidden_size=8,
                vocab_size=97,
                num_experts=num_experts,
                top_k=top_k,
                expert_width=4,
            )
        return

    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=97,
        num_experts=num_experts,
        top_k=top_k,
        shared_width=12,
        expert_width=4,
    )
    assert config.stored_routed_width == num_experts * 4
    assert config.active_routed_width == top_k * 4
    assert config.stored_mlp_width == 12 + num_experts * 4
    assert config.active_mlp_width == 12 + top_k * 4


@pytest.mark.parametrize(
    "strategy",
    [
        TRHashStrategy.MODULO_CYCLIC,
        TRHashStrategy.BALANCED_HASH,
        TRHashStrategy.AFFINE_HASH,
    ],
)
@pytest.mark.parametrize("num_experts,top_k", [(2, 1), (4, 2), (8, 4), (16, 4)])
def test_route_tables_are_deterministic_distinct_and_balanced_enough(
    strategy,
    num_experts,
    top_k,
):
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=997,
        num_experts=num_experts,
        top_k=top_k,
        expert_width=4,
        routing_strategy=strategy,
        layer_index=3,
    )
    first = build_route_table(config)
    second = build_route_table(config)

    assert torch.equal(first, second)
    assert first.shape == (top_k, 997)
    for token_routes in first.T:
        assert len(set(token_routes.tolist())) == top_k
    for route_index in range(top_k):
        counts = torch.bincount(first[route_index], minlength=num_experts)
        # Exact route-table balancing for modulo/balanced; affine is a stable
        # hash and therefore only expected to avoid pathological collapse.
        tolerance = 1 if strategy is not TRHashStrategy.AFFINE_HASH else 80
        assert int(counts.max() - counts.min()) <= tolerance


def test_balanced_hash_varies_by_layer_without_changing_loads():
    common = dict(
        hidden_size=8,
        vocab_size=1_009,
        num_experts=8,
        top_k=4,
        expert_width=4,
        routing_strategy=TRHashStrategy.BALANCED_HASH,
    )
    layer_zero = build_route_table(TRHashEngineConfig(**common, layer_index=0))
    layer_one = build_route_table(TRHashEngineConfig(**common, layer_index=1))

    assert not torch.equal(layer_zero, layer_one)
    for route_index in range(4):
        counts_zero = torch.bincount(layer_zero[route_index], minlength=8)
        counts_one = torch.bincount(layer_one[route_index], minlength=8)
        assert int(counts_zero.max() - counts_zero.min()) <= 1
        assert int(counts_one.max() - counts_one.min()) <= 1


@pytest.mark.parametrize(
    "strategy",
    [
        TRHashStrategy.MODULO_CYCLIC,
        TRHashStrategy.BALANCED_HASH,
        TRHashStrategy.AFFINE_HASH,
    ],
)
@pytest.mark.parametrize("num_experts", [2, 4])
def test_compact_top2_pair_metadata_round_trips(strategy, num_experts):
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=997,
        num_experts=num_experts,
        top_k=2,
        expert_width=4,
        routing_strategy=strategy,
        layer_index=3,
    )
    routes = build_route_table(config)
    route_codes, expert_pairs = compile_top2_pair_metadata(
        routes,
        num_experts=num_experts,
    )

    assert route_codes.dtype is torch.uint8
    assert route_codes.shape == (config.vocab_size,)
    assert expert_pairs.shape == (num_experts * (num_experts - 1) // 2, 2)
    assert torch.equal(
        decode_top2_pair_metadata(route_codes, expert_pairs),
        routes,
    )


def test_compact_pair_metadata_rejects_unsupported_shape():
    routes = torch.tensor([[0, 1], [1, 2]])
    with pytest.raises(ValueError, match="two to four"):
        compile_top2_pair_metadata(routes, num_experts=8)
    with pytest.raises(ValueError, match="distinct"):
        compile_top2_pair_metadata(
            torch.tensor([[0, 1], [0, 2]]),
            num_experts=4,
        )


def _naive_engine_output(engine, hidden, token_ids):
    config = engine.config
    output = torch.empty_like(hidden)
    routes = engine.route_table[:, token_ids]
    for batch_index in range(hidden.size(0)):
        for token_index in range(hidden.size(1)):
            x = hidden[batch_index, token_index]
            if engine.shared_gate is None:
                shared = torch.zeros_like(x)
            else:
                shared = engine.shared_down(F.silu(engine.shared_gate(x)) * engine.shared_up(x))
            routed = torch.zeros_like(x)
            for route_index in range(config.top_k):
                expert = int(routes[route_index, batch_index, token_index])
                intermediate = F.silu(x @ engine.expert_gate[expert]) * (
                    x @ engine.expert_up[expert]
                )
                routed = routed + (
                    config.route_weights[route_index] * (intermediate @ engine.expert_down[expert])
                )
            output[batch_index, token_index] = (
                config.shared_output_scale * shared + config.routed_output_scale * routed
            )
    return output


@pytest.mark.parametrize("num_experts,top_k", [(2, 1), (4, 2), (8, 4), (16, 4)])
def test_reference_engine_matches_token_by_token_definition_and_gradients(
    num_experts,
    top_k,
):
    torch.manual_seed(7)
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=101,
        num_experts=num_experts,
        top_k=top_k,
        shared_width=12,
        expert_width=4,
        precision=TRHashPrecision.FP32,
        backend=TRHashBackend.PYTORCH,
    )
    engine = TRHashEngine(config)
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    token_ids = torch.randint(0, config.vocab_size, (2, 3))

    actual = engine(hidden, token_ids)
    expected = _naive_engine_output(engine, hidden, token_ids)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)

    actual.square().mean().backward()
    assert hidden.grad is not None
    assert torch.isfinite(hidden.grad).all()
    for parameter in engine.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_attention_backbone_is_metadata_not_a_different_mlp_function():
    common = dict(
        hidden_size=8,
        vocab_size=101,
        num_experts=4,
        top_k=2,
        shared_width=12,
        expert_width=4,
        precision=TRHashPrecision.FP32,
        backend=TRHashBackend.PYTORCH,
    )
    gqa = TRHashEngine(TRHashEngineConfig(**common, attention_backbone=AttentionBackbone.GQA))
    mha = TRHashEngine(TRHashEngineConfig(**common, attention_backbone=AttentionBackbone.MHA))
    mha.load_state_dict(gqa.state_dict())
    hidden = torch.randn(2, 3, 8)
    token_ids = torch.randint(0, 101, (2, 3))

    assert torch.equal(gqa(hidden, token_ids), mha(hidden, token_ids))


def test_auto_backend_reports_cpu_fallback_and_multi_gpu_contract():
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=101,
        world_size=4,
        precision=TRHashPrecision.FP32,
    )
    engine = TRHashEngine(config)
    summary = engine.capability_summary("cpu")

    assert summary["selected_backend"] == "pytorch"
    assert summary["parallelism"] == "replicated_ddp"
    assert "CGGR requires CUDA" in summary["backend_reasons"]


def test_backend_selection_prefers_fused_cuda_for_supported_pair_shape():
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=101,
        precision=TRHashPrecision.FP16,
    )
    decision = select_backend(
        config,
        device_type="cuda",
        cggr_available=True,
        fused_cuda_available=True,
    )
    assert decision.selected is TRHashBackend.FUSED_CUDA
    assert supports_fused_cuda(config)


def test_backend_selection_falls_back_to_general_cggr_shape():
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=101,
        num_experts=8,
        top_k=4,
        precision=TRHashPrecision.FP16,
    )
    decision = select_backend(
        config,
        device_type="cuda",
        cggr_available=True,
        fused_cuda_available=True,
    )
    assert decision.selected is TRHashBackend.CGGR
    assert not supports_fused_cuda(config)


def test_explicit_fused_cuda_rejects_unsupported_shape():
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=101,
        num_experts=8,
        top_k=4,
        precision=TRHashPrecision.FP16,
        backend=TRHashBackend.FUSED_CUDA,
    )
    with pytest.raises(RuntimeError, match="top_k=2"):
        select_backend(
            config,
            device_type="cuda",
            cggr_available=True,
            fused_cuda_available=True,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_cuda_matches_reference_forward_and_gradients():
    from complexity.tr_hash.engine import HAS_FUSED_CUDA

    if not HAS_FUSED_CUDA:
        pytest.skip("hash-native Triton kernels are unavailable")

    torch.manual_seed(29)
    common = dict(
        hidden_size=16,
        vocab_size=257,
        num_experts=4,
        top_k=2,
        shared_width=24,
        expert_width=8,
        precision=TRHashPrecision.FP32,
    )
    fused = TRHashEngine(
        TRHashEngineConfig(
            **common,
            backend=TRHashBackend.FUSED_CUDA,
        )
    ).cuda()
    reference = TRHashEngine(
        TRHashEngineConfig(
            **common,
            backend=TRHashBackend.PYTORCH,
        )
    ).cuda()
    reference.load_state_dict(fused.state_dict())

    token_ids = torch.randint(0, 257, (2, 11), device="cuda")
    fused_hidden = torch.randn(
        2,
        11,
        16,
        device="cuda",
        requires_grad=True,
    )
    reference_hidden = fused_hidden.detach().clone().requires_grad_(True)
    fused_output = fused(fused_hidden, token_ids)
    reference_output = reference(reference_hidden, token_ids)
    assert torch.allclose(
        fused_output,
        reference_output,
        atol=2e-3,
        rtol=2e-3,
    )

    output_gradient = torch.randn_like(fused_output)
    fused_output.backward(output_gradient)
    reference_output.backward(output_gradient)
    assert torch.allclose(
        fused_hidden.grad,
        reference_hidden.grad,
        atol=3e-3,
        rtol=3e-3,
    )
    for fused_parameter, reference_parameter in zip(
        fused.parameters(),
        reference.parameters(),
    ):
        assert fused_parameter.grad is not None
        assert reference_parameter.grad is not None
        assert torch.allclose(
            fused_parameter.grad,
            reference_parameter.grad,
            atol=3e-3,
            rtol=3e-3,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_fused_cuda_accepts_fp32_residuals_with_bf16_expert_weights():
    from complexity.tr_hash.engine import HAS_FUSED_CUDA

    if not HAS_FUSED_CUDA:
        pytest.skip("hash-native Triton kernels are unavailable")

    torch.manual_seed(31)
    common = dict(
        hidden_size=16,
        vocab_size=257,
        num_experts=4,
        top_k=2,
        shared_width=0,
        expert_width=8,
        precision=TRHashPrecision.BF16,
    )
    fused = TRHashEngine(
        TRHashEngineConfig(**common, backend=TRHashBackend.FUSED_CUDA)
    ).cuda().to(torch.bfloat16).eval()
    reference = TRHashEngine(
        TRHashEngineConfig(**common, backend=TRHashBackend.PYTORCH)
    ).cuda().to(torch.bfloat16).eval()
    reference.load_state_dict(fused.state_dict())

    token_ids = torch.randint(0, 257, (1, 19), device="cuda")
    residual = torch.randn(1, 19, 16, device="cuda", dtype=torch.float32)
    with torch.no_grad():
        fused_output = fused(residual, token_ids)
        reference_output = reference(residual.to(torch.bfloat16), token_ids).float()

    assert fused_output.dtype is torch.float32
    assert torch.allclose(fused_output, reference_output, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("precision", [TRHashPrecision.FP8, TRHashPrecision.INT8])
def test_quantized_modes_fail_explicitly_until_phase_two_kernel(precision):
    config = TRHashEngineConfig(
        hidden_size=8,
        vocab_size=101,
        precision=precision,
    )
    with pytest.raises(NotImplementedError, match="phase-2"):
        select_backend(config, device_type="cuda", cggr_available=True)


def test_cuda_graph_bucket_selection_is_smallest_containing_shape():
    planner = GraphBucketPlanner(
        [
            GraphBucket(1, 128),
            GraphBucket(2, 128),
            GraphBucket(4, 256),
            GraphBucket(1, 512),
        ]
    )
    assert planner.select(1, 80) == GraphBucket(1, 128)
    assert planner.select(2, 80) == GraphBucket(2, 128)
    assert planner.select(3, 200) == GraphBucket(4, 256)
    with pytest.raises(ValueError, match="no CUDA Graph bucket"):
        planner.select(8, 512)


def test_cuda_graph_config_is_inference_only_and_requires_buckets():
    with pytest.raises(ValueError, match="inference-only"):
        TRHashEngineConfig(
            hidden_size=8,
            vocab_size=101,
            backend=TRHashBackend.CUDA_GRAPH,
            graph_buckets=(GraphBucket(1, 128),),
        )
    with pytest.raises(ValueError, match="requires at least one"):
        TRHashEngineConfig(
            hidden_size=8,
            vocab_size=101,
            phase=TRHashPhase.INFERENCE,
            backend=TRHashBackend.CUDA_GRAPH,
        )
