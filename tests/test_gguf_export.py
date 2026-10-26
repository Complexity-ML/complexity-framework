"""Reject unrepresentable TR-HASH exports before touching the destination."""

import struct
from types import SimpleNamespace

import pytest
from torch import nn

from complexity import ComplexityModel, ModelConfig
from complexity.core.attention import AttentionConfig
from complexity.core.attention.gqa import TRHashAttention
from complexity.quantization import export_gguf
from complexity.quantization import gguf_export
from complexity.tr_hash import TRHashEngine, TRHashEngineConfig
from complexity.tr_hash.attention import DomainSelfAttention


def _routed_model(kind):
    if kind == "native":
        return ComplexityModel(
            ModelConfig(
                vocab_size=16,
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=1,
                intermediate_size=16,
                use_custom_kernels=False,
            )
        )
    if kind == "nested_engine":
        return nn.Sequential(
            TRHashEngine(
                TRHashEngineConfig(
                    hidden_size=16,
                    vocab_size=16,
                    shared_width=32,
                    expert_width=4,
                )
            )
        )
    if kind == "domain_attention":
        return nn.Sequential(DomainSelfAttention(16))
    if kind == "language_attention":
        return TRHashAttention(
            AttentionConfig(
                hidden_size=16,
                num_attention_heads=4,
                num_key_value_heads=1,
            )
        )
    model = nn.Linear(4, 4, bias=False)
    if kind == "exported_config":
        model.config = SimpleNamespace(model_type="tr_hash_moe")
    else:
        model.config = {"mlp_type": "tr_hash_moe"}
    return model


@pytest.mark.parametrize(
    "kind",
    [
        "native",
        "nested_engine",
        "domain_attention",
        "language_attention",
        "exported_config",
        "mapping_config",
    ],
)
@pytest.mark.parametrize("existing", [False, True])
def test_tr_hash_rejected_before_writer_or_output_changes(tmp_path, monkeypatch, kind, existing):
    model = _routed_model(kind)
    destination = tmp_path / "model.gguf"
    sentinel = b"previous complete export"
    if existing:
        destination.write_bytes(sentinel)

    def unexpected_writer(*args, **kwargs):
        raise AssertionError("TR-HASH must fail before writer construction")

    monkeypatch.setattr(gguf_export, "GGUFWriter", unexpected_writer)
    with pytest.raises(ValueError, match="GGUF export cannot represent TR-HASH.*two-lag"):
        export_gguf(model, str(destination), quantization="F32")
    if existing:
        assert destination.read_bytes() == sentinel
    else:
        assert not destination.exists()


def test_dense_generic_model_still_exports(tmp_path):
    model = nn.Linear(4, 4, bias=False)
    # Generic configs also carry the default routing string; the MLP actually
    # selected must determine whether this is a routed architecture.
    model.config = SimpleNamespace(
        mlp_type="swiglu",
        attention_type="gqa",
        routing_strategy="token_id_composed_modulo",
        hidden_size=4,
        num_hidden_layers=1,
        intermediate_size=16,
        num_attention_heads=1,
        num_key_value_heads=1,
    )
    destination = tmp_path / "dense.gguf"
    export_gguf(model, str(destination), quantization="F32")
    payload = destination.read_bytes()
    assert payload[:4] == b"GGUF"
    assert struct.unpack_from("<I", payload, 4)[0] == 3
    assert struct.unpack_from("<Q", payload, 8)[0] == 1
    assert b"llama" in payload
    assert payload.endswith(model.weight.detach().numpy().tobytes())
