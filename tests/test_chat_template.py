from __future__ import annotations

from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    render_inference_prompt,
    render_messages_before_assistant,
)
from scripts.export_tr_hash_vllm import build_config, copy_tokenizer_files


def test_template_renders_single_turn_exactly() -> None:
    template = default_chat_template()
    assert template["training_projection"] == (
        "naturalize_card_hand_target_final_assistant"
    )
    assert render_inference_prompt("Hello", template) == (
        "System:\n"
        "You are Complexity, a concise and grounded assistant. Answer the user "
        "directly. Use provided evidence when present and do not invent missing facts.\n\n"
        "User:\nHello\n\nAssistant:\n"
    )


def test_template_renders_prior_assistant_turn() -> None:
    template = default_chat_template()
    prompt = render_messages_before_assistant(
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ],
        template,
    )
    assert "Assistant:\nAnswer\n\nUser:\nFollow-up" in prompt
    assert prompt.endswith("Assistant:\n")


def test_vllm_config_declares_exported_template() -> None:
    template = default_chat_template()
    config = build_config(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 256,
        },
        template,
    )
    assert config["chat_template_id"] == CHAT_TEMPLATE_ID
    assert config["chat_template_file"] == "chat_template.json"


def test_vllm_export_preserves_legacy_modulo_cyclic_routing() -> None:
    legacy = build_config(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 256,
        }
    )
    current = build_config(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 256,
            "routing_strategy": "token_id_pair_coverage_hash",
        }
    )

    assert legacy["routing_strategy"] == "modulo_cyclic"
    assert current["routing_strategy"] == "token_id_pair_coverage_hash"


def test_export_copies_only_tokenizer_assets(tmp_path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    (source / "tokenizer.json").write_text("tokenizer")
    (source / "tokenizer_config.json").write_text("tokenizer config")
    (source / "model.safetensors").write_text("base weights")
    (source / "config.json").write_text("base config")
    (output / "model.safetensors").write_text("fine-tuned weights")
    (output / "config.json").write_text("export config")

    copied = copy_tokenizer_files(source, output)

    assert copied == ["tokenizer.json", "tokenizer_config.json"]
    assert (output / "model.safetensors").read_text() == "fine-tuned weights"
    assert (output / "config.json").read_text() == "export config"
