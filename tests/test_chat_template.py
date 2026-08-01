from __future__ import annotations

from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    render_inference_prompt,
    render_messages_before_assistant,
)
from scripts.export_tr_hash_vllm import build_config


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
