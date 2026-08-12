from __future__ import annotations

from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    default_chat_template,
    huggingface_chat_template,
    render_inference_prompt,
    render_messages_before_assistant,
)
from scripts.convert_pt_to_mlx import write_chat_template
from scripts.export_tr_hash_vllm import (
    build_config,
    copy_tokenizer_files,
    strip_tokenizer_chat_template,
    write_tokenizer_chat_template,
)


def test_template_renders_single_turn_exactly() -> None:
    template = default_chat_template()
    assert template["training_projection"] == (
        "naturalize_card_hand_supervise_all_assistant_turns"
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
    assert "Assistant:\nAnswer<|endoftext|>User:\nFollow-up" in prompt
    assert prompt.endswith("Assistant:\n")


def test_huggingface_template_uses_the_same_prompt_contract() -> None:
    rendered = huggingface_chat_template(default_chat_template())

    assert "System:\\n" in rendered
    assert "User:\\n" in rendered
    assert "Assistant:\\n" in rendered
    assert '"<|endoftext|>"' not in rendered
    assert "+ eos_token" in rendered


def test_huggingface_template_matches_sft_for_multi_turn_conversation() -> None:
    from jinja2 import Template

    contract = default_chat_template()
    messages = [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Answer"},
        {"role": "user", "content": "Follow-up"},
    ]

    rendered = Template(huggingface_chat_template(contract)).render(
        messages=messages,
        eos_token="</s>",
        add_generation_prompt=True,
    )

    assert rendered == (
        contract["system_format"].format(content=contract["system_prompt"])
        + contract["user_format"].format(content="First")
        + contract["assistant_prefix"]
        + "Answer</s>"
        + contract["user_format"].format(content="Follow-up")
        + contract["assistant_prefix"]
    )


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
    (source / "tokenizer_config.json").write_text("{}")
    (source / "model.safetensors").write_text("base weights")
    (source / "config.json").write_text("base config")
    (output / "model.safetensors").write_text("fine-tuned weights")
    (output / "config.json").write_text("export config")

    copied = copy_tokenizer_files(source, output)

    assert copied == ["tokenizer.json", "tokenizer_config.json"]
    assert (output / "model.safetensors").read_text() == "fine-tuned weights"
    assert (output / "config.json").read_text() == "export config"


def test_export_replaces_stale_tokenizer_chat_template(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "tokenizer_config.json").write_text(
        '{"chat_template": "stale"}',
        encoding="utf-8",
    )

    path = write_tokenizer_chat_template(output, default_chat_template())
    config = __import__("json").loads(path.read_text(encoding="utf-8"))

    assert config["chat_template"] != "stale"
    assert config["chat_template_id"] == CHAT_TEMPLATE_ID
    assert '"<|endoftext|>"' not in config["chat_template"]
    assert "+ eos_token" in config["chat_template"]


def test_base_export_removes_tokenizer_chat_template(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "tokenizer_config.json").write_text(
        '{"chat_template": "stale", "chat_template_id": "chat-v1", '
        '"model_max_length": 2048}',
        encoding="utf-8",
    )

    path = strip_tokenizer_chat_template(output)
    config = __import__("json").loads(path.read_text(encoding="utf-8"))

    assert "chat_template" not in config
    assert "chat_template_id" not in config
    assert config["model_max_length"] == 2048


def test_base_config_has_no_chat_template_metadata() -> None:
    config = build_config(
        {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "intermediate_size": 128,
            "vocab_size": 256,
        },
        chat_template=None,
    )

    assert "chat_template_id" not in config
    assert "chat_template_file" not in config


def test_mlx_export_preserves_checkpoint_chat_template(tmp_path) -> None:
    template = default_chat_template()
    template["training_projection"] = (
        "naturalize_card_hand_preserve_assistant_turns"
    )

    written = write_chat_template({"chat_template": template}, tmp_path)

    assert written == template
    assert (tmp_path / "chat_template.json").read_text().endswith("\n")
    assert (
        __import__("json").loads(
            (tmp_path / "chat_template.json").read_text()
        )["training_projection"]
        == "naturalize_card_hand_preserve_assistant_turns"
    )
