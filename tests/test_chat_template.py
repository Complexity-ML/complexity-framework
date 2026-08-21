from __future__ import annotations

import json
from pathlib import Path

import pytest

from complexity.inference.chat_template import (
    CHAT_TEMPLATE_ID,
    LEGACY_CHAT_TEMPLATE_ID,
    THINK_FINAL_ENVELOPE,
    default_chat_template,
    huggingface_chat_template,
    load_chat_template_jinja,
    render_assistant_envelope,
    render_inference_prompt,
    render_jinja_inference_prompt,
    render_jinja_messages,
    render_messages_before_assistant,
    render_thinking_inference_prompt,
    validate_chat_template,
)
from scripts.convert_pt_to_mlx import write_chat_template
from scripts.export_tr_hash_vllm import (
    build_config,
    configure_tokenizer_chat_template,
    copy_tokenizer_files,
    strip_tokenizer_chat_template,
)


def test_template_renders_single_turn_exactly() -> None:
    template = default_chat_template()
    assert template["training_projection"] == (
        "naturalize_card_hand_supervise_all_assistant_turns"
    )
    assert template["id"] == "complexity-chat-v2"
    assert template["system_prompt"] == ""
    assert template["assistant_envelope"] == THINK_FINAL_ENVELOPE
    assert render_inference_prompt("Hello", template) == (
        "User:\nHello\n\nAssistant:\n"
    )


def test_card_corpus_v2_direct_projection_is_supported() -> None:
    template = default_chat_template()
    template["training_projection"] = "card_corpus_v2_direct"

    assert validate_chat_template(template)["training_projection"] == (
        "card_corpus_v2_direct"
    )


def test_template_renders_explicit_system_only_when_provided() -> None:
    template = default_chat_template()
    prompt = render_messages_before_assistant(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
        template,
    )

    assert prompt == (
        "System:\nBe concise.\n\nUser:\nHello\n\nAssistant:\n"
    )


def test_template_renders_canonical_think_final_envelope() -> None:
    rendered = render_assistant_envelope(
        "Compute, then verify.",
        "The answer is 4.",
        default_chat_template(),
    )

    assert rendered == (
        "<think>\nCompute, then verify.\n</think>\n"
        "<final>\nThe answer is 4.\n</final>"
    )


def test_thinking_inference_prefills_canonical_start() -> None:
    assert render_thinking_inference_prompt("Hello", default_chat_template()) == (
        "User:\nHello\n\nAssistant:\n<think>\n"
    )


def test_legacy_v1_template_remains_readable() -> None:
    template = default_chat_template()
    template.pop("assistant_envelope")
    template.update(
        {
            "id": LEGACY_CHAT_TEMPLATE_ID,
            "version": 1,
            "system_prompt": "Legacy system.",
        }
    )

    assert validate_chat_template(template)["id"] == LEGACY_CHAT_TEMPLATE_ID
    assert render_inference_prompt("Hello", template).startswith(
        "System:\nLegacy system.\n\n"
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


def test_huggingface_thinking_template_prefills_envelope() -> None:
    from jinja2 import Template

    rendered = Template(
        huggingface_chat_template(default_chat_template(), force_thinking=True)
    ).render(
        messages=[{"role": "user", "content": "Hello"}],
        eos_token="</s>",
        add_generation_prompt=True,
    )
    assert rendered == "User:\nHello\n\nAssistant:\n<think>\n"


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

    assert rendered == render_messages_before_assistant(messages, contract).replace(
        contract["eos_token"],
        "</s>",
    )


def test_huggingface_template_matches_explicit_system_message() -> None:
    from jinja2 import Template

    contract = default_chat_template()
    messages = [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello"},
    ]
    rendered = Template(huggingface_chat_template(contract)).render(
        messages=messages,
        eos_token="</s>",
        add_generation_prompt=True,
    )

    assert rendered == render_messages_before_assistant(messages, contract)


def test_standalone_jinja_renderer_matches_hf_and_vllm_contract(tmp_path) -> None:
    contract = default_chat_template()
    source = huggingface_chat_template(contract)
    (tmp_path / "chat_template.jinja").write_text(source, encoding="utf-8")

    rendered = render_jinja_messages(
        [
            {"role": "user", "content": "First"},
            {"role": "assistant", "content": "Answer"},
            {"role": "user", "content": "Follow-up"},
        ],
        load_chat_template_jinja(tmp_path),
        eos_token="</s>",
        add_generation_prompt=True,
    )

    assert rendered == (
        "User:\nFirst\n\nAssistant:\nAnswer</s>"
        "User:\nFollow-up\n\nAssistant:\n"
    )


def test_standalone_jinja_renderer_builds_mlx_generation_prompt() -> None:
    rendered = render_jinja_inference_prompt(
        "Hello",
        huggingface_chat_template(default_chat_template()),
        eos_token="</s>",
    )

    assert rendered == "User:\nHello\n\nAssistant:\n"


def test_standalone_jinja_loader_rejects_incomplete_mlx_bundle(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="standalone Jinja"):
        load_chat_template_jinja(tmp_path)


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
    assert config["chat_template_file"] == "chat_template.jinja"


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


def test_every_export_removes_stale_tokenizer_chat_template(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "tokenizer_config.json").write_text(
        '{"chat_template": "stale"}',
        encoding="utf-8",
    )

    path = strip_tokenizer_chat_template(output)
    config = __import__("json").loads(path.read_text(encoding="utf-8"))

    assert "chat_template" not in config
    assert "chat_template_id" not in config


def test_repository_tokenizers_never_embed_chat_templates() -> None:
    repository = Path(__file__).resolve().parents[1]
    configs = (
        repository / "tokenizer" / "tokenizer_config.json",
        repository / "tokenizer-code" / "tokenizer_config.json",
    )
    for path in configs:
        config = json.loads(path.read_text(encoding="utf-8"))
        assert "chat_template" not in config, path
        assert "chat_template_id" not in config, path
    assert (repository / "tokenizer" / "chat_template.jinja").read_text(
        encoding="utf-8"
    ) == huggingface_chat_template(default_chat_template()) + "\n"


def test_hf_chat_template_is_a_standalone_jinja_file(tmp_path) -> None:
    (tmp_path / "tokenizer_config.json").write_text(
        '{"chat_template": "stale", "chat_template_id": "stale"}',
        encoding="utf-8",
    )
    path = configure_tokenizer_chat_template(tmp_path, default_chat_template())

    assert path is not None
    assert path.name == "chat_template.jinja"
    assert "User:\\n" in path.read_text(encoding="utf-8")
    assert "+ eos_token" in path.read_text(encoding="utf-8")
    tokenizer_config = json.loads(
        (tmp_path / "tokenizer_config.json").read_text(encoding="utf-8")
    )
    assert "chat_template" not in tokenizer_config
    assert "chat_template_id" not in tokenizer_config


def test_base_export_removes_tokenizer_chat_template(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "tokenizer_config.json").write_text(
        '{"chat_template": "stale", "chat_template_id": "chat-v1", '
        '"model_max_length": 2048}',
        encoding="utf-8",
    )

    (output / "chat_template.jinja").write_text("stale", encoding="utf-8")
    jinja_path = configure_tokenizer_chat_template(output, None)
    config = json.loads(
        (output / "tokenizer_config.json").read_text(encoding="utf-8")
    )

    assert jinja_path is None
    assert not (output / "chat_template.jinja").exists()
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
    assert (tmp_path / "chat_template.jinja").read_text(encoding="utf-8") == (
        huggingface_chat_template(template) + "\n"
    )
    assert (
        __import__("json").loads(
            (tmp_path / "chat_template.json").read_text()
        )["training_projection"]
        == "naturalize_card_hand_preserve_assistant_turns"
    )
