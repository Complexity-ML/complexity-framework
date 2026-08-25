"""Portable chat-template contract shared by SFT, export, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, StrictUndefined

CHAT_TEMPLATE_ID = "complexity-chat-v2"
LEGACY_CHAT_TEMPLATE_ID = "complexity-chat-v1"
SUPPORTED_CHAT_TEMPLATE_IDS = {
    LEGACY_CHAT_TEMPLATE_ID,
    CHAT_TEMPLATE_ID,
}
SUPPORTED_TRAINING_PROJECTIONS = {
    "naturalize_card_hand_preserve_assistant_turns",
    "naturalize_card_hand_target_final_assistant",
    "naturalize_card_hand_supervise_all_assistant_turns",
    "card_corpus_v2_direct",
}
DEFAULT_SYSTEM_PROMPT = ""
THINK_FINAL_ENVELOPE = {
    "type": "optional_think_final",
    "think_start": "<think>\n",
    "think_end": "\n</think>",
    "final_start": "\n<final>\n",
    "final_end": "\n</final>",
    "scope": "reasoning_tasks",
}
REQUIRED_FIELDS = (
    "id",
    "version",
    "system_prompt",
    "system_format",
    "user_format",
    "assistant_prefix",
    "turn_separator",
    "eos_token",
    "assistant_only_loss",
    "training_projection",
)


def default_chat_template() -> dict[str, Any]:
    return {
        "id": CHAT_TEMPLATE_ID,
        "version": 2,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_format": "System:\n{content}\n\n",
        "user_format": "User:\n{content}\n\n",
        "assistant_prefix": "Assistant:\n",
        "turn_separator": "\n\n",
        "eos_token": "<|endoftext|>",
        "assistant_only_loss": True,
        "training_projection": "naturalize_card_hand_supervise_all_assistant_turns",
        "assistant_envelope": dict(THINK_FINAL_ENVELOPE),
    }


def validate_chat_template(template: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in template]
    if missing:
        raise ValueError(f"Chat template is missing fields: {', '.join(missing)}")
    if template["id"] not in SUPPORTED_CHAT_TEMPLATE_IDS:
        raise ValueError(f"Unsupported chat template: {template['id']}")
    if template["id"] == CHAT_TEMPLATE_ID:
        envelope = template.get("assistant_envelope")
        if envelope != THINK_FINAL_ENVELOPE:
            raise ValueError("complexity-chat-v2 requires the canonical think/final protocol")
    if not template["assistant_only_loss"]:
        raise ValueError("Complexity SFT requires assistant_only_loss=true")
    if template["training_projection"] not in SUPPORTED_TRAINING_PROJECTIONS:
        raise ValueError(f"Unsupported SFT training projection: {template['training_projection']}")
    for field in ("system_format", "user_format"):
        if "{content}" not in template[field]:
            raise ValueError(f"Chat template {field} must contain {{content}}")
    return dict(template)


def align_chat_template_eos(
    template: dict[str, Any],
    *,
    eos_token: str,
) -> dict[str, Any]:
    """Align a chat template to the loaded tokenizer's EOS spelling.

    The semantic contract is the EOS token ID. Native and exported tokenizers
    can spell that same ID differently, so SFT must not encode an unknown EOS
    spelling as a sequence of ordinary tokens between conversation turns.
    """

    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("chat-template EOS alignment requires a non-empty token")
    aligned = validate_chat_template(template)
    aligned["eos_token"] = eos_token
    return aligned


def load_chat_template(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return default_chat_template()
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "chat_template.json"
    if not candidate.exists():
        return default_chat_template()
    return validate_chat_template(json.loads(candidate.read_text(encoding="utf-8")))


def load_chat_template_jinja(path: str | Path) -> str:
    """Load the standalone Jinja contract shipped beside a model bundle."""

    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "chat_template.jinja"
    if not candidate.is_file():
        raise FileNotFoundError(f"Model bundle has no standalone Jinja chat template: {candidate}")
    source = candidate.read_text(encoding="utf-8")
    if not source.strip():
        raise ValueError(f"Jinja chat template is empty: {candidate}")
    return source


def render_jinja_messages(
    messages: Iterable[dict[str, Any]],
    jinja_source: str,
    *,
    eos_token: str,
    add_generation_prompt: bool,
) -> str:
    """Render the same standalone Jinja contract used by HF and vLLM."""

    if not isinstance(eos_token, str) or not eos_token:
        raise ValueError("A non-empty tokenizer EOS token is required for Jinja rendering")
    environment = Environment(
        autoescape=False,
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )
    return environment.from_string(jinja_source).render(
        messages=list(messages),
        eos_token=eos_token,
        add_generation_prompt=add_generation_prompt,
    )


def render_jinja_inference_prompt(
    user_content: str,
    jinja_source: str,
    *,
    eos_token: str,
) -> str:
    """Render one user turn and the assistant generation prefix via Jinja."""

    return render_jinja_messages(
        [{"role": "user", "content": user_content}],
        jinja_source,
        eos_token=eos_token,
        add_generation_prompt=True,
    )


def render_system_prefix(template: dict[str, Any]) -> str:
    system_prompt = str(template["system_prompt"]).strip()
    if not system_prompt:
        return ""
    return template["system_format"].format(content=system_prompt)


def render_system_turn(content: str, template: dict[str, Any]) -> str:
    content = content.strip()
    if not content:
        return ""
    return template["system_format"].format(content=content)


def render_user_turn(content: str, template: dict[str, Any]) -> str:
    return template["user_format"].format(content=content.strip())


def render_inference_prompt(user_content: str, template: dict[str, Any]) -> str:
    return (
        render_system_prefix(template)
        + render_user_turn(user_content, template)
        + template["assistant_prefix"]
    )


def render_thinking_inference_prompt(user_content: str, template: dict[str, Any]) -> str:
    """Render inference with the canonical thinking envelope prefilled.

    Training keeps ``<think>`` in the supervised completion. At inference,
    prefilling that learned first token makes Thinking mode deterministic
    without changing the training projection or the ordinary chat mode.
    """

    template = validate_chat_template(template)
    envelope = template.get("assistant_envelope")
    if not envelope:
        raise ValueError("Chat template has no assistant envelope")
    return render_inference_prompt(user_content, template) + envelope["think_start"]


def render_messages_before_assistant(
    messages: Iterable[dict[str, Any]],
    template: dict[str, Any],
) -> str:
    """Render prior turns and finish at the next assistant prefix."""

    messages = list(messages)
    explicit_systems = [
        str(message.get("content", "")).strip()
        for message in messages
        if str(message.get("role", "")).strip().lower() == "system"
        and str(message.get("content", "")).strip()
    ]
    parts = (
        [render_system_prefix(template)]
        if template["id"] == LEGACY_CHAT_TEMPLATE_ID or not explicit_systems
        else []
    )
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            if template["id"] == CHAT_TEMPLATE_ID:
                parts.append(render_system_turn(content, template))
            continue
        if role == "user":
            parts.append(render_user_turn(content, template))
        elif role == "assistant":
            parts.append(template["assistant_prefix"] + content + template["eos_token"])
        else:
            raise ValueError(f"Unsupported chat role: {role}")
    parts.append(template["assistant_prefix"])
    return "".join(parts)


def render_assistant_envelope(
    reasoning: str,
    final: str,
    template: dict[str, Any],
) -> str:
    """Render the canonical supervised response body for chat-v2."""

    template = validate_chat_template(template)
    envelope = template.get("assistant_envelope")
    if not envelope:
        raise ValueError("Chat template has no assistant envelope")
    return (
        envelope["think_start"]
        + reasoning.strip()
        + envelope["think_end"]
        + envelope["final_start"]
        + final.strip()
        + envelope["final_end"]
    )


def huggingface_chat_template(template: dict[str, Any], *, force_thinking: bool = False) -> str:
    """Build a Jinja template aligned with the exported tokenizer's EOS ID.

    Native training tokenizers may spell ID 0 as ``<|endoftext|>`` while the
    Hugging Face tokenizer exported beside the model spells the same ID as
    ``</s>``. Referencing Jinja's ``eos_token`` preserves the trained token ID
    instead of encoding the native spelling as ordinary text tokens.
    """

    template = validate_chat_template(template)
    user_prefix, user_suffix = template["user_format"].split("{content}", 1)
    literals = {
        "system": json.dumps(render_system_prefix(template)),
        "system_prefix": json.dumps(template["system_format"].split("{content}", 1)[0]),
        "system_suffix": json.dumps(template["system_format"].split("{content}", 1)[1]),
        "user_prefix": json.dumps(user_prefix),
        "user_suffix": json.dumps(user_suffix),
        "assistant": json.dumps(template["assistant_prefix"]),
        "thinking_generation": json.dumps(
            template["assistant_prefix"]
            + (
                template["assistant_envelope"]["think_start"]
                if force_thinking and template.get("assistant_envelope")
                else ""
            )
        ),
    }
    if template["id"] == CHAT_TEMPLATE_ID:
        role_branches = (
            "{%- if message['role'] == 'system' -%}"
            f"{{{{- {literals['system_prefix']} + (message['content'] | trim) + "
            f"{literals['system_suffix']} -}}}}"
            "{%- elif message['role'] == 'user' -%}"
            f"{{{{- {literals['user_prefix']} + (message['content'] | trim) + "
            f"{literals['user_suffix']} -}}}}"
            "{%- elif message['role'] == 'assistant' -%}"
            f"{{{{- {literals['assistant']} + (message['content'] | trim) + eos_token -}}}}"
            "{%- endif -%}"
        )
    else:
        role_branches = (
            "{%- if message['role'] == 'user' -%}"
            f"{{{{- {literals['user_prefix']} + (message['content'] | trim) + "
            f"{literals['user_suffix']} -}}}}"
            "{%- elif message['role'] == 'assistant' -%}"
            f"{{{{- {literals['assistant']} + (message['content'] | trim) + eos_token -}}}}"
            "{%- endif -%}"
        )
    return (
        f"{{{{- {literals['system']} -}}}}"
        + "{%- for message in messages -%}"
        + role_branches
        + "{%- endfor -%}"
        + "{%- if add_generation_prompt -%}"
        + f"{{{{- {literals['thinking_generation']} -}}}}"
        + "{%- endif -%}"
    )
