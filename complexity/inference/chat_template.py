"""Portable chat-template contract shared by SFT, export, and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

CHAT_TEMPLATE_ID = "complexity-chat-v1"
DEFAULT_SYSTEM_PROMPT = (
    "You are Complexity, a concise and grounded assistant. Answer the user "
    "directly. Use provided evidence when present and do not invent missing facts."
)
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
        "version": 1,
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "system_format": "System:\n{content}\n\n",
        "user_format": "User:\n{content}\n\n",
        "assistant_prefix": "Assistant:\n",
        "turn_separator": "\n\n",
        "eos_token": "<|endoftext|>",
        "assistant_only_loss": True,
        "training_projection": "naturalize_card_hand_target_final_assistant",
    }


def validate_chat_template(template: dict[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in template]
    if missing:
        raise ValueError(f"Chat template is missing fields: {', '.join(missing)}")
    if template["id"] != CHAT_TEMPLATE_ID:
        raise ValueError(f"Unsupported chat template: {template['id']}")
    if not template["assistant_only_loss"]:
        raise ValueError("Complexity SFT requires assistant_only_loss=true")
    supported_projections = {
        "naturalize_card_hand_preserve_assistant_turns",
        "naturalize_card_hand_target_final_assistant",
    }
    if template["training_projection"] not in supported_projections:
        raise ValueError(
            "Unsupported SFT training projection: "
            f"{template['training_projection']}"
        )
    for field in ("system_format", "user_format"):
        if "{content}" not in template[field]:
            raise ValueError(f"Chat template {field} must contain {{content}}")
    return dict(template)


def load_chat_template(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return default_chat_template()
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / "chat_template.json"
    if not candidate.exists():
        return default_chat_template()
    return validate_chat_template(json.loads(candidate.read_text(encoding="utf-8")))


def render_system_prefix(template: dict[str, Any]) -> str:
    return template["system_format"].format(content=template["system_prompt"])


def render_user_turn(content: str, template: dict[str, Any]) -> str:
    return template["user_format"].format(content=content.strip())


def render_inference_prompt(user_content: str, template: dict[str, Any]) -> str:
    return (
        render_system_prefix(template)
        + render_user_turn(user_content, template)
        + template["assistant_prefix"]
    )


def render_messages_before_assistant(
    messages: Iterable[dict[str, Any]],
    template: dict[str, Any],
) -> str:
    """Render prior turns and finish at the next assistant prefix."""

    parts = [render_system_prefix(template)]
    for message in messages:
        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not content:
            continue
        if role == "system":
            continue
        if role == "user":
            parts.append(render_user_turn(content, template))
        elif role == "assistant":
            parts.append(
                template["assistant_prefix"]
                + content
                + template["turn_separator"]
            )
        else:
            raise ValueError(f"Unsupported chat role: {role}")
    parts.append(template["assistant_prefix"])
    return "".join(parts)
