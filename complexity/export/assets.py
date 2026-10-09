"""Portable tokenizer assets and the explicit chat contract for model bundles.

These helpers write metadata only. Tensor conversion and supported model
architectures remain the responsibility of the canonical Transformers exporter.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from complexity.inference.chat_template import (
    default_chat_template,
    huggingface_chat_template,
    validate_chat_template,
)

from .transformers import TOKENIZER_FILES, build_transformers_config


def build_chat_export_config(raw: dict, chat_template: dict | None = None) -> dict:
    """Build supported Transformers metadata and declare an optional chat file.

    Older native checkpoints omitted routing_strategy and used modulo_cyclic.
    Explicit routing and all supported TR-HASH dimensions/scales are preserved.
    This metadata does not imply that a serving runtime supports the model.
    """
    config = build_transformers_config(raw)
    config.setdefault("routing_strategy", "modulo_cyclic")
    for field in ("chat_template", "chat_template_id", "chat_template_file"):
        config.pop(field, None)
    if chat_template is not None:
        template = validate_chat_template(chat_template)
        config["chat_template_id"] = template["id"]
        config["chat_template_file"] = "chat_template.jinja"
    return config


def copy_tokenizer_files(source: Path, output: Path) -> list[str]:
    """Copy known tokenizer files, leaving model weights and config untouched."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError("Tokenizer source and output must be separate directories")
    paths = sorted(
        (path for path in source.iterdir() if path.is_file() and path.name in TOKENIZER_FILES),
        key=lambda path: path.name,
    )
    if not paths:
        raise FileNotFoundError(f"No tokenizer assets found in {source}")
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, output / path.name)
    return [path.name for path in paths]


def strip_tokenizer_chat_template(output: Path) -> Path:
    """Remove embedded prompt metadata while preserving other tokenizer fields."""
    path = Path(output) / "tokenizer_config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("tokenizer_config.json must contain a JSON object")
    config.pop("chat_template", None)
    config.pop("chat_template_id", None)
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def configure_tokenizer_chat_template(output: Path, chat_template: dict | None) -> Path | None:
    """Install standalone Jinja for a chat model, or remove it for a base model."""
    template = validate_chat_template(chat_template) if chat_template is not None else None
    strip_tokenizer_chat_template(output)
    path = Path(output) / "chat_template.jinja"
    if template is None:
        path.unlink(missing_ok=True)
        return None
    path.write_text(huggingface_chat_template(template) + "\n", encoding="utf-8")
    return path


def write_chat_template(checkpoint: dict, output: Path) -> dict:
    """Preserve a checkpoint's exact chat contract as JSON and standalone Jinja."""
    template = validate_chat_template(checkpoint.get("chat_template", default_chat_template()))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "chat_template.json").write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "chat_template.jinja").write_text(
        huggingface_chat_template(template) + "\n", encoding="utf-8"
    )
    return template
