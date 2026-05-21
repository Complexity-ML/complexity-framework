"""Composable file-backed JSONL builders for staged SFT replay mixes."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from .config import DatasetMix, SourceConfig


def dump_record(handle, prompt: str, completion: str) -> None:
    handle.write(
        json.dumps(
            {"prompt": prompt.rstrip() + "\n", "completion": completion.strip()},
            ensure_ascii=False,
        )
        + "\n"
    )


def chat_prompt(user: str) -> str:
    return f"User:\n{user}\n\nAssistant:\n"


class StackSFTDatasetBuilder:
    """Build stage JSONL files by weighted sampling from JSONL/parquet sources."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._cache: dict[tuple, list[dict[str, str]]] = {}

    def build(self, mix: DatasetMix, out: str | Path) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        rng = random.Random(mix.seed + self.seed)
        counts = self._counts(mix)
        with out.open("w", encoding="utf-8") as handle:
            for source, count in counts:
                self._write_file_source(handle, source, count, rng)
        # JSON strings from web corpora can contain Unicode line separators
        # that str.splitlines() treats as record boundaries. JSONL records are
        # delimited only by literal LF bytes, so split on "\n" exactly.
        lines = [line for line in out.read_text(encoding="utf-8").split("\n") if line]
        rng.shuffle(lines)
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return out

    def _counts(self, mix: DatasetMix) -> list[tuple[SourceConfig, int]]:
        total_weight = sum(max(0.0, source.weight) for source in mix.sources)
        if total_weight <= 0:
            raise ValueError("dataset mix weights must sum to a positive value")
        counts = [
            (source, int(mix.records * max(0.0, source.weight) / total_weight))
            for source in mix.sources
        ]
        counts[0] = (counts[0][0], counts[0][1] + mix.records - sum(count for _, count in counts))
        return counts

    def _write_file_source(self, handle, source: SourceConfig, count: int, rng: random.Random) -> None:
        if count <= 0:
            return
        if not source.path:
            raise ValueError(
                f"stack SFT source {source.name!r} must define a JSONL/parquet path; "
                "hardcoded synthetic sources are not supported"
            )
        records = self._load_file_records(source)
        if not records:
            raise ValueError(f"file source {source.name!r} produced no usable records")
        for _ in range(count):
            record = rng.choice(records)
            dump_record(handle, record["prompt"], record["completion"])

    def _load_file_records(self, source: SourceConfig) -> list[dict[str, str]]:
        key = (
            source.path,
            source.format,
            source.prompt_field,
            source.completion_field,
            source.messages_field,
            source.instruction_field,
            source.input_field,
            source.output_field,
            source.max_records,
        )
        if key in self._cache:
            return self._cache[key]

        path = Path(source.path or "")
        fmt = (source.format or path.suffix.lstrip(".") or "jsonl").lower()
        if fmt in {"jsonl", "json"}:
            raw_records = self._load_jsonl(path)
        elif fmt == "parquet":
            raw_records = self._load_parquet(path)
        else:
            raise ValueError(f"unsupported stack SFT source format: {fmt}")

        records = []
        for row in raw_records:
            formatted = self._format_record(row, source)
            if formatted is not None:
                records.append(formatted)
            if source.max_records is not None and len(records) >= int(source.max_records):
                break
        self._cache[key] = records
        return records

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
                if isinstance(row, dict):
                    rows.append(row)
        return rows

    def _load_parquet(self, path: Path) -> list[dict[str, Any]]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError("parquet stack SFT sources require pyarrow") from exc
        table = pq.read_table(path)
        return table.to_pylist()

    def _format_record(self, row: dict[str, Any], source: SourceConfig) -> dict[str, str] | None:
        if source.messages_field in row:
            return self._format_messages(row.get(source.messages_field))
        if source.prompt_field in row and source.completion_field in row:
            prompt = str(row.get(source.prompt_field) or "").strip()
            completion = str(row.get(source.completion_field) or "").strip()
            if prompt and completion:
                return {"prompt": prompt, "completion": completion}
        if source.instruction_field in row or source.output_field in row:
            instruction = str(row.get(source.instruction_field) or "").strip()
            extra_input = str(row.get(source.input_field) or "").strip()
            completion = str(row.get(source.output_field) or row.get("response") or "").strip()
            if instruction and completion:
                user = instruction if not extra_input else f"{instruction}\n\n{extra_input}"
                return {"prompt": chat_prompt(user), "completion": completion}
        return None

    def _format_messages(self, messages: Any) -> dict[str, str] | None:
        if not isinstance(messages, list) or not messages:
            return None
        assistant_idx = None
        for idx in range(len(messages) - 1, -1, -1):
            if isinstance(messages[idx], dict) and messages[idx].get("role") == "assistant":
                assistant_idx = idx
                break
        if assistant_idx is None:
            return None
        prompt_parts: list[str] = []
        for msg in messages[:assistant_idx]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).strip().title()
            content = str(msg.get("content", "")).strip()
            if content:
                prompt_parts.append(f"{role}:\n{content}")
        completion = str(messages[assistant_idx].get("content", "")).strip()
        if not prompt_parts or not completion:
            return None
        return {"prompt": "\n\n".join(prompt_parts) + "\n\nAssistant:\n", "completion": completion}
