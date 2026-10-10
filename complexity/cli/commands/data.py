"""Reusable corpus preparation and zero-copy replay commands."""

from __future__ import annotations

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import typer

from complexity._optional import require_optional

from ..inspection import load_mapping

data = typer.Typer(name="data", help="Prepare and inspect token corpora", no_args_is_help=True)


@data.command("sft")
def compile_sft(
    dataset: Path = typer.Argument(..., help="Local train.jsonl/eval.jsonl directory"),
    output: Path = typer.Argument(..., help="New SFT token-shard directory"),
    tokenizer: Path = typer.Option(..., "--tokenizer", help="Local native tokenizer directory"),
    assistant_supervision: str = typer.Option(
        ...,
        "--assistant-supervision",
        help="Explicit final_assistant_only or all_assistant_turns",
    ),
    sequence_length: int = typer.Option(2048, "--sequence-length", min=1),
):
    """Compile local SFT records with masks, EOS and source checksums."""
    from complexity.training import compile_sft_dataset

    try:
        with redirect_stdout(sys.stderr):
            report = compile_sft_dataset(
                dataset,
                tokenizer,
                output,
                sequence_length=sequence_length,
                assistant_supervision=assistant_supervision,
            )
    except (OSError, TypeError, ValueError, KeyError, ImportError) as exc:
        _fail(exc)
    typer.echo(json.dumps(report, indent=2))


def _keys(config: dict, *, required: set[str], optional: set[str] | None = None) -> None:
    missing = required - set(config)
    unknown = set(config) - required - (optional or set())
    if missing or unknown:
        raise ValueError(
            f"configuration fields: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _fail(exc: Exception) -> None:
    typer.echo(str(exc), err=True)
    raise typer.Exit(1) from exc


def _sources(config: dict):
    from complexity.training import TextCorpusSource

    if not isinstance(config["sources"], list):
        raise ValueError("sources must be a list of corpus descriptions")
    sources = tuple(TextCorpusSource(**source) for source in config["sources"])
    for source in sources:
        if not isinstance(source.name, str) or not isinstance(source.text_field, str):
            raise ValueError("source name and text_field must be strings")
    return sources


@data.command("inspect")
def inspect_corpus(path: Path = typer.Argument(..., help="Local corpus-mixture directory")):
    """Validate manifests and print source quotas without starting training."""
    from complexity.training import PretokenizedCorpusMixtureDataset

    try:
        dataset = PretokenizedCorpusMixtureDataset(path)
        report = {
            "path": str(path.resolve()),
            "seq_len": dataset.seq_len,
            "unique_tokens": dataset.unique_tokens,
            "trained_tokens": dataset.trained_tokens,
            "sources": [
                {
                    "name": source.name,
                    "weight": source.weight,
                    "rows": dataset._rows_by_source[source.name],
                }
                for source in dataset.sources
            ],
            "validation": "manifests and file sizes; shard hashes are checked by the training reader",
        }
    except (OSError, TypeError, ValueError, KeyError) as exc:
        _fail(exc)
    typer.echo(json.dumps(report, indent=2))


@data.command("prepare")
def prepare_corpus(
    config: Path = typer.Argument(..., help="JSON/YAML source mixture and tokenization layout"),
    output: Path = typer.Argument(..., help="New local token-mixture directory"),
    tokenizer: Path | None = typer.Option(None, "--tokenizer", help="Local tokenizer directory"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate quotas without loading data or tokenizer"
    ),
):
    """Tokenize explicit sources into the framework's corpus-mixture format."""
    from complexity.training.corpus_preparation import prepare_corpus_mixture, resolve_layout

    try:
        values = load_mapping(config)
        _keys(
            values,
            required={
                "sources",
                "target_tokens",
                "seq_len",
                "global_batch_sequences",
                "rows_per_shard",
            },
        )
        sources = _sources(values)
        from complexity.training.corpus_preparation import _positive_integer

        _positive_integer(values["rows_per_shard"], "rows_per_shard")
        rows, actual_tokens, quotas = resolve_layout(
            target_tokens=values["target_tokens"],
            seq_len=values["seq_len"],
            global_batch_sequences=values["global_batch_sequences"],
            sources=sources,
        )
        if dry_run:
            typer.echo(
                json.dumps(
                    {"rows": rows, "actual_tokens": actual_tokens, "rows_by_source": quotas},
                    indent=2,
                )
            )
            return
        if tokenizer is None or not tokenizer.is_dir():
            raise ValueError("--tokenizer must name a local tokenizer directory")
        require_optional("transformers", "hf")
        from transformers import PreTrainedTokenizerFast

        loaded_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(tokenizer), local_files_only=True
        )
        if len(loaded_tokenizer) > 65536:
            raise ValueError("uint16 corpus storage supports at most 65,536 tokens")
        manifest = prepare_corpus_mixture(
            output_root=output,
            tokenizer=loaded_tokenizer,
            sources=sources,
            **{
                name: values[name]
                for name in ("target_tokens", "seq_len", "global_batch_sequences", "rows_per_shard")
            },
        )
    except (OSError, TypeError, ValueError, KeyError, ImportError) as exc:
        _fail(exc)
    typer.echo(str(manifest))


@data.command("replay")
def plan_replay(
    path: Path = typer.Argument(..., help="Local corpus-mixture directory"),
    config: Path = typer.Argument(
        ..., help="Explicit source budgets/passes/alignment JSON or YAML"
    ),
    output: Path = typer.Argument(..., help="New replay-plan JSON file"),
):
    """Select or correct shard exposure without copying or modifying token data."""
    from complexity.training import (
        PretokenizedCorpusMixtureDataset,
        build_corrective_replay_plan,
        build_replay_plan,
    )

    try:
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        values = load_mapping(config)
        _keys(
            values,
            required={"unique_token_budgets", "replay_passes", "row_alignment"},
            optional={"quality_scores", "already_double_exposed_shards"},
        )
        dataset = PretokenizedCorpusMixtureDataset(path)
        arguments = {
            name: values[name]
            for name in ("unique_token_budgets", "replay_passes", "row_alignment")
        }
        if "already_double_exposed_shards" in values:
            if "quality_scores" in values:
                raise ValueError(
                    "exposure correction requires manifest-order selection; omit quality_scores"
                )
            plan = build_corrective_replay_plan(
                dataset,
                **arguments,
                already_double_exposed_shards=values["already_double_exposed_shards"],
            )
        else:
            plan = build_replay_plan(
                dataset, **arguments, quality_scores=values.get("quality_scores")
            )
        # Use the reader's complete replay validation before publishing a plan.
        PretokenizedCorpusMixtureDataset(path, replay_plan=plan)
        plan["dataset"] = str(path.resolve())
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(plan, indent=2) + "\n")
    except (OSError, TypeError, ValueError, KeyError) as exc:
        _fail(exc)
    typer.echo(
        json.dumps(
            {
                "plan": str(output),
                "unique_tokens": plan["unique_tokens"],
                "trained_tokens": plan["trained_tokens"],
            },
            indent=2,
        )
    )
