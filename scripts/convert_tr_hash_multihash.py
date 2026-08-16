"""Create a separate multi-hash checkpoint from a TR-Hash text model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import save_file

from complexity.utils.multi_hash_conversion import (
    convert_checkpoint_dir_to_multi_hash,
)

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "generation_config.json",
    "chat_template.json",
    "chat_template.jinja",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--route-hash-count", type=int, default=2)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("output must differ from source; the base model is immutable")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    model = convert_checkpoint_dir_to_multi_hash(
        source,
        route_hash_count=args.route_hash_count,
    )
    output.mkdir(parents=True, exist_ok=True)
    state = {
        key: value.detach().cpu().contiguous()
        for key, value in model.state_dict().items()
    }
    save_file(state, str(output / "model.safetensors"))
    model.config.save(str(output / "config.json"))
    for filename in TOKENIZER_FILES:
        candidate = source / filename
        if candidate.exists():
            shutil.copy2(candidate, output / filename)
    print(f"multi-hash checkpoint written to {output}")


if __name__ == "__main__":
    main()
