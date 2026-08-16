"""Create a separate, function-preserving 4→8 expert TR-Hash checkpoint."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from safetensors.torch import save_file

from complexity.utils.expert_expansion import (
    convert_checkpoint_dir_to_expanded_experts,
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
    parser.add_argument("--target-experts", type=int, default=8, choices=(8,))
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if source == output:
        raise ValueError("output must differ from source; the base model is immutable")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}")

    model = convert_checkpoint_dir_to_expanded_experts(
        source,
        target_num_experts=args.target_experts,
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
    print(f"function-preserving 8-expert checkpoint written to {output}")


if __name__ == "__main__":
    main()
