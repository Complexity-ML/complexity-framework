"""Package and optionally publish a TR-Hash detector on Hugging Face Hub."""

from __future__ import annotations

import argparse
from pathlib import Path

from complexity.generative.detection.hub import (
    COCO_CLASS_NAMES,
    VOC_CLASS_NAMES,
    export_detector_for_hub,
    upload_detector_to_hub,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="AETHORIA-AI/TR-HASH-Vision-v8-2M-COCO")
    parser.add_argument("--dataset", choices=("coco", "voc"), default="coco")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hf/tr-hash-vision-v8-2m-coco")
    )
    parser.add_argument("--training", action="store_true", help="Publish a card-only draft")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--public", action="store_true", help="Create a public repository")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.training and args.checkpoint is None:
        raise ValueError("--checkpoint is required unless --training is selected")
    class_names = COCO_CLASS_NAMES if args.dataset == "coco" else VOC_CLASS_NAMES
    export_detector_for_hub(
        args.output,
        args.repo_id,
        checkpoint=None if args.training else args.checkpoint,
        class_names=class_names,
        training=args.training,
        dataset=args.dataset,
        require_native_random_init=not args.training and args.dataset == "coco",
    )
    print(f"prepared Hugging Face folder: {args.output}")
    if args.push:
        revision = upload_detector_to_hub(
            args.output,
            args.repo_id,
            private=not args.public,
            commit_message=(
                "Add training-in-progress TR-Hash Vision card"
                if args.training
                else "Publish validated TR-Hash Vision checkpoint"
            ),
        )
        print(f"published: https://huggingface.co/{args.repo_id}/commit/{revision}")


if __name__ == "__main__":
    main()
