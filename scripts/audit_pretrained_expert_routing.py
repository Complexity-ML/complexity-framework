"""Audit real TR-Hash route loads and routed-branch scale on probe tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from complexity.tokenizer import Tokenizer
from scripts.viz_pretrained_expert_tsne_3d import (
    DEFAULT_ARTIFACT_LABEL,
    DEFAULT_CHECKPOINT,
    DEFAULT_LAYERS,
    DEFAULT_PROBE,
    DEFAULT_SOURCE_TOKEN_EXPOSURE,
    audit_full_probe_routing,
    collect_contexts,
    encode_probe,
    load_tr_hash_model,
    parse_layers,
    read_probe_records,
    resolve_device,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--probe", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=parse_layers, default=DEFAULT_LAYERS)
    parser.add_argument("--probe-records", type=int, default=128)
    parser.add_argument("--sequences", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--audit-batch-size", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-label", default="TR-Hash 500M pretrained")
    parser.add_argument("--artifact-label", default=DEFAULT_ARTIFACT_LABEL)
    parser.add_argument(
        "--source-token-exposure",
        type=int,
        default=DEFAULT_SOURCE_TOKEN_EXPOSURE,
    )
    parser.add_argument("--sft-tokens", type=int, default=0)
    args = parser.parse_args()

    checkpoint = args.checkpoint.resolve()
    probe = args.probe.resolve()
    output = args.output.resolve()
    device = resolve_device(args.device)

    records = read_probe_records(probe, args.probe_records)
    tokenizer = Tokenizer.load(str(checkpoint))
    encoded = encode_probe(tokenizer, records)
    model = load_tr_hash_model(checkpoint, device)
    if max(args.layers) >= len(model.layers):
        raise ValueError(
            f"requested layer {max(args.layers)}, model has {len(model.layers)} layers"
        )

    contexts = collect_contexts(
        model,
        encoded,
        args.layers,
        sequence_length=args.sequence_length,
        sequences=args.sequences,
        device=device,
    )
    layer_audits = {}
    for layer_idx in args.layers:
        hidden, residual, ids, _sequence_ids, _positions = contexts[layer_idx]
        audit = audit_full_probe_routing(
            model.layers[layer_idx].mlp,
            hidden,
            residual,
            ids,
            device,
            batch_size=args.audit_batch_size,
        )
        layer_audits[str(layer_idx)] = audit
        ratio = audit["routed_to_residual_norm_ratio"]
        print(
            f"layer={layer_idx} tokens={audit['tokens']} "
            f"routes={audit['route_counts']} repeated={audit['tokens_with_repeated_expert_across_routes']} "
            f"routed/residual p50={ratio['p50']:.6f} p99={ratio['p99']:.6f} max={ratio['max']:.6f}"
        )

    report = {
        "title": f"{args.model_label} full-probe routing and norm audit",
        "scope": (
            "Unsampled route-load and routed-branch scale diagnostics on the "
            "collected probe tokens; not a downstream-quality claim."
        ),
        "checkpoint": {
            "artifact": args.artifact_label,
            "weights_sha256": sha256_file(checkpoint / "model.safetensors"),
            "config_sha256": sha256_file(checkpoint / "config.json"),
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "source_token_exposure": args.source_token_exposure,
            "sft_tokens": args.sft_tokens,
        },
        "probe": {
            "name": "PIQA validation",
            "source": "ybisk/piqa",
            "file": probe.name,
            "sha256": sha256_file(probe),
            "records_used": len(records),
            "encoded_tokens_available": len(encoded),
            "chat_template": False,
        },
        "method": {
            "layers_zero_indexed": list(args.layers),
            "sequences": args.sequences,
            "sequence_length": args.sequence_length,
            "tokens_per_layer": args.sequences * args.sequence_length,
            "sampling": "all_collected_probe_tokens",
            "shared_output_included": False,
            "norm_ratio_denominator": "pre-MLP residual stream",
        },
        "layers": layer_audits,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"report={output}")


if __name__ == "__main__":
    main()
