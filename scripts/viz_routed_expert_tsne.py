"""t-SNE of routed-only Token-Routed expert activations.

This diagnostic intentionally removes the shared expert from the visualized
signal. Each point is the mean routed output for one expert, one route
(primary/secondary), one layer, and one sampled batch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from complexity.config import ModelConfig
from complexity.core.mlp.token_routed import TokenRoutedMLP
from complexity.models import ComplexityModel
from complexity.training.o200k import (
    LocalTextDataset,
    build_ctx_expert_mapping,
    text_context_sig_top_n,
    text_token_frequencies,
    tokenizer_token_classes,
)
from complexity.training.o200k.data import split_tokens
from complexity.tokenizer import Tokenizer


def _load_checkpoint(path: Path) -> dict:
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict) or "model" not in data or "config" not in data:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    return data


def _build_model(data: dict, device: torch.device) -> ComplexityModel:
    args = data.get("args", {})
    cfg = dict(data["config"])
    cfg["token_frequencies"] = text_token_frequencies(
        args["text_file"],
        args["tokenizer"],
        cfg["vocab_size"],
    )
    if cfg.get("routing_strategy") == "zipf_context_sig":
        class_table = tokenizer_token_classes(args["tokenizer"], cfg["vocab_size"])
        ctx_keys, ctx_counts = text_context_sig_top_n(
            args["text_file"],
            args["tokenizer"],
            cfg["vocab_size"],
            top_n=int(args.get("context_top_n", 1000)),
            window=int(args.get("context_window", cfg.get("ctx_window", 4))),
            num_buckets=int(args.get("context_buckets", cfg.get("ctx_num_buckets", 32))),
            token_class_table=class_table,
        )
        cfg["ctx_sig_keys"] = ctx_keys
        cfg["ctx_sig_experts"] = build_ctx_expert_mapping(
            args.get("ctx_expert_mapping", "balance"),
            ctx_keys,
            ctx_counts,
            cfg["num_experts"],
            text_file=args["text_file"],
            tokenizer_path=args["tokenizer"],
            vocab_size=cfg["vocab_size"],
            window=int(args.get("context_window", cfg.get("ctx_window", 4))),
            slack=float(args.get("ctx_cluster_slack", 1.05)),
        )
        cfg["token_class_table"] = class_table
        cfg["ctx_window"] = int(args.get("context_window", cfg.get("ctx_window", 4)))
        cfg["ctx_num_buckets"] = int(args.get("context_buckets", cfg.get("ctx_num_buckets", 32)))

    config = ModelConfig(**cfg)
    model = ComplexityModel(config)
    missing, unexpected = model.load_state_dict(data["model"], strict=False)
    ignored_missing = {
        name
        for name in missing
        if any(
            suffix in name
            for suffix in (
                "ctx_sig_keys",
                "ctx_sig_experts",
                "token_class_table",
                "ctx_class_weights",
                "expert_counts",
                "last_shared_rms",
                "last_routed_rms",
            )
        )
    }
    real_missing = sorted(set(missing) - ignored_missing)
    if real_missing or unexpected:
        raise RuntimeError(
            f"State dict mismatch. missing={real_missing[:8]} unexpected={unexpected[:8]}"
        )
    model.eval().to(device)
    return model


def _route_ids(module: TokenRoutedMLP, token_ids: torch.Tensor) -> torch.Tensor:
    token_ids = token_ids.clamp(0, module.vocab_size - 1)
    expert_ids = module.token_to_expert[token_ids]
    routes = module.topk_token_to_expert[:, token_ids]
    if getattr(module, "has_ctx_sig_routing", False) and module.top_k >= 2:
        batch, seq = token_ids.shape
        window = module.ctx_window
        class_table = module.token_class_table.to(token_ids.device)
        class_weights = module.ctx_class_weights.to(token_ids.device)
        class_seq = class_table[token_ids]
        padded = torch.cat(
            [torch.zeros((batch, window), dtype=class_seq.dtype, device=class_seq.device), class_seq],
            dim=1,
        )
        windows = padded.unfold(dimension=1, size=window, step=1)[:, :seq, :]
        sig = (windows.long() * class_weights).sum(-1) % module.ctx_num_buckets
        keys = sig.long() * module.vocab_size + token_ids.long()
        flat_keys = keys.reshape(-1)
        ctx_keys = module.ctx_sig_keys.to(flat_keys.device)
        ctx_experts = module.ctx_sig_experts.to(flat_keys.device)
        idx = torch.searchsorted(ctx_keys, flat_keys)
        idx = idx.clamp(max=ctx_keys.numel() - 1)
        found = ctx_keys[idx] == flat_keys
        ctx_exp = ctx_experts[idx].to(expert_ids.dtype)
        primary = expert_ids.reshape(-1)
        secondary = routes[1].reshape(-1)
        routes[1] = torch.where(found & (ctx_exp != primary), ctx_exp, secondary).view(batch, seq)
    return routes


class RoutedOnlyCollector:
    def __init__(self, model: ComplexityModel, max_batches: int):
        self.model = model
        self.max_batches = max_batches
        self.batch_idx = 0
        self.records: list[tuple[int, int, str, int, np.ndarray]] = []
        self.handles = []

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, TokenRoutedMLP):
                self.handles.append(layer.mlp.register_forward_pre_hook(self._hook(layer_idx), with_kwargs=True))
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()

    def _hook(self, layer_idx: int):
        def hook(module: TokenRoutedMLP, args, kwargs):
            hidden = args[0].detach()
            token_ids = kwargs.get("token_ids")
            if token_ids is None:
                return
            routes = _route_ids(module, token_ids)
            flat_x = hidden.reshape(-1, hidden.shape[-1])
            gate_w = module.gate_proj_w
            up_w = module.up_proj_w
            down_w = module.down_proj_w
            route_names = ["primary"] + [f"secondary_{idx}" for idx in range(1, module.top_k)]

            for route_idx in range(module.top_k):
                expert_ids = routes[route_idx].reshape(-1)
                routed = module._dispatch_once(
                    flat_x,
                    expert_ids,
                    gate_w,
                    up_w,
                    down_w,
                    False,
                    hidden.shape[-1],
                ).view_as(hidden)
                for expert_id in range(module.num_experts):
                    mask = routes[route_idx] == expert_id
                    if not bool(mask.any().item()):
                        continue
                    vec = routed[mask].mean(dim=0).detach().cpu().float().numpy()
                    self.records.append((layer_idx, expert_id, route_names[route_idx], self.batch_idx, vec))

        return hook


def _collect_batches(args: dict, seq_len: int, batch_size: int, batches: int) -> DataLoader:
    tokenizer = Tokenizer.load(args["tokenizer"])
    text = Path(args["text_file"]).read_text(encoding="utf-8")
    tokens = tokenizer.encode(text)
    train_tokens, _ = split_tokens(tokens, float(args.get("eval_ratio", 0.05)))
    dataset = LocalTextDataset(train_tokens, seq_len, int(args.get("seed", 42)) + 777)
    return DataLoader(dataset, batch_size=batch_size), batches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="figures/routed_only_tsne")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = _load_checkpoint(checkpoint)
    model = _build_model(data, device)
    loader, max_batches = _collect_batches(data["args"], args.seq_len, args.batch_size, args.batches)

    with torch.no_grad(), RoutedOnlyCollector(model, max_batches) as collector:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= max_batches:
                break
            collector.batch_idx = batch_idx
            model(batch["input_ids"].to(device), return_logits=False)

    rows = collector.records
    vectors = np.stack([row[4] for row in rows], axis=0)
    pca_dim = min(32, vectors.shape[0] - 1, vectors.shape[1])
    x_pca = PCA(n_components=pca_dim, random_state=42).fit_transform(vectors)
    perplexity = min(float(args.perplexity), max(2.0, x_pca.shape[0] - 1))
    x_2d = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000).fit_transform(x_pca)

    layer_ids = np.array([row[0] for row in rows])
    expert_ids = np.array([row[1] for row in rows])
    routes = np.array([row[2] for row in rows])
    colors = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for expert_id in sorted(set(expert_ids.tolist())):
        mask = expert_ids == expert_id
        axes[0].scatter(x_2d[mask, 0], x_2d[mask, 1], s=28, alpha=0.72, color=colors[expert_id % len(colors)], label=f"E{expert_id}")
    axes[0].set_title("Routed-only by expert")
    axes[0].legend(frameon=False, ncol=2)

    for route in sorted(set(routes.tolist())):
        mask = routes == route
        axes[1].scatter(x_2d[mask, 0], x_2d[mask, 1], s=28, alpha=0.72, label=route)
    axes[1].set_title("Primary vs secondary")
    axes[1].legend(frameon=False)

    scatter = axes[2].scatter(x_2d[:, 0], x_2d[:, 1], c=layer_ids, cmap="viridis", s=28, alpha=0.72)
    axes[2].set_title("By layer")
    fig.colorbar(scatter, ax=axes[2], label="layer")
    for ax in axes:
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(alpha=0.15)
    fig.tight_layout()

    figure_path = output_dir / "routed_only_expert_tsne_2d.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = output_dir / "routed_only_expert_tsne_points.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "layer", "expert", "route", "batch"])
        for point, row in zip(x_2d, rows):
            writer.writerow([point[0], point[1], row[0], row[1], row[2], row[3]])

    summary = {
        "records": len(rows),
        "checkpoint": str(checkpoint),
        "figure": str(figure_path),
        "csv": str(csv_path),
        "perplexity": perplexity,
        "pca_dim": pca_dim,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
