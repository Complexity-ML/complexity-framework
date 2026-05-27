"""Expert-specialization diagnostics for Token-Routed MoE layers.

Unlike ``viz_routed_expert_tsne.py`` (which plots per-expert *mean* outputs,
i.e. centroids), this script keeps per-token signal and answers the two
questions that actually define specialization:

  1. Do experts compute *different functions*?  -> linear CKA between expert
     outputs on the *same* token inputs (per layer).  This is the metric that
     directly reflects the expert-diversity regulariser.
  2. Does routing partition the input space coherently? -> per-token t-SNE of
     routed outputs colored by expert, with a silhouette score computed in the
     original high-dimensional space (never on the 2-D coordinates).

It also reports the weight-cosine (Gram) matrix the regulariser optimises and
the static ``expert x token-class`` routing map, plus per-layer scalar metrics
in ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader

from complexity.core.mlp.token_routed import TokenRoutedMLP
from complexity.models import ComplexityModel
from complexity.training.o200k import tokenizer_token_classes
from complexity.training.o200k.data import split_tokens
from complexity.tokenizer import Tokenizer

# Reuse checkpoint loading / model building / deterministic routing from the
# sibling script so the two diagnostics never drift apart.
from viz_routed_expert_tsne import _build_model, _load_checkpoint, _route_ids, LocalTextDataset

CLASS_NAMES = ["other", "space", "digit", "word", "alnum", "punct", "unicode", "misc"]
EXPERT_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#17becf", "#8c564b", "#e377c2"]


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Centered linear CKA between two (n_tokens, d) activation matrices."""

    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denom = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / denom) if denom > 0 else 0.0


def routing_entropy(expert_ids: np.ndarray, num_experts: int) -> float:
    """Normalised entropy of the expert load (1.0 = perfectly balanced)."""

    counts = np.bincount(expert_ids, minlength=num_experts).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum() / np.log(num_experts)) if num_experts > 1 else 0.0


class PerTokenCollector:
    """Capture per-token MoE inputs + deterministic routing, per layer."""

    def __init__(self, model: ComplexityModel, max_tokens: int, seed: int = 0):
        self.model = model
        self.max_tokens = max_tokens
        self.rng = np.random.default_rng(seed)
        self.modules: dict[int, TokenRoutedMLP] = {}
        self.hidden: dict[int, list[np.ndarray]] = {}
        self.experts: dict[int, list[np.ndarray]] = {}
        self.tokens: dict[int, list[np.ndarray]] = {}
        self.handles = []

    def __enter__(self):
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer.mlp, TokenRoutedMLP):
                self.modules[layer_idx] = layer.mlp
                self.hidden[layer_idx] = []
                self.experts[layer_idx] = []
                self.tokens[layer_idx] = []
                self.handles.append(
                    layer.mlp.register_forward_pre_hook(self._hook(layer_idx), with_kwargs=True)
                )
        return self

    def __exit__(self, *_exc):
        for handle in self.handles:
            handle.remove()

    def _count(self, layer_idx: int) -> int:
        return sum(part.shape[0] for part in self.hidden[layer_idx])

    def _hook(self, layer_idx: int):
        def hook(module: TokenRoutedMLP, args, kwargs):
            if self._count(layer_idx) >= self.max_tokens:
                return
            token_ids = kwargs.get("token_ids")
            if token_ids is None:
                return
            hidden = args[0].detach()
            flat_x = hidden.reshape(-1, hidden.shape[-1])
            if getattr(module, "has_lsh_routing", False):
                # semantic routing: primary expert from the LSH hash of h
                planes = module.lsh_planes.to(flat_x.dtype)
                bit_vals = module.lsh_bit_values.to(flat_x.device)
                proj = flat_x @ planes.t()
                if getattr(module.config, "lsh_threshold_mode", "batch_median") == "zero":
                    thresh = torch.zeros(proj.shape[-1], dtype=proj.dtype, device=proj.device)
                else:
                    thresh = proj.median(dim=0).values  # per-plane median (matches forward)
                bucket = ((proj > thresh).long() * bit_vals).sum(-1)
                primary = module.lsh_bucket_to_expert[bucket]
            else:
                primary = _route_ids(module, token_ids)[0].reshape(-1)
            flat_tokens = token_ids.reshape(-1)

            budget = self.max_tokens - self._count(layer_idx)
            take = min(budget, flat_x.shape[0])
            idx = self.rng.choice(flat_x.shape[0], size=take, replace=False)
            idx = torch.from_numpy(idx).to(flat_x.device)
            self.hidden[layer_idx].append(flat_x[idx].cpu().float().numpy())
            self.experts[layer_idx].append(primary[idx].cpu().numpy())
            self.tokens[layer_idx].append(flat_tokens[idx].cpu().numpy())

        return hook


def expert_outputs(module: TokenRoutedMLP, x: np.ndarray) -> np.ndarray:
    """Run every expert over the same inputs -> (num_experts, n_tokens, H)."""

    device = module.gate_proj_w.device
    flat_x = torch.from_numpy(x).to(device=device, dtype=module.gate_proj_w.dtype)
    h = flat_x.shape[-1]
    outs = []
    for expert_id in range(module.num_experts):
        ids = torch.full((flat_x.shape[0],), expert_id, dtype=torch.long, device=device)
        out = module._dispatch_once(
            flat_x, ids, module.gate_proj_w, module.up_proj_w, module.down_proj_w, False, h
        )
        outs.append(out.detach().cpu().float().numpy())
    return np.stack(outs, axis=0)


def weight_cosine(module: TokenRoutedMLP) -> np.ndarray:
    """Cosine matrix of flattened per-expert down-projection weights."""

    w = module.down_proj_w.detach().cpu().float()
    flat = w.reshape(module.num_experts, -1).numpy()
    flat /= np.linalg.norm(flat, axis=1, keepdims=True) + 1e-12
    return flat @ flat.T


def _heatmap_grid(matrices: dict[int, np.ndarray], title: str, path: Path, vmin=0.0, vmax=1.0,
                  cmap="viridis", annotate=True):
    layers = sorted(matrices)
    cols = min(4, len(layers))
    rows = (len(layers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.2 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    last = None
    for pos, layer_idx in enumerate(layers):
        ax = axes[pos // cols][pos % cols]
        ax.axis("on")
        m = matrices[layer_idx]
        last = ax.imshow(m, vmin=vmin, vmax=vmax, cmap=cmap)
        ax.set_title(f"layer {layer_idx}", fontsize=9)
        ax.set_xticks(range(m.shape[1]))
        ax.set_yticks(range(m.shape[0]))
        if annotate and m.shape[0] <= 8:
            for i in range(m.shape[0]):
                for j in range(m.shape[1]):
                    ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center",
                            color="white" if m[i, j] < (vmin + vmax) / 2 else "black", fontsize=7)
    fig.suptitle(title)
    if last is not None:
        fig.colorbar(last, ax=axes.ravel().tolist(), shrink=0.6)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="figures/expert_specialization")
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=1500, help="per-layer token sample cap")
    parser.add_argument("--cka-tokens", type=int, default=512, help="tokens used for the CKA pass")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    data = _load_checkpoint(Path(args.checkpoint))
    model = _build_model(data, device)

    run_args = data["args"]
    tokenizer = Tokenizer.load(run_args["tokenizer"])
    tokens = tokenizer.encode(Path(run_args["text_file"]).read_text(encoding="utf-8"))
    train_tokens, _ = split_tokens(tokens, float(run_args.get("eval_ratio", 0.05)))
    dataset = LocalTextDataset(train_tokens, args.seq_len, int(run_args.get("seed", 42)) + 777)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    vocab_size = int(data["config"]["vocab_size"])
    token_class = tokenizer_token_classes(run_args["tokenizer"], vocab_size).numpy()

    with torch.no_grad(), PerTokenCollector(model, args.max_tokens) as collector:
        for batch_idx, batch in enumerate(loader):
            if batch_idx >= args.batches:
                break
            model(batch["input_ids"].to(device), return_logits=False)

        cka_mats: dict[int, np.ndarray] = {}
        wcos_mats: dict[int, np.ndarray] = {}
        class_mats: dict[int, np.ndarray] = {}
        tsne_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        metrics: dict[str, dict] = {}

        for layer_idx, module in collector.modules.items():
            if not collector.hidden[layer_idx]:
                continue
            x = np.concatenate(collector.hidden[layer_idx], axis=0)
            primary = np.concatenate(collector.experts[layer_idx], axis=0)
            tok = np.concatenate(collector.tokens[layer_idx], axis=0)
            n_experts = module.num_experts

            # (1) functional diversity: CKA between experts on identical inputs.
            sample = x[: args.cka_tokens]
            outs = expert_outputs(module, sample)  # (E, N, H)
            cka = np.eye(n_experts)
            for i in range(n_experts):
                for j in range(i + 1, n_experts):
                    cka[i, j] = cka[j, i] = linear_cka(outs[i], outs[j])
            cka_mats[layer_idx] = cka

            # (2) weight orthogonality the regulariser targets.
            wcos_mats[layer_idx] = np.abs(weight_cosine(module))

            # (3) static routing map: P(expert | token class).
            cls = token_class[np.clip(tok, 0, vocab_size - 1)]
            cm = np.zeros((n_experts, len(CLASS_NAMES)))
            np.add.at(cm, (primary, cls), 1)
            cm /= cm.sum(axis=0, keepdims=True) + 1e-9
            class_mats[layer_idx] = cm

            # (4) per-token routed output, for t-SNE colored by expert.
            routed = outs[primary[: args.cka_tokens], np.arange(min(args.cka_tokens, len(primary)))]
            tsne_data[layer_idx] = (routed, primary[: routed.shape[0]])

            off = cka[~np.eye(n_experts, dtype=bool)]
            woff = wcos_mats[layer_idx][~np.eye(n_experts, dtype=bool)]
            sil = None
            labels = primary[: routed.shape[0]]
            if len(set(labels.tolist())) > 1 and np.bincount(labels).min() > 1:
                sil = float(silhouette_score(routed, labels))
            metrics[f"layer_{layer_idx}"] = {
                "tokens": int(x.shape[0]),
                "mean_offdiag_cka": float(off.mean()),
                "max_offdiag_cka": float(off.max()),
                "mean_offdiag_weight_cos": float(woff.mean()),
                "routing_entropy": routing_entropy(primary, n_experts),
                "output_silhouette": sil,
            }

    _heatmap_grid(cka_mats, "Expert output CKA (low off-diag = distinct functions)",
                  output_dir / "expert_cka_by_layer.png", cmap="magma")
    _heatmap_grid(wcos_mats, "Expert weight |cosine| (regulariser target)",
                  output_dir / "expert_weight_cosine_by_layer.png", cmap="magma")

    # expert x class heatmaps share their own grid (non-square).
    layers = sorted(class_mats)
    cols = min(4, len(layers))
    rows = (len(layers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.0 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    im = None
    for pos, layer_idx in enumerate(layers):
        ax = axes[pos // cols][pos % cols]
        ax.axis("on")
        im = ax.imshow(class_mats[layer_idx], vmin=0, vmax=1, aspect="auto", cmap="viridis")
        ax.set_title(f"layer {layer_idx}", fontsize=9)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel("expert")
    fig.suptitle("P(expert | token class) — static routing map")
    fig.subplots_adjust(hspace=0.75)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig.savefig(output_dir / "expert_class_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # per-token routed-output t-SNE, one panel per layer, colored by expert.
    layers = sorted(tsne_data)
    cols = min(4, len(layers))
    rows = (len(layers) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.6 * cols, 3.4 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for pos, layer_idx in enumerate(layers):
        ax = axes[pos // cols][pos % cols]
        ax.axis("on")
        vecs, labels = tsne_data[layer_idx]
        pca_dim = min(32, vecs.shape[0] - 1, vecs.shape[1])
        reduced = PCA(n_components=pca_dim, random_state=42).fit_transform(vecs)
        perp = min(float(args.perplexity), max(2.0, reduced.shape[0] - 1))
        emb = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca").fit_transform(reduced)
        for expert_id in sorted(set(labels.tolist())):
            m = labels == expert_id
            ax.scatter(emb[m, 0], emb[m, 1], s=6, alpha=0.6,
                       color=EXPERT_COLORS[expert_id % len(EXPERT_COLORS)], label=f"E{expert_id}")
        sil = metrics[f"layer_{layer_idx}"]["output_silhouette"]
        ax.set_title(f"layer {layer_idx}  (sil={sil:.3f})" if sil is not None else f"layer {layer_idx}",
                     fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    handles, labels_ = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper right", frameon=False, ncol=2)
    fig.suptitle("Routed-output t-SNE per layer (silhouette computed in original dim)")
    fig.savefig(output_dir / "routed_output_tsne_by_layer.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "checkpoint": str(args.checkpoint),
        "layers": metrics,
        "global": {
            "mean_offdiag_cka": float(np.mean([m["mean_offdiag_cka"] for m in metrics.values()])),
            "mean_offdiag_weight_cos": float(np.mean([m["mean_offdiag_weight_cos"] for m in metrics.values()])),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
