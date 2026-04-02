"""
3D T-SNE visualization of expert activations.

Loads model, hooks TokenRoutedMLP layers, collects per-expert activations,
then runs PCA + 3D T-SNE. Saves interactive HTML (plotly) and static PNG.

Usage:
    python scripts/tsne_experts_3d.py --checkpoint checkpoints/final.pt --device cuda
    python scripts/tsne_experts_3d.py --checkpoint checkpoints/final.pt --device cuda --num-samples 64

Complexity-ML — 2026
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path


def load_checkpoint(checkpoint_path):
    """Load model state dict."""
    path = Path(checkpoint_path)
    if path.suffix == ".pt":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = data.get("model_state_dict", data.get("model", data))
    else:
        from safetensors.torch import load_file
        from safetensors.torch import load as safetensors_load
        with open(str(path), "rb") as f:
            data = f.read()
        state_dict = safetensors_load(data)

    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("model.", "") if k.startswith("model.") else k
        cleaned[key] = v
    return cleaned


def build_model(state_dict):
    """Build model from state dict — supports complexity-framework format."""
    sys.path.insert(0, str(Path(__file__).parents[2] / "complexity-framework"))
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    q_key = next(k for k in state_dict if "q_proj.weight" in k and "layers.0." in k)
    hidden = state_dict[q_key].shape[0]
    num_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")) + 1
    vocab = state_dict["embed_tokens.weight"].shape[0]

    # Detect expert format: gate_proj_w [E, H, I] or gate_proj.weight [I, H]
    gate_3d_key = next((k for k in state_dict if "gate_proj_w" in k and "layers.0." in k), None)
    gate_lin_key = next((k for k in state_dict if "experts.0.gate_proj.weight" in k and "layers.0." in k), None)
    if gate_3d_key:
        # 3D Parameter format: gate_proj_w [E, H, I]
        num_experts_found = state_dict[gate_3d_key].shape[0]
        expert_inter = state_dict[gate_3d_key].shape[2]
        inter = expert_inter * num_experts_found
        mlp_type = "token_routed"
    elif gate_lin_key:
        expert_inter = state_dict[gate_lin_key].shape[0]
        num_experts_found = sum(1 for k in state_dict if "layers.0.mlp.experts." in k and "gate_proj.weight" in k)
        inter = expert_inter * num_experts_found
        mlp_type = "token_routed"
    else:
        num_experts_found = 1
        inter_key = next((k for k in state_dict if "gate_proj" in k and "layers.0." in k), None)
        if inter_key is None:
            inter_key = next(k for k in state_dict if "up_proj" in k and "layers.0." in k)
        w = state_dict[inter_key]
        inter = w.shape[0] if w.dim() == 2 else w.shape[2]
        mlp_type = "swiglu"

    head_dim = 64
    q_key = next(k for k in state_dict if "q_proj.weight" in k and "layers.0." in k)
    num_attention_heads = state_dict[q_key].shape[0] // head_dim
    k_key = next(k for k in state_dict if "k_proj.weight" in k and "layers.0." in k)
    num_kv_heads = state_dict[k_key].shape[0] // head_dim

    print(f"  Inferred: hidden={hidden}, layers={num_layers}, vocab={vocab}, "
          f"inter={inter}, experts={num_experts_found}, heads={num_attention_heads}, kv_heads={num_kv_heads}, mlp={mlp_type}")

    config = ModelConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=inter,
        vocab_size=vocab,
        num_experts=num_experts_found,
        max_position_embeddings=2048,
        mlp_type=mlp_type,
        use_mu_guidance=True,
        use_qk_norm=True,
    )

    model = ComplexityModel(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


class ActivationCollector3D:
    """Hook to collect per-expert mean activations per layer using real routing."""

    def __init__(self):
        self.activations = []  # (expert_id, layer_idx, vector)
        self.handles = []
        self.current_input_ids = None

    def register(self, model):
        layers = model.layers if hasattr(model, "layers") else model.model.layers
        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            is_routed = (
                (hasattr(mlp, "gate_proj_w") and mlp.gate_proj_w.dim() == 3)
                or (hasattr(mlp, "gate_up_proj") and mlp.gate_up_proj.dim() == 3)
                or hasattr(mlp, "experts")
            )
            if is_routed:
                handle = mlp.register_forward_hook(self._make_hook(layer_idx))
                self.handles.append(handle)
        print(f"Hooks on {len(self.handles)} TokenRoutedMLP layers")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Use OUTPUT (after expert dispatch), not input (before)
            x = output if isinstance(output, torch.Tensor) else output[0]
            batch_size, seq_len, hidden = x.shape

            if hasattr(module, "token_to_expert") and self.current_input_ids is not None:
                token_ids = self.current_input_ids.clamp(0, module.vocab_size - 1)
                expert_ids = module.token_to_expert[token_ids]
                num_experts = module.num_experts
            else:
                num_experts = 4
                expert_ids = torch.arange(seq_len, device=x.device).unsqueeze(0) % num_experts
                expert_ids = expert_ids.expand(batch_size, -1)

            for expert_id in range(num_experts):
                for b in range(batch_size):
                    mask = (expert_ids[b] == expert_id)
                    if not mask.any():
                        continue
                    act = x[b, mask, :].mean(dim=0).detach().cpu().float().numpy()
                    self.activations.append((expert_id, layer_idx, act))
        return hook_fn

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def collect(model, num_samples=64, seq_len=512, vocab_size=32000, device="cuda"):
    """Collect activations on random sequences."""
    collector = ActivationCollector3D()
    collector.register(model)
    model = model.to(device)

    print(f"Collecting activations ({num_samples} x {seq_len} tokens)...")
    for i in range(num_samples):
        input_ids = torch.randint(3, vocab_size, (1, seq_len), device=device)
        collector.current_input_ids = input_ids
        model(input_ids)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{num_samples} ({len(collector.activations)} activations)")

    collector.remove()
    print(f"Total: {len(collector.activations)} activations")
    return collector.activations


def plot_3d_matplotlib(X_3d, expert_ids, layer_ids, num_experts, output_dir):
    """Static 3D plots with matplotlib."""
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    # Plot 1: colored by expert
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    for eid in range(num_experts):
        mask = expert_ids == eid
        if not mask.any():
            continue
        ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                   c=colors[eid % len(colors)], label=f"Expert {eid}",
                   alpha=0.7, s=30, edgecolors="white", linewidth=0.2)

    ax.set_title("Expert Activation T-SNE (3D)", fontsize=16, fontweight="bold")
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")
    ax.set_zlabel("T-SNE 3")
    ax.legend(fontsize=12, loc="upper left")

    path = output_dir / "expert_tsne_3d_by_expert.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Plot 2: colored by layer
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    scatter = ax.scatter(X_3d[:, 0], X_3d[:, 1], X_3d[:, 2],
                         c=layer_ids, cmap="viridis",
                         alpha=0.7, s=30, edgecolors="white", linewidth=0.2)
    fig.colorbar(scatter, ax=ax, label="Layer index", shrink=0.6)

    ax.set_title("Layer Depth T-SNE (3D)", fontsize=16, fontweight="bold")
    ax.set_xlabel("T-SNE 1")
    ax.set_ylabel("T-SNE 2")
    ax.set_zlabel("T-SNE 3")

    path = output_dir / "expert_tsne_3d_by_layer.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")

    # Plot 3: multi-angle view (4 angles)
    fig, axes_list = plt.subplots(2, 2, figsize=(20, 16),
                                   subplot_kw={"projection": "3d"})
    angles = [(30, 45), (30, 135), (60, 45), (10, 90)]

    for ax, (elev, azim) in zip(axes_list.flat, angles):
        for eid in range(num_experts):
            mask = expert_ids == eid
            if not mask.any():
                continue
            ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
                       c=colors[eid % len(colors)], label=f"E{eid}",
                       alpha=0.6, s=20, edgecolors="white", linewidth=0.1)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(f"elev={elev}°, azim={azim}°", fontsize=11)
        ax.set_xlabel("T1", fontsize=9)
        ax.set_ylabel("T2", fontsize=9)
        ax.set_zlabel("T3", fontsize=9)

    axes_list.flat[0].legend(fontsize=9, loc="upper left")
    plt.suptitle("Expert Activation T-SNE — Multi-Angle View", fontsize=16, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "expert_tsne_3d_multiangle.png"
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_3d_plotly(X_3d, expert_ids, layer_ids, num_experts, output_dir):
    """Interactive 3D plots with plotly (HTML)."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("plotly not installed, skipping interactive HTML (pip install plotly)")
        return

    colors = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]

    # By expert
    fig = go.Figure()
    for eid in range(num_experts):
        mask = expert_ids == eid
        if not mask.any():
            continue
        fig.add_trace(go.Scatter3d(
            x=X_3d[mask, 0], y=X_3d[mask, 1], z=X_3d[mask, 2],
            mode="markers",
            marker=dict(size=3, color=colors[eid % len(colors)], opacity=0.7),
            name=f"Expert {eid}",
        ))
    fig.update_layout(
        title="Expert Activation T-SNE 3D (Interactive)",
        scene=dict(xaxis_title="T-SNE 1", yaxis_title="T-SNE 2", zaxis_title="T-SNE 3"),
        width=1000, height=800,
    )
    path = output_dir / "expert_tsne_3d_interactive.html"
    fig.write_html(str(path))
    print(f"Saved: {path}")

    # By layer
    fig2 = go.Figure(data=[go.Scatter3d(
        x=X_3d[:, 0], y=X_3d[:, 1], z=X_3d[:, 2],
        mode="markers",
        marker=dict(size=3, color=layer_ids, colorscale="Viridis",
                    opacity=0.7, colorbar=dict(title="Layer")),
    )])
    fig2.update_layout(
        title="Layer Depth T-SNE 3D (Interactive)",
        scene=dict(xaxis_title="T-SNE 1", yaxis_title="T-SNE 2", zaxis_title="T-SNE 3"),
        width=1000, height=800,
    )
    path = output_dir / "layer_tsne_3d_interactive.html"
    fig2.write_html(str(path))
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="3D T-SNE of expert activations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./plots_3d")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    print(f"Loading: {args.checkpoint}")
    state_dict = load_checkpoint(args.checkpoint)
    model, config = build_model(state_dict)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params, {config.num_hidden_layers} layers, "
          f"{config.num_experts} experts")

    # Collect
    activations = collect(
        model, num_samples=args.num_samples, seq_len=args.seq_len,
        vocab_size=config.vocab_size, device=args.device,
    )

    # Free GPU
    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Unpack
    expert_ids = np.array([a[0] for a in activations])
    layer_ids = np.array([a[1] for a in activations])
    vectors = np.array([a[2] for a in activations])

    print(f"\nActivation matrix: {vectors.shape}")
    for eid in range(config.num_experts):
        print(f"  Expert {eid}: {(expert_ids == eid).sum()} activations")

    # PCA -> 3D T-SNE
    pca_dim = min(50, vectors.shape[0] - 1, vectors.shape[1])
    print(f"\nPCA: {vectors.shape[1]} -> {pca_dim} dims")
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(vectors)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    perplexity = min(args.perplexity, X_pca.shape[0] - 1)
    print(f"3D T-SNE on {X_pca.shape[0]} vectors (perplexity={perplexity:.0f})...")
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=42, max_iter=1000)
    X_3d = tsne.fit_transform(X_pca)
    print(f"T-SNE done! KL divergence: {tsne.kl_divergence_:.4f}")

    # Clamp representation outliers (2-98 percentile per axis)
    p_low, p_high = np.percentile(X_3d, [2, 98], axis=0)
    X_3d = np.clip(X_3d, p_low, p_high)
    print(f"Clamped to [{p_low.round(1)}, {p_high.round(1)}]")

    # Plots
    print(f"\nGenerating 3D plots in {output_dir}/\n")
    plot_3d_matplotlib(X_3d, expert_ids, layer_ids, config.num_experts, output_dir)
    plot_3d_plotly(X_3d, expert_ids, layer_ids, config.num_experts, output_dir)

    print(f"\nDone! Plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
