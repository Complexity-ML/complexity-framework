"""
2D t-SNE visualization of expert specialization.

Collects mean activations per expert per layer from a token-routed model,
then plots 2D t-SNE colored by expert and by layer depth.

Each point = 1 expert × 1 layer (mean activation over all tokens routed
to that expert at that layer). This shows expert clustering clearly.

Usage:
    python scripts/viz_expert_tsne.py --checkpoint checkpoints/run2-iso-shared/model.safetensors
    python scripts/viz_expert_tsne.py --checkpoint checkpoints/run2-iso-shared/model.safetensors --num-samples 64

Complexity-ML — 2026
"""

import argparse
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_checkpoint(checkpoint_path):
    """Load model state dict (safetensors or .pt)."""
    path = Path(checkpoint_path)
    if path.suffix == ".pt":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = data.get("model", data)
    else:
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
    """Build model from state dict."""
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    q_key = next(k for k in state_dict if "q_proj.weight" in k and "layers.0." in k)
    hidden = state_dict[q_key].shape[0]
    num_layers = max(int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")) + 1
    vocab = state_dict["embed_tokens.weight"].shape[0]

    gate_3d_key = next((k for k in state_dict if "gate_proj_w" in k and "layers.0." in k), None)
    if gate_3d_key:
        num_experts_found = state_dict[gate_3d_key].shape[0]
        expert_inter = state_dict[gate_3d_key].shape[2]
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

    k_key = next(k for k in state_dict if "k_proj.weight" in k and "layers.0." in k)
    num_kv_heads = state_dict[k_key].shape[0] // (hidden // 12)

    logger.info(f"Inferred: hidden={hidden}, layers={num_layers}, experts={num_experts_found}, inter={inter}")

    config = ModelConfig(
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_attention_heads=12,
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


class ActivationCollector:
    """Hook to collect per-expert mean activations per layer using real routing."""

    def __init__(self):
        self.activations = []  # (expert_id, layer_idx, vector)
        self.handles = []
        self.current_input_ids = None  # set before each forward

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
        logger.info(f"Hooks on {len(self.handles)} TokenRoutedMLP layers")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # Use OUTPUT (after expert dispatch), not input (before)
            x = output if isinstance(output, torch.Tensor) else output[0]
            batch_size, seq_len, hidden = x.shape

            # Use real token_to_expert mapping
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
    collector = ActivationCollector()
    collector.register(model)
    model = model.to(device)

    logger.info(f"Collecting activations ({num_samples} x {seq_len} tokens)...")
    for i in range(num_samples):
        input_ids = torch.randint(3, vocab_size, (1, seq_len), device=device)
        collector.current_input_ids = input_ids
        model(input_ids)
        if (i + 1) % 10 == 0:
            logger.info(f"  {i+1}/{num_samples} ({len(collector.activations)} activations)")

    collector.remove()
    logger.info(f"Total: {len(collector.activations)} activations")
    return collector.activations


def main():
    parser = argparse.ArgumentParser(description="2D t-SNE of expert activations (mean per expert per layer)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, default="./figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load
    logger.info(f"Loading: {args.checkpoint}")
    state_dict = load_checkpoint(args.checkpoint)
    model, config = build_model(state_dict)

    # Collect
    activations = collect(
        model, num_samples=args.num_samples, seq_len=args.seq_len,
        vocab_size=config.vocab_size, device=args.device,
    )

    del model
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Unpack
    expert_ids = np.array([a[0] for a in activations])
    layer_ids = np.array([a[1] for a in activations])
    vectors = np.array([a[2] for a in activations])

    logger.info(f"Activation matrix: {vectors.shape}")

    # PCA + 2D t-SNE
    pca_dim = min(50, vectors.shape[0] - 1, vectors.shape[1])
    pca = PCA(n_components=pca_dim, random_state=42)
    X_pca = pca.fit_transform(vectors)
    logger.info(f"PCA: {vectors.shape[1]} -> {pca_dim} dims, variance={pca.explained_variance_ratio_.sum():.1%}")

    perplexity = min(args.perplexity, X_pca.shape[0] - 1)
    logger.info(f"2D t-SNE on {X_pca.shape[0]} vectors (perplexity={perplexity:.0f})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    X_2d = tsne.fit_transform(X_pca)

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']

    # Plot 1: by expert + by layer
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for eid in range(config.num_experts):
        mask = expert_ids == eid
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1],
                        c=colors[eid % len(colors)], label=f'Expert {eid}',
                        alpha=0.7, s=30, edgecolors='white', linewidth=0.2)
    axes[0].legend(fontsize=12)
    axes[0].set_title('Expert Activation t-SNE', fontsize=14)
    axes[0].set_xlabel('t-SNE 1')
    axes[0].set_ylabel('t-SNE 2')

    scatter = axes[1].scatter(X_2d[:, 0], X_2d[:, 1],
                               c=layer_ids, cmap='viridis',
                               alpha=0.7, s=30, edgecolors='white', linewidth=0.2)
    fig.colorbar(scatter, ax=axes[1], label='Layer index')
    axes[1].set_title('Layer Depth t-SNE', fontsize=14)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')

    plt.tight_layout()
    path = output_dir / "expert_tsne_2d.png"
    plt.savefig(path, dpi=200, bbox_inches='tight')
    logger.info(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    main()
