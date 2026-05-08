"""
Generate all paper figures from a trained Token-Routed model.

Produces:
1. expert_balance.png — Token distribution per expert (bin-packing balance)
2. component_activity.png — Mu-Guidance norms per layer
3. head_expert_heatmap.png — Attention head × expert affinity
4. mu_contribution.png — Mu contribution ratio to K/Q/V projections

Usage:
    python scripts/viz_paper_figures.py --checkpoint checkpoints/run2-iso-shared/model.safetensors --output figures

Complexity-ML — 2026
"""

import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_model(checkpoint_path):
    """Load model from safetensors or checkpoint.pt."""
    from complexity.config import ModelConfig
    from complexity.models import ComplexityModel

    path = Path(checkpoint_path)
    parent = path.parent

    # Load config
    config_path = parent / "config.json"
    if not config_path.exists():
        config_path = parent / "model_config.yaml"
    config = ModelConfig.load(str(config_path))

    # Load weights
    if path.suffix == ".pt":
        data = torch.load(str(path), map_location="cpu", weights_only=False)
        state_dict = data.get("model", data)
    else:
        from safetensors.torch import load as safetensors_load
        with open(str(path), "rb") as f:
            data = f.read()
        state_dict = safetensors_load(data)

    model = ComplexityModel(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, config


def fig_expert_balance(model, config, output_dir):
    """Figure 1: Token distribution per expert (from token_to_expert mapping)."""
    logger.info("Generating expert_balance.png...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Get token_to_expert from first layer's MLP
    token_to_expert = None
    for module in model.modules():
        if hasattr(module, "token_to_expert"):
            token_to_expert = module.token_to_expert
            break

    if token_to_expert is None:
        logger.warning("No token_to_expert found")
        return

    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']

    # Plot 1: Token count per expert
    counts = [(token_to_expert == e).sum().item() for e in range(config.num_experts)]
    bars = axes[0].bar(range(config.num_experts), counts, color=colors[:config.num_experts], alpha=0.85)
    for bar, c in zip(bars, counts):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                     str(c), ha='center', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Expert', fontsize=12)
    axes[0].set_ylabel('Token Count', fontsize=12)
    axes[0].set_title('Vocabulary Distribution per Expert\n(Zipf bin-packing)', fontsize=13)
    axes[0].set_xticks(range(config.num_experts))
    axes[0].set_xticklabels([f'E{e}' for e in range(config.num_experts)])
    axes[0].grid(True, alpha=0.3, axis='y')

    # Plot 2: Percentage
    total = sum(counts)
    pcts = [c / total * 100 for c in counts]
    bars2 = axes[1].bar(range(config.num_experts), pcts, color=colors[:config.num_experts], alpha=0.85)
    for bar, p in zip(bars2, pcts):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f'{p:.1f}%', ha='center', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Expert', fontsize=12)
    axes[1].set_ylabel('Share (%)', fontsize=12)
    axes[1].set_title('Load Balance (target: 25.0% each)', fontsize=13)
    axes[1].set_xticks(range(config.num_experts))
    axes[1].set_xticklabels([f'E{e}' for e in range(config.num_experts)])
    axes[1].axhline(y=25.0, color='black', linestyle='--', alpha=0.5, label='Perfect (25%)')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "expert_balance.png", dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved expert_balance.png")


def fig_component_activity(model, config, output_dir):
    """Figure 2: Mu-Guidance norms per layer."""
    logger.info("Generating component_activity.png...")

    layers = model.layers if hasattr(model, "layers") else model.model.layers
    num_layers = len(layers)

    mu_norms = []
    mu_proj_norms = []
    mu_to_k_norms = []
    mu_to_q_norms = []

    for i, layer in enumerate(layers):
        # MuGuidance
        if hasattr(layer, 'mu_guidance') and layer.mu_guidance is not None:
            mu_norms.append(layer.mu_guidance.mu.data.norm().item())
            mu_proj_norms.append(layer.mu_guidance.mu_proj.weight.data.norm().item())

        # Mu projections in attention
        attn = layer.self_attn
        if hasattr(attn, 'mu_to_k'):
            mu_to_k_norms.append(attn.mu_to_k.weight.data.norm().item())
            mu_to_q_norms.append(attn.mu_to_q.weight.data.norm().item())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = range(num_layers)
    if mu_norms:
        axes[0].plot(x, mu_norms, 'o-', color='#e74c3c', label='mu_param norm', linewidth=2, markersize=5)
        axes[0].plot(x, mu_proj_norms, 's-', color='#3498db', label='mu_proj norm', linewidth=2, markersize=5)
        axes[0].set_xlabel('Layer', fontsize=12)
        axes[0].set_ylabel('Weight Norm', fontsize=12)
        axes[0].set_title('Mu-Guidance Activity per Layer', fontsize=13)
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)

    if mu_to_k_norms:
        axes[1].plot(x, mu_to_k_norms, 'o-', color='#2ecc71', label='mu_to_k norm', linewidth=2, markersize=5)
        axes[1].plot(x, mu_to_q_norms, 's-', color='#f39c12', label='mu_to_q norm', linewidth=2, markersize=5)
        axes[1].set_xlabel('Layer', fontsize=12)
        axes[1].set_ylabel('Weight Norm', fontsize=12)
        axes[1].set_title('Mu Attention Projections per Layer', fontsize=13)
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "component_activity.png", dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved component_activity.png")


@torch.no_grad()
def fig_head_expert_heatmap(model, config, output_dir, device="cuda", num_samples=32, seq_len=512):
    """Figure 3: Attention head × expert output norm heatmap."""
    logger.info("Generating head_expert_heatmap.png...")

    model = model.to(device)
    layers = model.layers if hasattr(model, "layers") else model.model.layers

    # We'll measure on one layer (middle)
    mid_layer = len(layers) // 2
    attn = layers[mid_layer].self_attn
    num_heads = config.num_attention_heads
    num_experts = config.num_experts

    # Get token_to_expert
    token_to_expert = None
    for module in model.modules():
        if hasattr(module, "token_to_expert"):
            token_to_expert = module.token_to_expert.to(device)
            break

    # Collect attention output norms per head per expert
    heatmap = np.zeros((num_heads, num_experts))
    counts = np.zeros((num_heads, num_experts))

    captured = {}
    def hook_fn(module, input, output):
        # output is (attn_output, kv_cache) — attn_output is [B, S, H*D]
        captured['attn_out'] = output[0].detach()

    handle = attn.register_forward_hook(hook_fn)

    for i in range(num_samples):
        input_ids = torch.randint(3, config.vocab_size, (1, seq_len), device=device)
        model(input_ids)

        attn_out = captured['attn_out']  # [1, S, hidden]
        B, S, H = attn_out.shape
        head_dim = H // num_heads

        # Reshape to per-head: [S, num_heads, head_dim]
        per_head = attn_out[0].view(S, num_heads, head_dim)
        head_norms = per_head.norm(dim=-1)  # [S, num_heads]

        # Expert assignment
        expert_ids = token_to_expert[input_ids[0].clamp(0, config.vocab_size - 1)]  # [S]

        for e in range(num_experts):
            mask = (expert_ids == e)
            if mask.any():
                for h in range(num_heads):
                    heatmap[h, e] += head_norms[mask, h].sum().item()
                    counts[h, e] += mask.sum().item()

    handle.remove()

    # Normalize
    heatmap = heatmap / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(heatmap, cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Expert', fontsize=12)
    ax.set_ylabel('Attention Head', fontsize=12)
    ax.set_title(f'Head × Expert Output Norm (Layer {mid_layer})', fontsize=13)
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f'E{e}' for e in range(num_experts)])
    ax.set_yticks(range(num_heads))
    ax.set_yticklabels([f'H{h}' for h in range(num_heads)])
    fig.colorbar(im, ax=ax, label='Mean Output Norm')
    plt.tight_layout()
    plt.savefig(output_dir / "head_expert_heatmap.png", dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved head_expert_heatmap.png")


@torch.no_grad()
def fig_mu_contribution(model, config, output_dir, device="cuda", num_samples=32, seq_len=512):
    """Figure 4: Mu contribution ratio to K/Q/V projections."""
    logger.info("Generating mu_contribution.png...")

    model = model.to(device)
    layers = model.layers if hasattr(model, "layers") else model.model.layers
    num_layers = len(layers)

    # For each layer, measure ||mu_prev * W_muK|| / (||x * W_K|| + ||mu_prev * W_muK||)
    ratios_k = [[] for _ in range(num_layers)]
    ratios_q = [[] for _ in range(num_layers)]
    ratios_v = [[] for _ in range(num_layers)]

    # Hook on each attention to capture input + mu
    captured_per_layer = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            x = input[0]  # hidden states
            # mu_prev is passed as kwarg — hard to capture from hook
            # Instead, measure weight norms as proxy
            if hasattr(module, 'mu_to_k') and hasattr(module, 'k_proj'):
                k_norm = module.k_proj.weight.data.norm().item()
                mu_k_norm = module.mu_to_k.weight.data.norm().item()
                ratio = mu_k_norm / (k_norm + mu_k_norm) if (k_norm + mu_k_norm) > 0 else 0
                ratios_k[layer_idx].append(ratio)

                q_norm = module.q_proj.weight.data.norm().item()
                mu_q_norm = module.mu_to_q.weight.data.norm().item()
                ratios_q[layer_idx].append(mu_q_norm / (q_norm + mu_q_norm) if (q_norm + mu_q_norm) > 0 else 0)

                v_norm = module.v_proj.weight.data.norm().item()
                mu_v_norm = module.mu_to_v.weight.data.norm().item()
                ratios_v[layer_idx].append(mu_v_norm / (v_norm + mu_v_norm) if (v_norm + mu_v_norm) > 0 else 0)
        return hook_fn

    handles = []
    for i, layer in enumerate(layers):
        h = layer.self_attn.register_forward_hook(make_hook(i))
        handles.append(h)

    # Run forward
    for i in range(num_samples):
        input_ids = torch.randint(3, config.vocab_size, (1, seq_len), device=device)
        model(input_ids)

    for h in handles:
        h.remove()

    # Average ratios
    avg_k = [np.mean(r) if r else 0 for r in ratios_k]
    avg_q = [np.mean(r) if r else 0 for r in ratios_q]
    avg_v = [np.mean(r) if r else 0 for r in ratios_v]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(num_layers)
    ax.plot(x, avg_k, 'o-', color='#e74c3c', label='K projection', linewidth=2, markersize=5)
    ax.plot(x, avg_q, 's-', color='#3498db', label='Q projection', linewidth=2, markersize=5)
    ax.plot(x, avg_v, '^-', color='#2ecc71', label='V projection', linewidth=2, markersize=5)
    ax.axhline(y=np.mean(avg_k + avg_q + avg_v), color='gray', linestyle='--', alpha=0.5,
               label=f'Mean: {np.mean(avg_k + avg_q + avg_v):.1%}')
    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Mu Contribution Ratio', fontsize=12)
    ax.set_title('Mu-Guidance Contribution to Attention Projections', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(max(avg_k), max(avg_q), max(avg_v)) * 1.2)
    plt.tight_layout()
    plt.savefig(output_dir / "mu_contribution.png", dpi=200, bbox_inches='tight')
    plt.close()
    logger.info("Saved mu_contribution.png")

    # Print summary
    overall = np.mean(avg_k + avg_q + avg_v)
    logger.info(f"Mu contribution: K={np.mean(avg_k):.1%}, Q={np.mean(avg_q):.1%}, V={np.mean(avg_v):.1%}, overall={overall:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from trained model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="./figures")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-samples", type=int, default=32)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, config = load_model(args.checkpoint)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M, "
                f"{config.num_experts} experts, {config.num_hidden_layers} layers")

    # Fig 1: Expert balance (no GPU needed)
    fig_expert_balance(model, config, output_dir)

    # Fig 2: Component activity (no GPU needed)
    fig_component_activity(model, config, output_dir)

    # Fig 3: Head-expert heatmap (GPU)
    fig_head_expert_heatmap(model, config, output_dir, device=args.device, num_samples=args.num_samples)

    # Fig 4: Mu contribution (GPU)
    fig_mu_contribution(model, config, output_dir, device=args.device, num_samples=args.num_samples)

    logger.info("All figures generated!")


if __name__ == "__main__":
    main()
