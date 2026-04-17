"""
Pre-training 1B MoE v1 — Token-Routed + Mu-Guidance.

Overtrain-light recipe: 1B total / ~800M active × 100B tokens (~100 tok/param).
FSDP full_shard, 4× B200 cible (~30h spot).

hidden=1536, 20 layers, GQA 24/6, inter=4096, 4 experts + shared full-width
→ ~1.0B total, ~800M active per token (shared dominates).

Usage:
    torchrun --nproc_per_node=4 scripts/train_1b_moe_v1.py
    torchrun --nproc_per_node=4 scripts/train_1b_moe_v1.py --resume checkpoints/1b-moe-v1/step_5000

Complexity-ML — 2026
"""

from complexity.gpu import setup_gpu
setup_gpu()

from complexity.config import ModelConfig
from complexity.training import TrainRunner


def make_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=1536,
        num_hidden_layers=20,
        num_attention_heads=24,
        num_key_value_heads=6,
        intermediate_size=4096,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=4,
        shared_expert=True,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=True,
    )


if __name__ == "__main__":
    from complexity.gpu.distributed_cleanup import safe_main
    safe_main(lambda: TrainRunner(
        make_config=make_config,
        run_name="1b-moe-v1",
        checkpoint_dir="./checkpoints/1b-moe-v1",
        default_lr=3e-4,
        default_batch_size=64,
        default_seq_len=2048,
        default_gradient_accumulation=2,
        default_target_tokens=100_000_000_000,
    ).run())
