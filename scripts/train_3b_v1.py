"""
Pre-training 3B ComplexityModel — Token-Routed MoE + Mu-Guidance.

Overtraining recipe: 3B params × 200B tokens (67× overtrain).
FSDP full_shard for 8× B200 multi-GPU.

hidden=2048, 26 layers, GQA 32/8, 8 experts, shared_expert full-width
inter=7552 → ~3.02B params

Usage:
    torchrun --nproc_per_node=8 scripts/train_3b_v1.py
    torchrun --nproc_per_node=8 scripts/train_3b_v1.py --resume checkpoints/3b-v1/step_10000

Complexity-ML — 2026
"""

from complexity.gpu import setup_gpu
setup_gpu()

from complexity.config import ModelConfig
from complexity.training import TrainRunner


def make_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=2048,
        num_hidden_layers=26,
        num_attention_heads=32,
        num_key_value_heads=8,
        intermediate_size=7552,
        vocab_size=32000,
        max_position_embeddings=4096,
        attention_type="gqa",
        mlp_type="token_routed",
        num_experts=8,
        shared_expert=True,
        norm_type="rmsnorm",
        use_qk_norm=True,
        use_mu_guidance=True,
    )


if __name__ == "__main__":
    from complexity.gpu.distributed_cleanup import safe_main
    safe_main(lambda: TrainRunner(
        make_config=make_config,
        run_name="3b-v1",
        checkpoint_dir="./checkpoints/3b-v1",
        default_lr=3e-4,
        default_batch_size=64,
        default_seq_len=2048,
        default_gradient_accumulation=2,
        default_target_tokens=200_000_000_000,
    ).run())
