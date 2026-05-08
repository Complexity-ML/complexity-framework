"""
Pre-training 400M v1 — Token-Routed MLP + Mu-Guidance.

hidden=1024, layers=20, heads=16, kv_heads=4, inter=2008, 4 experts → ~384M
(iso-params with train_400m_dense at inter=4358)

Usage:
    torchrun --nproc_per_node=2 scripts/train_400m_v1.py
    torchrun --nproc_per_node=2 scripts/train_400m_v1.py --resume checkpoints/400m-v1/step_10000

Complexity-ML — 2026
"""

from complexity.gpu import setup_gpu
setup_gpu()

from complexity.config import ModelConfig
from complexity.training import TrainRunner


def make_config() -> ModelConfig:
    return ModelConfig(
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=4,
        intermediate_size=2008,
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
        run_name="400m-v1",
        checkpoint_dir="./checkpoints/400m-v1",
        default_lr=2.1e-4,
        default_batch_size=128,
        default_seq_len=2048,
        default_target_tokens=8_000_000_000,
    ).run())
