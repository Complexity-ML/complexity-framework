"""CLI parser for the o200k pretraining runner."""

from __future__ import annotations

import argparse

from .profiles import PROFILES


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local o200k residual Token-Routed pretraining runner")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="100m")
    parser.add_argument("--dataset", choices=["random", "text", "fineweb", "tokens"], default="random")
    parser.add_argument("--text-file", type=str, default=None)
    parser.add_argument(
        "--tokens-path",
        type=str,
        default=None,
        help="Directory containing tokens.bin + tokens.idx.json for --dataset tokens.",
    )
    parser.add_argument("--tokenizer", type=str, default="./tokenizer-o200k")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--optimizer", choices=["adamw", "muon_tr"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--muon-lr", type=float, default=0.003)
    parser.add_argument("--muon-scope", choices=["expert", "expert_shared", "all"], default="expert")
    parser.add_argument("--muon-ns-steps", type=int, default=5)
    parser.add_argument("--muon-adaptive-ns", action="store_true")
    parser.add_argument("--muon-lr-warmup-steps", type=int, default=50)
    parser.add_argument("--muon-lr-decay-start-step", type=int, default=0)
    parser.add_argument("--muon-lr-decay-end-step", type=int, default=0)
    parser.add_argument("--muon-lr-decay-min-mult", type=float, default=1.0)
    parser.add_argument("--muon-skip-ns-warmup-steps", type=int, default=0)
    parser.add_argument("--muon-no-nesterov", action="store_true")
    parser.add_argument("--muon-orthogonal-blend", type=float, default=0.5)
    parser.add_argument("--muon-orthogonal-blend-start", type=float, default=None)
    parser.add_argument("--muon-orthogonal-blend-decay-steps", type=int, default=0)
    parser.add_argument("--muon-max-param-rms-ratio", type=float, default=None)
    parser.add_argument("--muon-token-count-scaling", action="store_true")
    parser.add_argument("--muon-max-lr-ratio", type=float, default=2.0)
    parser.add_argument("--muon-max-update-rms", type=float, default=1.0)
    parser.add_argument("--expert-lr-scale", type=float, default=1.0)
    parser.add_argument("--expert-weight-decay", type=float, default=0.005)
    parser.add_argument("--shared-lr-scale", type=float, default=1.0)
    parser.add_argument("--shared-weight-decay", type=float, default=0.01)
    parser.add_argument("--hidden-size", type=int, default=None)
    parser.add_argument("--num-hidden-layers", type=int, default=None)
    parser.add_argument("--num-attention-heads", type=int, default=None)
    parser.add_argument("--num-key-value-heads", type=int, default=None)
    parser.add_argument("--intermediate-size", type=int, default=None)
    parser.add_argument("--shared-intermediate-size", type=int, default=None)
    parser.add_argument(
        "--shared-expert-chunk-tokens",
        type=int,
        default=32768,
        help=(
            "Chunk the dense shared expert over this many tokens to reduce "
            "activation peak memory without model-wide gradient checkpointing. "
            "Set 0 to compute it in one pass."
        ),
    )
    parser.add_argument("--shared-gate-init", type=float, default=1.0)
    parser.add_argument("--routed-gate-init", type=float, default=0.1)
    parser.add_argument("--learn-shared-routed-gates", dest="learn_shared_routed_gates", action="store_true", default=True)
    parser.add_argument("--no-learn-shared-routed-gates", dest="learn_shared_routed_gates", action="store_false")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--top-k-primary-weight", type=float, default=0.5)
    parser.add_argument(
        "--top-k-primary-weight-final",
        type=float,
        default=0.85,
        help="Final primary route weight for scheduled Token-Routed specialization. Set equal to --top-k-primary-weight to disable.",
    )
    parser.add_argument(
        "--top-k-primary-weight-schedule-ratio",
        type=float,
        default=0.5,
        help="Fraction of training steps used to ramp primary route weight from start to final.",
    )
    parser.add_argument(
        "--use-custom-kernels",
        choices=["auto", "true", "false"],
        default="auto",
        help="Custom Triton/CUDA kernels. auto enables NVIDIA CUDA, disables ROCm by default.",
    )
    parser.add_argument(
        "--static-expert-capacity",
        action="store_true",
        help="Use export-friendly TR dispatch for torch.distributed.pipelining.",
    )
    parser.add_argument(
        "--cggr",
        choices=["auto", "true", "false"],
        default="auto",
        help="Token-Routed CGGR grouped-GEMM Triton dispatch. auto follows --use-custom-kernels.",
    )
    parser.add_argument(
        "--use-cggr",
        dest="cggr",
        action="store_const",
        const="true",
        help="Deprecated alias for --cggr true.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Wrap the model with torch.compile (Inductor).",
    )
    parser.add_argument(
        "--compile-mode",
        choices=["default", "reduce-overhead", "max-autotune"],
        default="default",
        help="torch.compile mode.",
    )
    parser.add_argument("--routing-strategy", choices=["zipf", "zipf_token_class"], default="zipf")
    parser.add_argument("--use-mu-guidance", action="store_true")
    parser.add_argument("--mu-clamp", action="store_true")
    parser.add_argument("--mu-norm", action="store_true")
    parser.add_argument("--mu-alpha-init", type=float, default=1.0)
    parser.add_argument("--mu-init-value", type=float, default=0.0)
    parser.add_argument("--mu-context-min", type=float, default=-2.0)
    parser.add_argument("--mu-context-max", type=float, default=2.0)
    parser.add_argument("--log-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--eval-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--grad-ckpt", dest="grad_ckpt", action="store_true", default=True)
    parser.add_argument("--no-grad-ckpt", dest="grad_ckpt", action="store_false")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--empty-cache-every", type=int, default=50)
    parser.add_argument("--max-grad-norm", type=float, default=0.0)
    parser.add_argument("--moe-telemetry", action="store_true")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--z-loss", type=float, default=0.0)
    parser.add_argument("--loss-chunk-tokens", type=int, default=1024)
    parser.add_argument("--loss-backend", choices=["auto", "chunked", "liger"], default="auto")
    parser.add_argument("--loss-checkpoint-chunks", action="store_true")
    parser.add_argument("--save-steps", type=int, default=5000)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--force-resume", action="store_true")
    parser.add_argument("--no-zipf-from-text", action="store_true")
    return parser
