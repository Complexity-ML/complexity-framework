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
    parser.add_argument("--attention-type", type=str, default="gqa")
    parser.add_argument("--causal-conv-kernel-size", type=int, default=4)
    parser.add_argument("--causal-conv-dilation-cycle", type=int, default=8)
    parser.add_argument("--causal-state-rank", type=int, default=16)
    parser.add_argument("--causal-context-gate-init", type=float, default=1.0)
    parser.add_argument("--causal-contextual-mix-init", type=float, default=0.0)
    parser.add_argument("--causal-context-fusion-size", type=int, default=0)
    parser.add_argument("--causal-stable-delta", action="store_true")
    parser.add_argument("--causal-delta-chunk-size", type=int, default=512)
    parser.add_argument("--causal-delta-timescales", type=int, default=1)
    parser.add_argument("--causal-delta-collision-normalized", action="store_true")
    parser.add_argument("--causal-delta-lexical-values", action="store_true")
    parser.add_argument("--causal-delta-lexical-forge", action="store_true")
    parser.add_argument("--causal-delta-occurrence-address", action="store_true")

    parser.add_argument(
        "--lexical-attention-layer-indices",
        type=int,
        nargs="*",
        default=(),
    )
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
    parser.add_argument("--shared-expert", dest="shared_expert", action="store_true", default=True)
    parser.add_argument("--no-shared-expert", dest="shared_expert", action="store_false")
    parser.add_argument("--shared-gate-init", type=float, default=1.0)
    parser.add_argument("--routed-gate-init", type=float, default=0.1)
    parser.add_argument(
        "--shared-gate-final",
        type=float,
        default=None,
        help="Optional final shared gate value for dense-to-expert curriculum. Disabled when omitted.",
    )
    parser.add_argument(
        "--routed-gate-final",
        type=float,
        default=None,
        help="Optional final routed gate value for dense-to-expert curriculum. Disabled when omitted.",
    )
    parser.add_argument(
        "--gate-schedule-ratio",
        type=float,
        default=0.5,
        help="Fraction of training steps used to ramp shared/routed gates to their final values.",
    )
    parser.add_argument(
        "--expert-diversity-lambda",
        type=float,
        default=0.0,
        help="Optional expert separation penalty. 0 disables it.",
    )
    parser.add_argument(
        "--expert-diversity-schedule-ratio",
        type=float,
        default=0.7,
        help="Fraction of training steps used to ramp expert diversity from 0 to lambda.",
    )
    parser.add_argument(
        "--expert-diversity-target",
        choices=["down", "all"],
        default="down",
        help="Expert weights used by the separation penalty.",
    )
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
    parser.add_argument(
        "--routing-strategy",
        choices=["zipf", "modulo", "round_robin", "random", "lsh_hidden"],
        default="zipf",
        help="Token-Routed strategy: zipf/modulo/round_robin/random lexical routing, or lsh_hidden semantic routing.",
    )
    parser.add_argument(
        "--lsh-bits",
        type=int,
        default=0,
        help="Number of random hyperplanes for lsh_hidden routing (0 = ceil(log2(num_experts))).",
    )
    parser.add_argument(
        "--lsh-from-layer",
        type=int,
        default=0,
        help="lsh_hidden routing applies only to layers >= this index; earlier layers stay lexical "
        "(h not yet semantic). 0 = LSH everywhere.",
    )
    parser.add_argument(
        "--lsh-threshold-mode",
        choices=["batch_median", "zero"],
        default="zero",
        help="Threshold for lsh_hidden bits. batch_median balances each training batch; zero is stable for inference.",
    )
    parser.add_argument(
        "--mlp-type",
        type=str,
        default=None,
        help="Override the MLP type (e.g. 'swiglu' for a dense baseline). Defaults to the profile's token_routed.",
    )
    parser.add_argument("--lexical-object-rank", type=int, default=16)
    parser.add_argument("--lexical-object-gate-init", type=float, default=0.1)
    parser.add_argument(
        "--disable-lexical-wrv-residual",
        action="store_true",
        help="Keep W/R/V lexical attention gates fixed at zero.",
    )
    parser.add_argument(
        "--disable-lexical-wrv-norms",
        action="store_true",
        help="Bypass and freeze W/R/V per-head read/write RMSNorm.",
    )
    parser.add_argument(
        "--lexical-wrv-hybrid",
        action="store_true",
        help=(
            "Inject the same lexical address into grouped reads and writes; "
            "values remain contextual."
        ),
    )
    parser.add_argument(
        "--lexical-wrv-gate-init",
        type=float,
        default=0.0,
        help="Initial shared lexical R/W gate before tanh (0 keeps legacy behavior).",
    )
    parser.add_argument(
        "--lexical-gqa-rank",
        type=int,
        default=16,
        help="Rank of the baseline-preserving lexical GQA score channel.",
    )
    parser.add_argument(
        "--lexical-gqa-gate-init",
        type=float,
        default=0.0,
        help="Initial lexical GQA score gate before tanh.",
    )
    parser.add_argument(
        "--disable-lexical-gqa-token-code",
        action="store_false",
        dest="lexical_gqa_use_token_code",
        help="Use only the learned lexical object as the lexical key source.",
    )
    parser.add_argument(
        "--tie-lexical-object-embeddings",
        action="store_true",
        help="Share one lexical modulation table across all transformer layers.",
    )
    parser.add_argument("--micro-num-experts", type=int, default=4)
    parser.add_argument("--micro-expert-width", type=int, default=16)
    parser.add_argument("--micro-expert-gate-init", type=float, default=0.1)
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
