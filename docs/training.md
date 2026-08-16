# Training

Complexity Framework currently exposes a LoRA-only instruction-tuning path for
the 500M native-32K TR-Hash model. The former `cf-o200k-pretrain` driver and its
Dense comparison architecture were removed; tracked legacy run configurations
are evidence and planning records, not directly executable training jobs.

## Supported text-training path

The low-level runner is `scripts/sft_500m_32k_tr.py`. Production runs should
normally use `scripts/run_sft_curriculum.py`, which resolves a tracked YAML
stage, audits the selected shard, launches DDP, and records the selected
checkpoint.

The runner requires a positive LoRA rank. Full-parameter SFT is deliberately
disabled. Supported LoRA targets include attention projections, shared MLP
projections, and TR-Hash expert projections.

```text
q_proj, k_proj, v_proj, o_proj
shared_gate, shared_up, shared_down
expert_gate, expert_up, expert_down
```

## Binary SFT shard

A native shard contains indexed `uint32` input IDs and `int32` labels:

```text
tokenized/32k-v2/
├── manifest.json
├── train/
│   ├── input_ids.bin
│   ├── labels.bin
│   ├── examples.jsonl
│   └── sft.idx.json
└── eval/
    ├── input_ids.bin
    ├── labels.bin
    ├── examples.jsonl
    └── sft.idx.json
```

Prompt and padding labels are `-100`. Only assistant labels contribute to the
loss. `--require-release-ready` rejects a shard whose manifest has not passed
the release-quality gates. The eval shard is finite and is never repeated
during one evaluation.

## Canonical Card Corpus V2 run

The canonical configuration visits the full training shard and weights loss
through a group-by-task matrix:

```bash
python -m scripts.run_sft_curriculum \
  --checkpoint artifacts/tr_hash_moe_500m_20b_hf \
  --sft-bin artifacts/complexity_card_corpus_v2_229026/tokenized/32k-v2 \
  --curriculum-config configs/sft_500m_32k_v2_balanced.yaml \
  --through-stage full-shard-weighted \
  --output-root artifacts/tr_hash_500m_32k_v2_229026_clean_lora_r32 \
  --tokenizer tokenizer \
  --world-size 8 \
  --batch-size 24 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --lora-dropout 0.05 \
  --lora-targets q_proj,k_proj,v_proj,o_proj,shared_gate,shared_up,shared_down,expert_gate,expert_up,expert_down \
  --expert-lr-multiplier 0.25 \
  --early-stopping-patience 2 \
  --dry-run
```

Remove `--dry-run` only after the exposure and loss-target audits pass. See
[Two-dimensional full-shard SFT weighting](sft-full-shard-2d-weighting.md) for
the complete matrix, formulas, coefficient cap, and continuation profile.

## Evaluation and checkpoint selection

For binary shards, use:

- `--eval-at-start` to measure the source checkpoint before optimization;
- `--eval-steps N` to evaluate on complete-pass boundaries;
- `--eval-batches 0` to consume the complete held-out shard;
- `--save-best` to retain validation-selected checkpoints;
- `--early-stopping-min-epochs` to prevent selection before a full pass;
- `--early-stopping-patience` for consecutive non-improving evaluations.

Training loss and matched eval loss need not share the same weighting. Report
the weighting policy with every loss value. Eval NLL alone is not a release
gate for an assistant model; also run behavioral probes for greeting, factual
recall, arithmetic, safety, instruction following, multiturn continuity, and
repetition.

## Resume and continuation

`--resume` is an exact continuation contract. It restores model, LoRA adapter,
optimizer, scheduler, data cursor, distributed RNG state, and best-eval state.
Training-critical arguments and world size must match the checkpoint.

Changing the epoch budget or loss matrix is a new stage, not an exact resume.
Load the selected merged checkpoint through `--checkpoint`, start a fresh LoRA
adapter and optimizer, and write to a new output root. The tracked two-epoch
continuation profile follows this model-weight continuation rule.

## Precision and acceleration

- `--bf16` enables backend-appropriate BF16 autocast;
- `--grad-ckpt` reduces activation memory;
- `--loss-chunk-tokens` bounds tied-head loss memory;
- `--sft-fp32-loss` computes the chunked loss in FP32;
- `--use-custom-kernels auto` enables tested CUDA/Triton paths on NVIDIA and
  leaves ROCm on the conservative fallback unless explicitly allowed;
- `--expert-lr-multiplier` scales only expert LoRA factors.

See [GPU and dispatch paths](cuda.md) for backend policy and reporting.

## DDP accounting

`scripts/run_sft_curriculum.py --world-size N` launches one process per GPU via
`torch.distributed.run`. `--batch-size` is per rank. The curriculum computes
exact steps per epoch from selected examples, per-rank batch size, and world
size; do not estimate the evaluation boundary manually.

## Run outputs

Rank zero writes metrics under `runs/<run-name>/metrics.csv`. Checkpoint roots
contain regular step checkpoints, optional `best/` checkpoints, `best.json`,
and the curriculum orchestrator's `curriculum-state.json`. Publish the selected
checkpoint path rather than assuming the last step is best.

## Planning tools

The remaining planners perform arithmetic and validation only:

```bash
cf-plan-run --tokens 20B --gpus 8 --batch-size 24 --seq-len 512
cf-plan-cluster --config configs/run_configs/8b_o200k_tr_32t_gb300_4608.yaml
```

They do not launch pretraining jobs.

## Verification

```bash
pytest -q \
  tests/test_sft_500m_v2_balanced_curriculum.py \
  tests/test_sft_curriculum.py \
  tests/test_sft_bin.py
```

For a publishable run, record the framework commit, source checkpoint, dataset
revision, tokenizer, exact loss matrix, LoRA configuration, optimizer schedule,
world size, hardware, precision, selected checkpoint, held-out metrics, and
behavioral probe results.
