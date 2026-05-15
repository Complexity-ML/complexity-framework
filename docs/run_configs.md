# Run Configs

Research runs can be launched from YAML configs so the command, run summary, and
resume checks stay reproducible.

Example:

```bash
torchrun --nproc_per_node=8 -m complexity.training.o200k_pretrain \
  --config configs/run_configs/100m_o200k_tr_30b_b300.yaml
```

CLI flags override YAML values:

```bash
torchrun --nproc_per_node=8 -m complexity.training.o200k_pretrain \
  --config configs/run_configs/100m_o200k_tr_30b_b300.yaml \
  --steps 100 --run-name smoke-yaml
```

At launch, rank 0 writes `runs/<run-name>/run_config.json` with:

- CLI/YAML args after defaults and overrides
- model config
- git commit
- parameter count
- tokens per step and total token budget

When resuming, the runner compares the saved `run_config.json` against the new
launch config and refuses mismatches in training-critical fields such as model
shape, tokenizer, dataset, batch size, sequence length, optimizer LR, and routing
options. Operational fields such as `save_steps`, `log_steps`, `eval_steps`,
`run_name`, and `save_dir` may change.

Use `--force-resume` only for intentional mismatches.

## Routing Strategies

The o200k Token-Routed runner supports:

- `zipf`: existing deterministic Zipf/frequency-balanced routing. This remains
  the default.
- `zipf_token_class`: optional class-balanced routing. It classifies tokenizer
  entries into coarse buckets such as whitespace, digits, ASCII words,
  punctuation, non-ASCII, and mixed tokens, then greedily balances both total
  token load and per-class load across experts. This keeps routing static and
  deterministic while reducing class skew inside experts.

Example:

```bash
torchrun --nproc_per_node=8 -m complexity.training.o200k_pretrain \
  --config configs/run_configs/100m_o200k_tr_30b_b300.yaml \
  --routing-strategy zipf_token_class \
  --run-name 30b-100m-o200k-tr-token-class
```
