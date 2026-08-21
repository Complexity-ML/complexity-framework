# Two-dimensional full-shard SFT weighting

> **Historical 500M LoRA experiment.** This weighting system remains
> executable for reproduction, but it is not the recipe behind the released
> TR-HASH MoE 200M assistant. The 200M release uses unweighted Luciole 16-way
> full-parameter SFT.

This guide documents the Card Corpus V2 LoRA-SFT mixture policy. The
policy changes loss contribution without filtering, duplicating, resampling,
or republishing dataset rows.

## Contract

The canonical profile is
[`configs/sft_500m_32k_v2_balanced.yaml`](../configs/sft_500m_32k_v2_balanced.yaml).
It guarantees:

- `max_examples: all`: every indexed training example is visited once per epoch;
- `balance_by: none`: the loader does not construct a sampled task mixture;
- prompt and padding labels remain masked with `-100`;
- only visible assistant labels contribute to the weighted causal loss;
- targets cover every selected task or task×domain cell exactly once;
- group targets and each group's task targets sum to one;
- no derived task coefficient may exceed `max_task_loss_weight`.

The dataset stays immutable. The policy is resolved at training startup from
the supervised labels that remain visible after sequence truncation.

## The two dimensions

Dimension one assigns a global loss share to each capability group:

| Group | Global target |
|---|---:|
| Distilled reasoning | 20% |
| Natural conversation | 20% |
| Instruction and structured tasks | 60% |

Dimension two distributes each group target among semantic loss cells. Most
cells are complete task families. `casual_conversation` is heterogeneous, so
its social, reasoning, and instruction domains are routed into three explicit
cells. This prevents arithmetic rows from being mislabeled as conversational
loss while keeping every row in the shard.

```yaml
stages:
  - name: full-shard-weighted
    max_examples: all
    epochs: 1
    lr: 3.0e-6
    balance_by: none
    max_task_loss_weight: 30.0
    loss_cells:
      casual_social:
        task: casual_conversation
        domains: [social_greeting, social_help, social_gratitude]
    loss_groups:
      natural_conversation:
        target_share: 0.20
        tasks:
          - casual_social
          - conversation_empathy
        task_target_shares:
          casual_social: 0.70
          conversation_empathy: 0.30
```

## Coefficient derivation

For loss cell `c` inside group `g`:

```text
global_target(c) = group_target(g) * cell_target(c | g)
loss_weight(c)   = global_target(c) / raw_visible_token_share(c)
```

The raw share is measured from visible supervised tokens, not rows and not
untruncated response lengths. This matters because a long reasoning response
can contribute many more labels than a short conversational response.

Each example receives the coefficient of its resolved cell. The loss implementation
normalizes by visible weighted-token mass, keeping the overall loss scale
stable while changing the relative gradient contribution.

## Runtime audit

`derive_task_loss_weights()` reports:

- `loss_group_targets`;
- `loss_group_task_targets`;
- `task_visible_supervised_tokens`;
- `task_loss_weights`;
- `weighted_group_shares`;
- `weighted_task_shares`;
- `overweight_tasks` and `weights_within_cap`.

Planning and training reject the stage when:

- a selected task is missing from the target matrix;
- a target names a task absent from the shard;
- a group or nested target does not sum to one;
- a task has no visible supervised labels;
- measured weighted shares differ from their configured targets;
- a coefficient exceeds `max_task_loss_weight`.

On the audited 224,654-example V2 train shard, every check passes and the
largest coefficient is `25.6761x` for `critique_revision`, below the `30x`
limit. The complete measured table is in the root
[`README.md`](../README.md#two-dimensional-full-shard-loss-balancing-card-corpus-v2).

## Dry-run before training

Always resolve the matrix against the actual token shard before allocating
GPU time:

```bash
python -m scripts.run_sft_curriculum \
  --checkpoint artifacts/tr_hash_moe_500m_20b_hf \
  --sft-bin artifacts/complexity_card_corpus_v2_229026/tokenized/32k-v2 \
  --curriculum-config configs/sft_500m_32k_v2_balanced.yaml \
  --through-stage full-shard-weighted \
  --output-root artifacts/tr_hash_500m_32k_v2_weighted_lora \
  --tokenizer tokenizer \
  --world-size 8 \
  --batch-size 24 \
  --lora-rank 32 \
  --lora-alpha 32 \
  --dry-run
```

A valid dry-run must report `passed: true`, `shares_match_targets: true`, an
empty `overweight_tasks`, and `weights_within_cap: true`.

## Multi-epoch continuation

The base profile performs one complete pass. The tracked
[`configs/sft_500m_32k_v2_balanced_continuation.yaml`](../configs/sft_500m_32k_v2_balanced_continuation.yaml)
performs two additional passes with the same 2D matrix.

The continuation loads the selected epoch-one model as its base, applies a new
LoRA adapter, and starts a fresh optimizer and scheduler. It is a model-weight
continuation, not an exact optimizer-state resume. With eight ranks and 24
examples per rank, each pass is 1,171 optimizer steps; evaluation and
best-checkpoint selection occur at each complete-pass boundary.

Do not mix one-dimensional and two-dimensional loss policies inside a single
run. Restart the continuation from the same selected source checkpoint when
the weighting matrix changes.

## Tests

The regression coverage lives in:

- `tests/test_sft_500m_v2_balanced_curriculum.py` for the production matrix,
  full-shard invariant, exact task shares, and coefficient cap;
- `tests/test_sft_curriculum.py` for curriculum parsing and selection;
- `tests/test_sft_bin.py` for visible-label counting and weighted-loss behavior.

Run the focused suite with:

```bash
pytest -q \
  tests/test_sft_500m_v2_balanced_curriculum.py \
  tests/test_sft_curriculum.py \
  tests/test_sft_bin.py
```

## Publishing boundary

The 2D matrix is a framework and model-training property, not a dataset text
rewrite. Existing token shards receive a semantic `loss_metadata.jsonl`
sidecar; token IDs, labels, offsets, rows, and exposure remain unchanged.
Publish the resolved matrix, source dataset revision, selected checkpoint,
evaluation losses, and behavioral probes in the resulting Hugging Face model
card.
