# Framework refactor status

Local refactor completed and audited **9 October 2026**. The complete
repository suite passes in this worktree. The results below qualify the local
interfaces and CPU contracts; hardware and research limits are listed separately.

## Implemented

| Area | Change |
|---|---|
| Public API | consistent native model/config loading, strict new authoring, explicit missing-weight failures and reusable training |
| CLI | common data, training, checkpoint, evaluation and vision command trees; lightweight discovery |
| Vision | physical `complexity.vision` package; legacy imports retain class/module identity; four script paths are thin compatibility wrappers |
| Data | reusable preparation, projections, quota recipes, candidate staging, transactional sharding and curriculum/replay plans |
| Training | explicit full-parameter SFT, staged LoRA, CPU-tested DDP SFT and checked optimization/resume boundaries |
| Evaluation | held-out loss, continuation scoring and matched checkpoint comparisons with explicit source/runtime provenance |
| Routing | live blend controls synchronize execution paths; diagnostics use actual current capacity and contributions |
| Export | full DTensor gathering, rank-zero replicated export, runtime-control persistence, rollback on commit errors and explicit unsupported adapter behavior |
| Documentation | shorter workflow entry point, separate data/Python training guides and grouped vision guides |

The package contains reusable operations rather than restored experiment
launchers. See [architecture](framework-architecture.md), [workflows](workflows.md)
and [API](api.md).

## Validation

| Gate | Latest result | Scope |
|---|---|---|
| `make test` | **1,772 passed, 26 skipped**, 53.75 s | complete collection/execution, no excluded files |
| `make test-core` | **880 passed, 8 skipped**, 39.98 s | selected API/model/data/training contracts; overlaps the complete suite |
| `make lint` / `git diff --check` | passed | entire package, including tracked modules covered by ignore patterns |
| Final installed wheel acceptance outside checkout | passed | 83 help paths, console entrypoint, vision aliases/pickle, data operations, two packaged recipes and autonomous CPU Transformers logits parity |

The local test environment is macOS / Python 3.14.6 with CPU/Gloo. Logs are
`/private/tmp/complexity-framework-full-validation-20261009.txt` and
`/private/tmp/complexity-framework-core-validation-20261009.txt`. Counts overlap
and are not added together. Skipped tests are not qualified by these results.

The final wheel is `complexity_framework-0.3.0-py3-none-any.whl`, 936,151 bytes,
SHA-256 `6c4bb1981ad3cae0e5a75790cd009c58ae2f5460cb0ab89de233331cdd8b5e1f`.
It was installed into a separate venv with `--no-index --no-deps`, using the
host's existing dependencies through `--system-site-packages`. Acceptance
blocked network/GPU operations and required the package origin inside that
venv. This verifies installed-package integration, not a fresh dependency or
backend matrix. Report:
`/private/tmp/complexity-final-wheel-20261009-0qv0ba_2/acceptance.json`.

CPU/Gloo tests include update parity, optimizer/schedule/cursor/RNG resume,
rank agreement, interrupted checkpoint publication and export rollback. Tiny
native Torch models and offline fixtures qualify interfaces and numerical
contracts, rather than full-size model quality.

The former 200K/10K SFT quota assertion was introduced in cleanup commit
`1a90efd`, which removed its source. Available source revisions contain
100K/5K. The test now verifies that historical recipe and its provenance; no
new production allocation is inferred. The 50 historical SFT-bin test functions
now target native reader, causal-target, weighting, validation and resume
contracts. Same-position legacy targets and missing trained routes are replaced
by causal v2 supervision with EOS and explicit route-integrity failures.

## Preserved 1B recipe

`model.json` and `recipe.json` in `scripts/pretrain_1b_fsdp/` remain
byte-identical to this worktree's starting `HEAD`:

| File | SHA-256 |
|---|---|
| `recipe.json` | `8cb72103026174e88be6c271e2147cf39f70e70d30cf70b181b1761755c0cf8d` |
| `model.json` | `a614a990ddd587dcda3a4ec0cf0b998c3de18ea0806892fe5e9c29f9300c3a8d` |

Four ranks × microbatch 2 × context 16,384 × 534,057 updates =
**69,999,919,104 tokens**, the whole-update realization of the fixed **70B
pretraining budget**. Refinement/refit remains a separate phase. No 1B launch
or published-checkpoint modification was performed. Its original model,
optimizer, data cursor, schedule and rank RNG remain the resume authority.

## Limits

- CUDA/NCCL/BF16 DDP, optimized GPU kernels and a real four-rank published 1B
  restore/save/reload require later hardware qualification.
- Dataset acquisition, large parallel producers and independent remote
  publication/readback are outside the verified local preparation scope.
- The pipeline-parallel prototype has no native causal-LM loss integration;
  it now rejects training instead of optimizing a placeholder activation mean.
- Native in-place export rolls back Python-level commit errors; it is not a
  crash-atomic directory exchange. If filesystem rollback fails, original
  backups are retained. Sealed SFT epoch/checkpoint publication has its own
  transaction contract.
- The separate direct-SFT comparison has no completed new benchmark result.
  Local framework tests do not establish the lexical/semantic research claim.
- Validation is local. Remote CI and a public package release require separate
  verification.

## Reproduce

```bash
make lint
make test-core
make test
```

Both targets select the local Gloo loopback interface and limit CPU threads.
The training CI runs the complete suite on its configured CPU Python matrix;
local results do not establish that the remote CI has run.
