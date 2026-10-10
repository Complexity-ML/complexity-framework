# Framework refactor status

The local refactor was completed on **9 October 2026**. Lightweight installation
and the packaged CPU/MPS workflow were verified on **10 October 2026**. The
complete repository suite passes; hardware and research limits remain separate.

## Implemented

| Area | Change |
|---|---|
| Public API | consistent native model/config loading, strict new authoring, explicit missing-weight failures and reusable training |
| Installation | lightweight native core; explicit `hf`, `tokenizers`, `data` and `tracking` extras; Torch installed separately |
| CLI | common data, training, checkpoint, evaluation and vision command trees; lightweight discovery |
| Vision | physical `complexity.vision` package; COCO/ONNX tools and configs integrated; 13 former scripts removed; legacy model imports retain class/module identity |
| Data | reusable preparation, projections, quota recipes, candidate staging, transactional sharding and curriculum/replay plans |
| Training | explicit full-parameter SFT, staged LoRA, CPU-tested DDP SFT and checked optimization/resume boundaries |
| Local workflow | packaged `train demo`: synthetic fixtures, pause/resume, independent reference, held-out evaluation and native export/reload on CPU or MPS |
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
| `make test` | **1,801 passed, 31 skipped**, 52.53 s | complete collection/execution, no excluded files |
| `make lint` / `git diff --check` | passed | entire package, including tracked modules covered by ignore patterns |
| Installed-wheel acceptance outside checkout | **5 passed in each base and vision profile** | isolated wheel; 90 discovery help paths; offline CPU demo; 12 Vision runtime helps, report merging and invalid-report rejection with repository scripts blocked |
| Installed-wheel MPS demo at `e841141` | passed | unchanged training path; real Apple Metal continuation and export/reload on the small synthetic fixture |

The repository suite uses macOS / Python 3.14.6 / Torch 2.13.0 with CPU/Gloo.
Log: `/private/tmp/complexity-vision-integration-tests-20261010.log`. Five skips
belong to the installed-wheel tests, which run separately in the isolated
environment; other skipped tests are not qualified by the suite result.

The installed wheel is `complexity_framework-0.3.0-py3-none-any.whl`, 986,233 bytes,
SHA-256 `fe1ebcddd1de158916ce49274bb59b74dd6c946d358db82d045b867f73855b29`.
It was built from a frozen source copy and installed with its resolved minimal
dependencies into a venv with `include-system-site-packages = false`.
Transformers, tokenizers, tiktoken, datasets, Hub, W&B and TensorBoard are absent
in the base profile. The vision profile adds `[export]`, including Hub and
memory measurement, while keeping text/data/tracking extras absent.
Python 3.14.6 / Torch 2.14.1 exercised both profiles on CPU; acceptance blocked
network access, repository-script imports and heavy imports during discovery.
Evidence: `/private/tmp/complexity-vision-integration-20261010-h5xzrk74/qualification.json`.

The demo uses 30,912 parameters, four updates, dropout and a fixed schedule.
CPU continuation is exact. The earlier MPS qualification at `e841141` uses tolerance `1e-6`, with observed maximum
logit difference `1.7881393432617188e-7`; export/reload difference is zero.
This is a tiny native-model/backend contract, not large-run or quality evidence.
Linux CI separately installs base profiles on Python 3.10/3.11/3.12 and full
profiles on 3.11/3.12 and the vision profile on 3.11, retaining wheel, dependency
and acceptance artifacts. Remote results are recorded by the commit's GitHub jobs.

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
- Local and remote CI results have separate evidence. A public package release
  requires its own verification and publication.

## Reproduce

```bash
make lint
make test-core
make test
```

Both targets select the local Gloo loopback interface and limit CPU threads.
The training CI runs the complete suite and the isolated installed-wheel gate;
check the selected commit's GitHub run for its remote result.
