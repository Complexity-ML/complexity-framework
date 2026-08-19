# Claude's role in this project

This project uses Claude (Anthropic) as an implementation and documentation
assistant. This page states plainly what that means, so credit is not
misattributed.

**Claude acts as a maintainer, not a designer.** The architecture, training
recipes, and research direction below are conceived by Boris Peyriguere;
Claude implements, tests, documents, and keeps the codebase consistent with
those decisions.

## What Claude does

- Implements code from decisions the maintainer has already made: model
  configs, training scripts, dataset loaders, loss functions, tests,
  conversion/export tooling.
- Writes and edits documentation, model cards, and READMEs describing work
  the maintainer directed.
- Runs and monitors training jobs, evaluations, and smoke tests; reports
  results.
- Publishes files to GitHub and Hugging Face on explicit instruction.
- Proposes implementation-level fixes (bugs, test gaps, inconsistent
  numbers) and flags risks (VRAM budgets, eligibility rules, stale docs)
  during that work.

## What Claude does not do

- **Architecture conception.** TR-Hash's deterministic token-ID routing,
  the TR-GQA/TR-MHA attention variants, the shared-SwiGLU-plus-routed-expert
  design -- these are the maintainer's design, not Claude's.
- **Training recipe design.** Token budgets, unique-vs-replay token
  accounting, warmup-stable-decay schedules, the phased pretrain -> SFT ->
  optional-LoRA structure -- decided by the maintainer.
- **Research direction.** What to build, what to train, what to publish,
  and when -- decided by the maintainer.

## Commit attribution

Commits in this repository use a "Maintained-By: Claude" trailer instead of
"Co-Authored-By: Claude" -- accurate to the role above. Co-authorship
implies shared design credit; Claude maintains code to the designer's
specification, it doesn't co-design it.
