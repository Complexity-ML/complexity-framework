# Use of generative AI tools

> **Compatibility filename.** This page was originally named for one assistant.
> The policy now applies to every generative AI coding or writing tool used on
> the project, including Claude and Codex.

The architecture, training objectives, release decisions, and research claims
are directed and approved by Boris Peyriguere. Generative AI tools assist with
implementation, testing, documentation, consistency checks, and artifact
packaging; they are not credited as the research designer or accountable
author.

## Permitted assistance

- implement maintainer-specified model, data, training, evaluation, and export
  changes;
- run tests and compare outputs against explicit invariants;
- edit documentation and model cards from measured artifacts;
- identify inconsistent dimensions, metrics, links, or release labels;
- prepare commits and publish them only when explicitly authorized;
- disclose uncertainty and distinguish inference from measured evidence.

## Human responsibility

The maintainer remains responsible for:

- architecture and experimental design;
- dataset provenance and license compliance;
- rental, training, and deployment authorization;
- benchmark protocol and contamination risk;
- the truthfulness of public claims;
- final review of code, weights, model cards, papers, and releases.

## Evidence rules

AI-generated prose is not evidence. A numerical claim must be checked against a
tracked log, metric file, checkpoint, manifest, or deterministic computation.
Tools must not turn a planned run into a completed result, call refinement SFT,
describe LoRA as full-parameter training, infer kernel use from a requested
flag, or promote a checkpoint without the stated selection protocol.

## Disclosure

Release papers should state material use of generative AI tools. The current
200M paper discloses assistance with language editing, consistency checks, and
PDF packaging and states that the author reviewed the numerical claims and
takes responsibility for the final text.
