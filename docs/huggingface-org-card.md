---
title: AETHORIA-AI
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# AETHORIA-AI

Complexity Framework is a PyTorch research stack for deterministic
Token-Routed Mixture-of-Experts language models.

## Main architecture families

- **TR-GQA**: grouped-query attention + TR-MoE.
- **TR-MHA**: multi-head attention + TR-MoE.
- **TR-MoE**: a dense shared SwiGLU path plus deterministic token-selected
  experts.

Token identity selects expert parameters. The selected experts still transform
the current contextual hidden state.

## Current language-model release

The reference lineage is **TR-HASH MoE 200M**: 201.2M parameters, a completed
129.996B-token base run, an interrupted 32.069B-token full-parameter
refinement, and three epochs of full-parameter instruction SFT. The promoted
epoch-2 checkpoint reaches 68.82% PIQA accuracy and 69.31% normalized accuracy
under the published zero-shot continuation protocol.

- [Base](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-130B)
- [Refinement](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement)
- [Full SFT](https://huggingface.co/AETHORIA-AI/TR-HASH-MoE-200M-160B-SFT)
- [Live chat](https://www.complexity-ai.fr/ai-lab)
- [Release paper](https://www.complexity-ai.fr/papers/tr-hash-200m-multi-hash-routing.pdf)

The older 492.1M/20B and LoRA artifacts are historical experiments and are not
the default assistant release.

## Research standards

Projects distinguish implementation, pilots, controlled comparisons, and
planned runs. Model cards should report:

- total and active parameters;
- tokenizer and context length;
- data and token budget;
- seeds;
- evaluation split;
- hardware and runtime;
- license and intended use;
- known limitations.

## Repository

<https://github.com/Complexity-ML/complexity-framework>

## Hugging Face organization

<https://huggingface.co/AETHORIA-AI>

## License

The framework repository uses CC BY-NC 4.0. Individual model repositories must
state their own license explicitly.
