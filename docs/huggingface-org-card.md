---
title: Complexity-ML
emoji: 🧩
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
---

# Complexity-ML

Complexity Framework is a PyTorch research stack for deterministic
Token-Routed Mixture-of-Experts language models.

## Main architecture families

- **TR-GQA**: grouped-query attention + TR-MoE.
- **TR-MHA**: multi-head attention + TR-MoE.
- **TR-MoE**: a dense shared SwiGLU path plus deterministic token-selected
  experts.

Token identity selects expert parameters. The selected experts still transform
the current contextual hidden state.

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

## License

The framework repository uses CC BY-NC 4.0. Individual model repositories must
state their own license explicitly.
