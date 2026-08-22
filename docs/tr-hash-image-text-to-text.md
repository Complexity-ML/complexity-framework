# TR-Hash Image + Text to Text

> **Experimental modality.** The architecture is implemented, but it is not a
> released 200M language-model checkpoint and does not inherit its PIQA result.

This model is a direct vision-language generator:

```text
image ──> ViT ──> 64 visual prefix tokens ──┐
                                             ├─> causal TR-Hash decoder ──> text
prompt ──────────> text token embeddings ───┘
```

It does not generate an intermediate caption. Visual tokens and prompt tokens
share one causal sequence, and the language-model loss is applied only to the
answer positions.

## Default architecture

- 224 px RGB input with 16 px patches;
- six-layer, width-384 vision tower;
- learned cross-attention resampler producing 64 visual tokens;
- 16-layer, width-768 GQA decoder with 12 query and four KV heads;
- canonical TR-Hash MLP engine with four experts, deterministic top-2 routing,
  and a shared SwiGLU branch;
- 32,000-token vocabulary and 2,048-position context;
- 190,568,960 total parameters.

The visual prefix receives stable synthetic routing IDs. Text positions retain
their real token IDs, so the decoder uses the same deterministic routing
contract as a text-only TR-Hash model.

## Python contract

```python
import torch

from complexity.generative.vision_language import TRHashImageTextToText

model = TRHashImageTextToText()
pixels = torch.randn(2, 3, 224, 224)
input_ids = torch.randint(0, 32_000, (2, 128))
labels = input_ids.clone()
labels[:, :32] = -100  # prompt positions are context, not targets

output = model(pixels, input_ids, labels=labels)
output["loss"].backward()
```

For initial alignment, an image-caption corpus can use a fixed user request
such as `Describe this image.` and supervise the caption. The aligned model is
then refined once over that exact same corpus with fresh optimization state.
Only after refinement does conversational SFT introduce image-grounded
questions, answers, corrections, comparisons, and multi-turn follow-ups.
Caption-only data is not enough to teach reliable visual dialogue, and new
dialogue data must not be labelled refinement.
