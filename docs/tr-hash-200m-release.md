# TR-HASH text recipe

The framework now constructs only the composed TR-HASH architecture described
in [the engine contract](tr-hash-engine.md). The text factory
`complexity.training.text_pretraining.make_200m_config` uses four query heads,
one KV head, four experts, top-2 equal weights and shared/routed scales 1/2.
Its width and layer count are explicit recipe settings; measure the resulting
parameter count rather than inferring it from the recipe name.

This code update does not produce pretrained weights or a new large-model
quality evaluation. A checkpoint from another routing law cannot be renamed
or loaded as this architecture. Fresh model construction, stage weight
initialization and exact optimizer continuation are distinct actions.

Use `ModelConfig.save`, native `save_pretrained`, and the autonomous
Transformers exporter for newly trained canonical models. Corpus proofs,
tokenizers, masks, schedules and measured metrics belong to their actual run.
Local notebook results remain small development experiments and do not prove
quality or efficiency at this recipe's scale.
