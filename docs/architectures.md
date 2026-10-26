# Model architecture

TR-HASH is the framework's deterministic shared-plus-expert architecture.
The complete contract is defined in [TR-HASH engine](tr-hash-engine.md).

| Component | Contract |
|---|---|
| Attention | four query heads, one KV head, Q/K RMSNorm and RoPE |
| Attention routing | two deterministic heads per current original ID |
| V binding | quarter-channel input rotation at physical layer zero |
| MLP | shared SwiGLU plus four experts, two routes per token |
| Context | current ID and two previous IDs, zero prefix at sequence start |
| Mixture | equal 0.5/0.5 expert weights; shared scale 1, routed scale 2 |
| Defaults at width H | shared width 2H, total routed width H |
| Serialization | explicit token_id_composed_modulo strategy and routing anchor |

`ComplexityModel` passes original routing IDs into every decoder layer. Cache
entries preserve their two-ID prefix. Vision, audio, video and sensor models
use their raw structural IDs with the same routing, head selection and V
binding. Task-specific cross-attention keeps its own task interface.

Dense controls are separately named models. They do not select a different
TR-HASH architecture. Parameter and active-compute claims must be measured
after construction.

The training sequence is pretraining, optional full-parameter refinement on
the same unique corpus, then supervised fine-tuning on its own data. SFT does
not acquire pretraining replay implicitly. A new stage starts its own declared
optimizer/schedule; exact continuation instead restores optimization state.

Related pages: [module boundaries](framework-architecture.md),
[training workflows](workflows.md), [configuration formats](run_configs.md).
