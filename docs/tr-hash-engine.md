# TR-HASH engine

TR-HASH has one architecture: a shared SwiGLU branch, four deterministic
experts, and routed GQA attention. There is no learned router or balancing loss.

## Architecture

The decoder uses four query heads, one key/value head, Q/K RMS normalization,
RoPE, and two selected attention heads per token. The selected heads are
multiplied by `sqrt(2)` before the output projection.

For hidden width `H`, the defaults are shared width `2H`, four expert widths
`H/4`, and fixed expert weights `(0.5, 0.5)`. The MLP output is:

```text
shared(x) + 2 * (0.5 * expert_a(x) + 0.5 * expert_b(x))
```

`intermediate_size` is the total stored routed width; it is divided by four.
Explicit widths resize this same architecture. The route law and branch
weights remain fixed.

## Composed causal routing

Let `t` be the original current token ID, `p1` and `p2` its two preceding IDs,
and `l` the absolute layer index starting at zero. Missing prefix IDs are zero.

```text
u = (p1 + 5*p2) % 9

if l == 1:
    a = ((t + 1) % 4) XOR (2 * ((u % 3) % 2))
    d = 1 + (t//4 + 2*(u//3) + 1) % 3
else:
    a = (t + l) % 4
    d = 1 + (t//4 + (1 + l%2)*(u%3) + l) % 3
b = (a + d) % 4
```

Attention uses the current ID. At physical layer one, its heads are
`a=(t//12)%4` and `b=(a+1)%4`. At other layers, `a=(t+l)%4`,
`d=1+(t//4+l)%3`, and `b=(a+d)%4`.

Before projecting V at physical layer zero, input channel `j` is read from
`(j + (H//4)*(t%4)) % H`. Other layers use the ordinary V input. The shared
MLP remains unmasked. Layer indices do not repeat on a two-layer cycle.

The pure implementation is `complexity.tr_hash.composed_routing`.

## Create a model

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    hidden_size=64, num_hidden_layers=2, vocab_size=256,
    intermediate_size=64, shared_intermediate_size=128,
    max_position_embeddings=64,
    use_custom_kernels=False, use_cggr=False,
)
model = ComplexityModel(config)
print(model.num_parameters())  # 110976
```

`routing_strategy` serializes as `token_id_composed_modulo`. Other routing
laws, expert counts, head counts and unequal expert weights are rejected.
Dense models use their separately named dense MLP.

## Caches and execution

Native cache entries contain `(K, V, last_two_original_ids)`. Generation must
retain routing history per batch row. `inputs_embeds` requires explicit
`routing_ids`; embeddings cannot recover original IDs. Domain models pass
unmodified patch, spatial or temporal IDs to this same routing law.

PyTorch is the numerical reference. CGGR dispatches the same dynamic routes
when available. A backend that cannot preserve context routing, selected heads
and V binding is rejected rather than substituted silently.

## Save and export

Native exports serialize the architecture, weights and canonical routing
anchor. Loading checks the explicit routing law and rejects a different anchor.
The anchor records zero-prefix routes; runtime routes still depend on history.

`complexity convert to-hf INPUT OUTPUT` creates an autonomous Transformers
bundle containing model code, composed routing helpers and the tensor weights.
Its cache keeps routing IDs through continuation, beam reordering and cropping.
The table-only custom MLX/vLLM layouts cannot represent this architecture and
raise an explicit error. Generic external serving endpoints remain separate.

## Evidence

The notebooks compare small models under matched data, masks, updates and
seeds. Parameter equality alone does not establish equal compute. Decoder
integration tests cover logits, gradients, save/load and cached continuation;
they do not establish quality at larger widths or in every modality.
