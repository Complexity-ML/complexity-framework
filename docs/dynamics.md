# Historical Mu-Guidance control

Mu-Guidance is an optional contextual state passed from one decoder layer to
the next. It remains implemented for historical reproduction and ablation
work.

It is **not** part of the current definition of TR-GQA, TR-MHA, or TR-MoE.

## Enable it

```python
from complexity import ComplexityModel, ModelConfig

config = ModelConfig(
    attention_type="gqa",
    mlp_type="tr_hash_engine",
    use_mu_guidance=True,
)
model = ComplexityModel(config)
```

The runner exposes:

```bash
--use-mu-guidance
--mu-clamp
--mu-norm
--mu-alpha-init 1.0
--mu-init-value 0.0
--mu-context-min -2.0
--mu-context-max 2.0
```

## Computation

After the MLP, each enabled layer produces:

```text
mu_contextual = clamp(mu_parameter) + alpha * projection(hidden_state)
```

The next supported attention layer receives this value as `mu_prev`.
Contextual clamping and RMS normalization are optional.

## Evidence status

Older documentation associated Mu-Guidance with unsourced aggregate loss and
serving claims. Those claims are not carried into the current documentation.
Any new Mu comparison must be parameter-, token-, seed-, and protocol-matched.

## Compatibility

Mu-Guidance adds state to the layer interface. A serving runtime must implement
that state explicitly; the external inference client does not provide it.
