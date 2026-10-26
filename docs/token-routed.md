# Deterministic token routing

The framework exposes one TR-HASH routing law:
`token_id_composed_modulo`. It combines the current original ID and the two
preceding IDs, selects two of four experts, and applies fixed equal weights.
The shared branch, routed attention heads and V channel binding are part of
the same architecture.

See [the complete formulas and cache contract](tr-hash-engine.md).

A saved checkpoint must explicitly declare this routing law. A static table
from a different routing algorithm cannot be relabeled as a compatible model.
Native and autonomous Transformers exports preserve the complete architecture.
