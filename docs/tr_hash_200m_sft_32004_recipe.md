# TR-HASH MoE 200M: 32,004-token full SFT recipe

This is the canonical post-tokenizer-migration recipe. It replaces the legacy
32,000-token SFT workflow; it does not modify the completed pretraining or the
published Refinement weights.

## Required lineage

1. Initialize from `AETHORIA-AI/TR-HASH-MoE-200M-160B-Refinement` after its
   embedding table has been extended to 32,004 rows.
2. Use `AETHORIA-AI/TR-HASH-Tokenizer-32K` with the following immutable IDs:

   | Token | ID |
   |---|---:|
   | `<|think_start|>` | 32000 |
   | `<|think_end|>` | 32001 |
   | `<|final_start|>` | 32002 |
   | `<|final_end|>` | 32003 |

3. Retokenize from canonical JSONL text. Never remap or reuse a 32,000-token
   binary shard.
4. Never resume a legacy SFT checkpoint: its four new embedding rows were not
   supervised.

## Dataset contract

Every assistant turn has one ordered envelope:

```text
<|think_start|>optional verified reasoning<|think_end|><|final_start|>answer<|final_end|>
```

Ordinary instruction, conversation and code responses keep an empty `think`
span and place the complete response in `final`. A non-empty `think` span is
used only when the source exposes a trustworthy reasoning/final split. No
hidden reasoning is invented from a plain answer.

The release audit must prove that:

- the tokenizer and model vocabulary are 32,004;
- all pre-existing token IDs remain unchanged;
- every retained example supervises IDs 32000, 32001, 32002 and 32003 exactly
  once in its final assistant completion;
- all tags close in order and no tag appears in a non-assistant turn;
- examples are never token-truncated;
- ARC-Easy, ARC-Challenge, PIQA, GSM8K and HellaSwag overlaps are absent;
- raw JSONL, binary shards and indexes have published SHA-256 values.

## Full-parameter SFT

The production launcher is `scripts/vast_sft_200m_32004_full_3e.sh`. The
validated baseline remains three full-parameter epochs with BF16, packed 2,048
token sequences, Liger fused CE, LR `2e-5`, weight decay `0.1`, betas
`(0.9, 0.95)`, and 3% warmup without a scheduler reset between epochs.

The launcher fails closed if it sees a 32,000-token checkpoint, stale training
artifacts, missing special-token supervision, truncation, or a dataset that is
not release-ready.

## Reasoning follow-up

The reasoning corpus is packaged separately under the same 32,004-token
contract. It is not silently mixed into the first full SFT. Any reasoning
follow-up must begin from an evaluated 32,004-token full-SFT checkpoint and be
promoted only after matched PIQA, ARC causal-continuation retention, generative
ARC parsing, and free-generation checks.
