# Semantic Expert Audit - 100M TR @ 65K
Heuristic audit on decoded context windows routed by the primary semantic LSH expert. These are tendencies, not hard labels.
## Aggregate Expert Tendencies
### Expert E0
- Routed tokens sampled: 18748
- Dominant probes: proper_entities=1651, math_numbers=570, narrative_social=262, list_structure=157, dialogue_quotes=67
- Example contexts:
  - token ` Independent`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of
  - token `\n`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and
  - token `For`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the
  - token ` Jane`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought

### Expert E1
- Routed tokens sampled: 22743
- Dominant probes: proper_entities=1646, math_numbers=474, narrative_social=267, list_structure=155, dialogue_quotes=64
- Example contexts:
  - token ` the`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to
  - token ` romance`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to choose. Elizabeth
  - token ` scandal`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to choose. Elizabeth’s refusal
  - token `\n`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and

### Expert E2
- Routed tokens sampled: 19037
- Dominant probes: proper_entities=1607, math_numbers=532, narrative_social=270, list_structure=136, dialogue_quotes=75
- Example contexts:
  - token ` Jane`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought
  - token ` Jane`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to choose. Elizabeth’s refusal of Mr
  - token ` what`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to choose. Elizabeth’s refusal of Mr. Collins offer of marriage
  - token `For`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the

### Expert E3
- Routed tokens sampled: 21392
- Dominant probes: proper_entities=1610, math_numbers=517, narrative_social=253, list_structure=157, dialogue_quotes=62
- Example contexts:
  - token `The`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence
  - token ` all`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom
  - token ` love`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence of thought and the freedom to choose
  - token `The`: The Independent Jane For all the love, romance and scandal in Jane Austen’s books, what they are really about is freedom and independence. Independence

## Layer Notes
### Layer 2
- E0: share=0.202, unique_tokens=689, proper_entities=225, math_numbers=72, narrative_social=36
- E1: share=0.232, unique_tokens=961, proper_entities=216, math_numbers=60, narrative_social=36
- E2: share=0.240, unique_tokens=1092, proper_entities=197, math_numbers=64, narrative_social=30
- E3: share=0.326, unique_tokens=1512, proper_entities=170, math_numbers=44, narrative_social=30

### Layer 3
- E0: share=0.269, unique_tokens=1009, proper_entities=230, math_numbers=77, narrative_social=33
- E1: share=0.235, unique_tokens=983, proper_entities=211, math_numbers=40, narrative_social=30
- E2: share=0.274, unique_tokens=1206, proper_entities=214, math_numbers=56, narrative_social=29
- E3: share=0.222, unique_tokens=1071, proper_entities=194, math_numbers=45, narrative_social=38

### Layer 4
- E0: share=0.355, unique_tokens=1427, proper_entities=205, math_numbers=86, narrative_social=30
- E1: share=0.276, unique_tokens=1250, proper_entities=181, math_numbers=66, narrative_social=35
- E2: share=0.221, unique_tokens=1062, proper_entities=206, math_numbers=58, narrative_social=34
- E3: share=0.149, unique_tokens=750, proper_entities=223, math_numbers=56, narrative_social=34

### Layer 5
- E0: share=0.211, unique_tokens=771, proper_entities=211, math_numbers=87, narrative_social=35
- E1: share=0.240, unique_tokens=1387, proper_entities=226, math_numbers=61, list_structure=27
- E2: share=0.253, unique_tokens=925, proper_entities=222, math_numbers=62, narrative_social=48
- E3: share=0.297, unique_tokens=1511, proper_entities=211, math_numbers=70, narrative_social=26

### Layer 6
- E0: share=0.223, unique_tokens=778, proper_entities=205, math_numbers=51, narrative_social=31
- E1: share=0.346, unique_tokens=1398, proper_entities=195, narrative_social=43, math_numbers=17
- E2: share=0.172, unique_tokens=935, proper_entities=151, math_numbers=57, narrative_social=27
- E3: share=0.259, unique_tokens=1557, proper_entities=219, math_numbers=83, narrative_social=30

### Layer 7
- E0: share=0.189, unique_tokens=956, proper_entities=207, math_numbers=109, narrative_social=37
- E1: share=0.302, unique_tokens=1567, proper_entities=181, math_numbers=73, narrative_social=31
- E2: share=0.234, unique_tokens=958, proper_entities=241, math_numbers=86, narrative_social=33
- E3: share=0.275, unique_tokens=1163, proper_entities=189, math_numbers=66, narrative_social=25

### Layer 8
- E0: share=0.140, unique_tokens=675, proper_entities=192, math_numbers=52, narrative_social=29
- E1: share=0.264, unique_tokens=1379, proper_entities=214, math_numbers=74, narrative_social=30
- E2: share=0.283, unique_tokens=1100, proper_entities=210, math_numbers=68, narrative_social=40
- E3: share=0.313, unique_tokens=1549, proper_entities=195, math_numbers=38, narrative_social=35

### Layer 9
- E0: share=0.243, unique_tokens=1063, proper_entities=176, math_numbers=36, narrative_social=31
- E1: share=0.327, unique_tokens=1551, proper_entities=222, math_numbers=83, narrative_social=45
- E2: share=0.183, unique_tokens=857, proper_entities=166, math_numbers=81, narrative_social=29
- E3: share=0.248, unique_tokens=1276, proper_entities=209, math_numbers=115, narrative_social=35
