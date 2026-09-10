# V2 dev reference verification

PASS: 8/8 references verified with Rocq 9.2.0, including both additions. All eight exact tokenizer paths and EOS remain allowed by constrained-v1 within 256 completion tokens. All eight Prompt v2 serializations plus their targets fit the existing 768-token limit.

| Row | Completion + EOS | Total sequence |
| --- | ---: | ---: |
| pd01_A | 28 | 575 |
| pd01_B | 29 | 587 |
| pd02_A | 30 | 599 |
| pd02_B | 31 | 611 |
| pd03_A | 29 | 612 |
| pd03_B | 30 | 624 |
| pd04_A | 32 | 668 |
| pd04_B | 33 | 664 |

All six new negative controls fail in Rocq: removal of the premise, removal of the IH, and replacement of the base premise with reflexivity, for each argument. These test usefulness in the authored scripts, not global proof minimality.

All four A/B reference-contract comparisons behave as intended: same variant MATCH, opposite variant MISMATCH. Both verified variants remain mathematically faithful to the same induction. These are reference QA checks, not model accuracy.

Schema, canonical alignment joins, features, local scope and public train/dev family checks pass. The original six dev lines are preserved byte-for-byte as the prefix of dev.jsonl. No weights were loaded, no model outputs generated, and no holdout or diagnostic proof files opened. No claim of exact duplicate exclusion against unopened holdout/diagnostic contents is made; see family_review.json.
