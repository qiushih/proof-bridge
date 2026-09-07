# Pilot v1 reference verification

**Rocq 9.2.0: 36/36 references PASS.** All passed the unchanged verifier, produced compiled proof artifacts and passed its global-assumption check.

| Split | PASS | FAIL | Theorem IDs | Same-theorem A/B groups |
| --- | ---: | ---: | ---: | ---: |
| train | 24 | 0 | 14 | 10 |
| dev | 6 | 0 | 3 | 3 |
| holdout | 6 | 0 | 3 | 3 |

All 36 exact reference token paths, including EOS, are allowed by frozen constrained-v1. The longest target has 33 tokens including EOS, within 256.

All 50 reference-removal ablations failed inside Rocq as required. These are construction quality checks on useful premise/IH steps, not generated samples or training targets. All 12 historical A/B references also reverified; their original files/results were preserved.

## Coverage by proof pattern

Counts overlap: one proof can exercise several patterns. A before/after count requires an actual `simpl` command in that order within the same branch.

| Pattern | Train / 24 | Dev / 6 | Holdout / 6 |
| --- | ---: | ---: | ---: |
| forward_rewrite | 10 | 3 | 3 |
| reverse_rewrite | 7 | 0 | 0 |
| rewrite_before_simpl | 5 | 0 | 0 |
| rewrite_after_simpl | 9 | 3 | 3 |
| premise_in_base_case | 8 | 0 | 6 |
| ih_via_rewrite | 7 | 3 | 3 |
| ih_via_congruence | 7 | 3 | 3 |
| premise_and_induction | 8 | 0 | 6 |

## Family and historical checks

36 unique row IDs, 20 theorem IDs, 16 same-theorem pairs and 13 conservative statement-equivalence classes. No exact statement/body duplicate and no cross-split family or detected statement equivalence.

Historical equivalent-statement overlap: train 16 rows, dev 2 rows, holdout 0 rows. Repeated statements are deliberate training/development targets, not independent observations. Historical family exposure is broader than exact overlap and is recorded in `family_audit.json`.

The six holdout references have fresh statements but belong to a historically exposed family. They were inspected only for construction and reference QA, then sealed. Subsequent checks use their committed byte hashes, without parsing or scoring them. Detailed holdout diagnostics remain under `reserved/`.

Draft reference preparation encountered the existing expression-depth limit and a simplification/rewrite-pattern error. The new draft examples were corrected before sealing; the verifier and historical proofs were not changed.

## Limits

All alignment and naturalness reviews are assistant reviews; human-reviewed remains false. The ablations do not prove global minimality. Most A/B contrasts are equivalent proof-step realizations of one mathematical strategy. This small release cannot establish broad informal-proof dependence or independent mathematical generalization.

**Training runs: 0. Model generations: 0.** Reference tokenization used the pinned local tokenizer only. See `validation.json` for final regression and replay evidence; no inference runner is part of this dataset release.
