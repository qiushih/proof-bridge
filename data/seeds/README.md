# First 30 arithmetic seed proof pairs

This release contains 30 individually selected and authored pairs: 6 proofs by definition, 9 using an equality premise, and 15 using one natural-number induction. They fit the existing addition-only grammar and use no global helper lemmas. Every proof passed the pinned Rocq 9.2.0 / Stdlib 9.1.0 verifier, including its axiom check, in an independent temporary environment. No model training or paraphrase expansion was performed.

These are **assistant-curated examples, not independently human-authored proofs**. Each informal argument was reviewed against its formal steps by the authoring assistant. The recorded kernel checks establish the formal propositions; they do not certify English alignment. The corpus intentionally contains related exercises and common proof patterns, so 30 pairs should not be interpreted as 30 independent mathematical families.

| File | Purpose |
| --- | --- |
| `curated.json` | Source: statements, proof paragraphs, unchanged steps, canonical bodies, features, families, and metadata |
| `pairs.jsonl` | 30 complete records, one JSON object per line, compatible with the example's `input` / `target` structure |
| `verification_report.json` | Actual results, per-method/group counts, compiler version, and checksums |
| `families.json` | Explicit family membership and rationale for keeping related variants together |
| `SCHEMA.md` | Full `seed-0.2` field definitions, derivations, provenance, and grouping policy |
| `../../scripts/verify_seeds.py` | Build records or check their integrity and rerun Rocq |

From the repository root:

```sh
python3 scripts/setup_rocq.py --check
python3 scripts/verify_seeds.py --audit
python3 scripts/verify_seeds.py --check
python3 -m unittest discover -s tests -v
```

To refresh exported pairs after an intentional source edit:

```sh
python3 scripts/verify_seeds.py --write
```

`--write` runs all 30 canonical proofs and the three original bodies whose canonical versions were normalized. It stores actual outcomes, including failures if any occur, and updates source provenance from the canonical results. It exits nonzero unless all canonical and original checks pass. `--check` verifies stored artifacts against current inputs before running the same compilations; it does not overwrite timestamps or results. `--audit` runs schema, duplicate, and family checks without compilation or writes. Changing the source, environment lock, verifier, build/schema scripts, or family manifest requires refreshing the evidence. Each compilation has the existing verifier's default ten-second timeout.

Schema `seed-0.2` exposes `informal_statement` (the theorem), `informal_proof` (the joined proof paragraph), `steps` (unchanged text/code alignment), and `proof_body` (the canonical candidate) directly in both source and exported records. It adds `argument_features`, `generation_family`, and explicit `metadata.provenance`. See [SCHEMA.md](SCHEMA.md) for every field and its validation rules. The existing `input`/`target` fields remain as compatibility copies.

Only three canonical bodies changed: seeds 009, 022, and 027 use `exact` for a directly matching local hypothesis. Their original `apply` lines remain in `steps`, and `metadata.proof_normalizations` records each replacement. All 30 original statements and step objects are unchanged. Alignment line numbers refer to the canonical `proof_body`; normalization preserves the lines. The full source in `target.rocq_source` is exactly the source identified by the verification hash and uses the verifier's renamed local variables. Each candidate is verified in isolation, so an earlier seed cannot supply a helper to a later one.

All records have `split: unassigned`. Seven `generation_family` values refine five conservative `split_group` boundaries. The original reflexivity/definition groups and successor/associativity groups were merged where variants overlap; prior labels are kept in metadata. Always keep the entire split group together in future partitions. The family manifest preserves connected variants instead of splitting them by tactic choice. There are 30 distinct statements modulo binder renaming, but only 26 distinct definitional forms: 001/002, 005/006, 017/022, and 018/026 are equivalent pairs and share families. Seed 016 is the right-zero example already present in `examples/add_zero_right.json`; `metadata.related_existing_example` records that link. Do not count it as another independent held-out problem.

In the table below all variables range over natural numbers, `S` denotes successor, and the equality before `->`, when present, is an explicit assumption. Every entry passed.

| ID | Statement | Method |
| --- | --- | --- |
| seed_001 | `n = n` | Definition/reflexivity |
| seed_002 | `0 + n = n` | Definition |
| seed_003 | `S n + m = S (n + m)` | Definition |
| seed_004 | `1 + n = S n` | Definition |
| seed_005 | `n + (0 + m) = n + m` | Definition |
| seed_006 | `(0 + n) + m = n + m` | Definition |
| seed_007 | `n = m -> n = m` | Direct premise |
| seed_008 | `n = m -> m = n` | Symmetry |
| seed_009 | `n = m -> S n = S m` | Congruence |
| seed_010 | `n = m -> n + p = m + p` | Rewriting |
| seed_011 | `n = m -> p + n = p + m` | Rewriting |
| seed_012 | `n = m -> n + n = m + m` | Reverse rewriting |
| seed_013 | `n = m -> n + m = m + n` | Rewriting |
| seed_014 | `n = 0 -> n + m = m` | Rewriting + definition |
| seed_015 | `n = S m -> n + p = S (m + p)` | Rewriting + definition |
| seed_016 | `n + 0 = n` | Induction on n |
| seed_017 | `n + 1 = S n` | Induction on n |
| seed_018 | `n + S m = S (n + m)` | Induction on n |
| seed_019 | `(a + b) + c = a + (b + c)` | Induction on a |
| seed_020 | `(n + 0) + 0 = n` | Induction on n |
| seed_021 | `(n + 0) + m = n + m` | Induction on n |
| seed_022 | `n + 1 = 1 + n` | Induction on n |
| seed_023 | `n + S (S m) = S (S (n + m))` | Induction on n |
| seed_024 | `(n + 1) + 1 = S (S n)` | Induction on n |
| seed_025 | `(n + 1) + m = S (n + m)` | Induction on n |
| seed_026 | `n + (1 + m) = 1 + (n + m)` | Induction on n |
| seed_027 | `m + 0 = m -> (n + m) + 0 = n + m` | Induction + premise |
| seed_028 | `m + 1 = S m -> (n + m) + 1 = S (n + m)` | Induction + premise |
| seed_029 | `m + p = p + m -> (n + m) + p = (n + p) + m` | Induction + premise |
| seed_030 | `(n + S m) + p = S (n + (m + p))` | Induction on n |

`training_eligible: true` means that the curated formal pair passed the current verifier. It is not a claim of human review, an assigned training split, or evidence that any training has taken place. Tests cover derived-field consistency, original-step preservation, normalization restrictions, premise-versus-induction features, duplicate/family checks, provenance, stale evidence rejection, and actual compilation of both canonical and preserved original bodies.
