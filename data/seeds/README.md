# First 30 arithmetic seed proof pairs

This release contains 30 individually selected and authored pairs: 6 proofs by definition, 9 using an equality premise, and 15 using one natural-number induction. They fit the existing addition-only grammar and use no global helper lemmas. Every proof passed the pinned Rocq 9.2.0 / Stdlib 9.1.0 verifier, including its axiom check, in an independent temporary environment. No model training or paraphrase expansion was performed.

These are **assistant-curated examples, not independently human-authored proofs**. Each informal argument was reviewed against its formal steps by the authoring assistant. The recorded kernel checks establish the formal propositions; they do not certify English alignment. The corpus intentionally contains related exercises and common proof patterns, so 30 pairs should not be interpreted as 30 independent mathematical families.

| File | Purpose |
| --- | --- |
| `curated.json` | Editable source: individually written statements and informal/formal step pairs |
| `pairs.jsonl` | 30 complete records, one JSON object per line, compatible with the example's `input` / `target` structure |
| `verification_report.json` | Actual results, per-method/group counts, compiler version, and checksums |
| `../../scripts/verify_seeds.py` | Build records or check their integrity and rerun Rocq |

From the repository root:

```sh
python3 scripts/setup_rocq.py --check
python3 scripts/verify_seeds.py --check
python3 -m unittest discover -s tests -v
```

To refresh exported pairs after an intentional source edit:

```sh
python3 scripts/verify_seeds.py --write
```

`--write` runs all 30 proofs and stores actual outcomes, including failures if any occur. It exits nonzero unless every proof passes. `--check` verifies the stored artifacts against current inputs before rechecking each proof; it does not overwrite timestamps or results. Changing the source, environment lock, verifier, or build script requires refreshing the evidence. Each compilation has the existing verifier's default ten-second timeout.

Each exported record contains the informal statement/proof, fixed formal statement, proof body, full rendered Rocq source, allowed tactics/helpers, method, induction variable when applicable, step-to-line alignment, provenance, grouping, and compiler evidence. The full source in `target.rocq_source` is exactly the source identified by the verification hash. Its local variables use the verifier's canonical names. Alignment line numbers refer instead to `target.proof_body`, which retains the curated names. Each candidate is verified in isolation, so an earlier seed cannot supply a helper to a later one.

All records have `split: unassigned`. The seven `split_group` values are conservative grouping hints for related exercises, not established train/test partitions or a guarantee against every form of leakage. Seed 016 is the canonical right-zero example already present in `examples/add_zero_right.json`; its provenance records that link. Do not count that existing example as an additional independent held-out problem. No other statements duplicate one another merely through renaming quantified variables.

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

`training_eligible: true` means that the curated formal pair passed the current verifier. It is not a claim of human review, an assigned training split, or evidence that any training has taken place. The seed tests also reject duplicate statements modulo binder names, mismatched induction annotations, changed proof bodies with reused PASS labels, and stale generated-source hashes.
