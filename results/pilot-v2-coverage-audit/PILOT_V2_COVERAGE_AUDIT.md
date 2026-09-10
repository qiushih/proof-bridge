# Pilot v2 coverage audit

**Training already contains all four target combinations. The immediate gap is development coverage: none of the six dev rows requires a premise in the induction base case.** The training evidence is also concentrated in a single induction family. More copies of the existing proof templates would not resolve that limitation.

This audit reviews only the frozen **24 training rows and 6 dev rows**. It does not reopen any holdout proof file, inspect diagnostic examples, author examples, generate model outputs, or train. All original statements, proof bodies, alignment steps, prompts, decoder, verifier, splits and historical results remain unchanged. The conclusions below follow from the public train/dev references, not a new analysis of holdout outputs.

## Coverage

Percentages use all rows in each split. The ten direct training proofs are outside the induction matrix.

| Base action | Successor action | Train rows | Dev rows | Train families | Dev families |
| --- | --- | ---: | ---: | ---: | ---: |
| Simplification/reflexivity | Rewrite IH | 3/24 (12.50%) | 3/6 (50%) | 1 | 1 |
| Simplification/reflexivity | `f_equal` + IH | 3/24 (12.50%) | 3/6 (50%) | 1 | 1 |
| Equality premise | Rewrite IH | 4/24 (16.67%) | 0/6 (0%) | 1 | 0 |
| Equality premise | `f_equal` + IH | 4/24 (16.67%) | 0/6 (0%) | 1 | 0 |
| No induction | Direct premise substitution | 10/24 (41.67%) | 0/6 (0%) | 1 | 0 |

- Computational-base training pairs: `pt08`, `pt09`, `pt10`. These are three theorem IDs but two equivalence classes because `pt08`/`pt09` reverse the equality.
- Premise-base training pairs: `pt11`, `pt12`, `pt13`, `pt14`. These are four theorem IDs but only two equivalence classes: `pt11`/`pt12` and `pt13`/`pt14` are orientation siblings.
- All 14 inductive training rows belong to `right_zero_variants`. Each occupied training cell therefore represents **one existing generation family**, not three or four independent families.
- The ten direct rows belong to `equality_transport`. Their premise use does not test branch selection or use of an induction hypothesis.
- All six dev rows belong to `successor_reassociation_variants`, representing three A/B theorem pairs. Every base case closes by computation and reflexivity.

Premise-dependent bases occur in **8/14 (57.14%) of inductive training rows**, so they are not the minority within that subset. The two successor methods are balanced both overall (7/7) and within premise-base examples (4/4). Premise-base IH rewrites are also balanced by direction: two forward and two reverse. Dev contains only forward IH rewrites and no premise-dependent base cases.

Family counts follow the frozen conservative ledger. They are the available grouping units, not proof of statistical independence. The existing syntactic equivalence check handles definitional reduction, binder permutation and equality orientation; it does not decide all mathematical equivalences.

## Informal/formal review

All 30 rows were assistant-reviewed against their statements, proof bodies and aligned steps. **No mathematical alignment error was identified.** The direction/order of direct substitutions and the base/successor roles in the inductions are explicitly described. `proof_body` and `informal_proof` agree with their canonical joins of `steps` in all 30 rows, and recomputed features agree with stored annotations.

The premise rows explicitly say the simplified base equality is the premise and use `exact H.`; successor branches use the IH. The saved reference-removal checks support the usefulness of these commands in the authored scripts. This does not establish that every premise is logically necessary or that these are globally shortest proofs. In particular, `pt11`/`pt12` assume right-zero identities that can themselves be proved; `pt13`/`pt14` still center on an `m + 0` residual.

Minor clarity issues, recorded per row in `coverage_audit.json`:

| Issue | Rows affected | Proposed treatment in a future version |
| --- | ---: | --- |
| Missing period before “This” after the displayed base equality | 20 | Add a sentence boundary in both the paragraph and aligned step text. |
| “Fix n as natural numbers” uses plural for one variable | 8 | Use “Fix a natural number n.” |
| “Keeping the other introduced data fixed” when no other data exists | 8 | Omit the empty boilerplate. |
| “Exactly the induction hypothesis” relies on definitional conversion | 2 (`pd01_B`, `pd03_B`) | Explain the reduction of the numeral expression. |
| The IH is displayed in a reduced form without saying so | 2 (`pd03_A`, `pd03_B`) | State that the right side has been unfolded by definition. |

Counts overlap. These are prose improvements, not failed proofs. All mathematical-fidelity reviews remain assistant judgments, with `human_reviewed: false`. Rewrite and congruence are equivalent realizations of the same induction here; they should differ in requested-step scoring without automatically differing in mathematical fidelity.

## Smallest proposed intervention

**Propose two new dev rows: one theorem with A/B proofs, both requiring the premise in the base case, with IH rewriting versus IH congruence in the successor case. Propose zero training additions now.** This fills the two empty cells with the smallest same-theorem pair. It would yield eight dev rows; it would not produce balanced base-case counts or a robust generalization benchmark.

Construction is a future task. The theorem must have natural, useful induction and premise steps and pass the unchanged verifier and decoder. It must fit the dev family after a conservative family/duplicate review; train-family relatives and consumed-holdout-family variants must not be reassigned to dev. The proposed two-row count is a lower bound, not a claim that a suitable theorem has already been constructed. If no natural leakage-safe candidate fits, revisit the v2 split design explicitly instead of weakening the family rule or forcing an artificial proof.

The frozen v1 dev set and its checkpoint-selection history stay unchanged. Additional rows belong to a separately versioned and frozen v2 dev release. Its evaluation protocol must specify how the added rows enter selection before training. The v1 selection rule is not edited by this audit.

Only after this evaluation gap is addressed should training coverage be expanded. A concrete structural gap could justify one natural A/B training pair; increased volume, paraphrases or reversed-equality siblings alone are not evidence of improved coverage. Any later training comparison needs an original-data control under the same hardware, optimization budget and v2 evaluation protocol. Report verification, verified mathematical fidelity, requested-step adherence and both-argument theorem success separately. A fresh holdout must be prepared and sealed before further training; none is designed or accessed here.

## Checks and reproduction

- **30/30 original references freshly verified by Rocq 9.2.0**, with kernel/assumption checks and source hashes matching frozen verification records.
- Existing schema, feature, frozen-decoder grammar and family checks passed on all 30 rows.
- 17 theorem IDs, 13 same-theorem A/B groups, 11 conservative equivalence classes across train/dev.
- Zero cross-split equivalences, zero cross-split families, zero duplicate statement/body pairs in this public subset. No claim about new candidates or a fresh holdout is made.
- Recorded input hashes are unchanged before/after the audit. No model or tokenizer inference runs were needed. Existing saved token-budget evidence is retained; tokenizer-level paths were not rerun.

From the repository root:

```sh
# Recompute counts, validate public references and compare the deterministic audit.
.venv/bin/python -m scripts.pilot_v2_coverage_audit

# Also compile the 30 original references; saves fresh compiler diagnostics.
.venv/bin/python -m scripts.pilot_v2_coverage_audit --reverify

# Confirm that tracked frozen artifacts have no changes.
git diff --exit-code
git diff --cached --exit-code
```

`coverage_audit.json` contains counts, IDs, family/equivalence groupings, source hashes, per-row assistant reviews, saved reference evidence and the proposed intervention. `rocq_reverification.json` contains the fresh compiler results. The audit script accepts no arbitrary dataset path or holdout split. Its default run compares deterministic output without overwriting different evidence; `--reverify` refreshes only this audit's compiler report.

The audit's checks and reference compilation were run locally. A full historical regression suite was not run for this additive reporting task. No dataset or runtime implementation changed, and no causal claim about model errors follows from coverage alone.
