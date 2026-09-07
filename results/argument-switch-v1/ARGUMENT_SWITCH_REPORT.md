# Development argument-switching probe

**Verified switching accuracy: 0/6 (0.0%).** Both A and B must verify and follow their respective fixed strategy requirements. A verified proof that ignores the supplied strategy is a switching failure.

**Verification accuracy: 9/18 (50.0%).** Exactly 18 new generations, one per theorem/condition, no repair or fine-tuning.

Constrained-v1 was frozen first at Git tag `proofbridge-constrained-v1` (checkpoint commit `b20e388`). Its 50-file manifest protects the decoder, prompt/runtime pins, prior evidence, and tests. All 12 A/B references then passed Rocq and exact-token-path checks under that decoder before inference; nine probe tests passed. Reference preparation and scoring were fixed in the preflight manifest.

## Accuracy by condition

| Input | Rocq verified | Attempted requested strategy | Verified + requested strategy |
| --- | ---: | ---: | ---: |
| theorem_only | 4/6 | N/A | N/A |
| argument_A | 5/6 | 5/6 | 5/6 |
| argument_B | 0/6 | 4/6 | 0/6 |

Attempted structural switching, independent of mathematical correctness, was **3/6**. The primary denominator remains all six theorems, including failures. Theorem-only has no supplied strategy and is reported as a control.

## Per-theorem results

| Seed | Strategy A | Strategy B | Theorem-only | A output | B output | Verified switch |
| --- | --- | --- | --- | --- | --- | --- |
| seed_003 | direct_definition | induction on n | FAIL; B | FAIL; B | FAIL; B | no |
| seed_004 | direct_definition | induction on n | FAIL; B | PASS; A | FAIL; B | no |
| seed_005 | direct_definition | induction on n | PASS; A | PASS; A | FAIL; B | no |
| seed_006 | direct_definition | induction on n | PASS; A | PASS; A | FAIL; B | no |
| seed_007 | direct_premise | induction on n | PASS; A | PASS; A | FAIL; neither | no |
| seed_011 | direct_rewrite | induction on p | PASS; A | PASS; A | FAIL; neither | no |

A/B in the output columns identify which reference strategy the output matches. Scoring checks the induction variable and required branch-local uses of IH/premises, not just the presence of an induction keyword. Equivalent tactics are accepted; the saved structure proxy remains separate. All 18 outputs were also manually reviewed by the assistant, not an independent human reviewer.

## Successful and failed switching examples

No verified successful A-to-B switch was observed. No success example is invented.

### seed_003: failed switch

Given argument A, the generated proof was FAIL and its requested-strategy match was False:

```rocq
intros n m.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

A asks for definitional computation. The output instead uses induction, and its successor rewrite also fails. Changing the informal paragraph did not recover the requested direct strategy here.

Given argument B, the generated proof was FAIL and its requested-strategy match was True:

```rocq
intros n.
intros m.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

The output chooses induction on the correct binder and refers to IH in the successor branch, matching the coarse strategy requirements. However, it simplifies before rewriting; the reference first rewrites the inner S k + m. The original IH pattern is no longer found, so this is not a successful implementation or switch.

### seed_004: failed switch

Given argument A, the generated proof was PASS and its requested-strategy match was True:

```rocq
intros n.
simpl.
reflexivity.
```

The output verifies and uses A's direct mathematical argument without induction. Simplification and reflexivity implement the stated definitional reductions.

Given argument B, the generated proof was FAIL and its requested-strategy match was True:

```rocq
intros n.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

The output switches to induction on n and uses IH, matching the coarse strategy requirements. The supplied B argument explicitly needs the IH from right to left; the output uses forward rewrite IH after simplification. Rocq finds no matching subterm. The attempted strategy switch does not verify and earns no switching credit.

## Interpretation and limits

All six B outputs choose induction on the requested binder, including p in seed_011. Thus the result is not evidence that the model ignores the informal text entirely. In seeds 004–006 it changes from A's valid direct argument to the requested induction outline, but cannot complete a valid alternative proof. No verified successful switch was observed.

The B outputs repeatedly use simplification/reflexivity for the base case and forward IH rewriting for the successor. This template ignores the reverse rewrites required by B in 004–006 and the branch-specific premise uses in 007 and 011. In 003 the model also uses induction when A requests only computation. These concrete failures support limited structural responsiveness with poor execution of the supplied argument.

Verification alone is insufficient: an independently generated direct proof does not satisfy an induction request. Conversely, an output with the requested induction outline can still fail because an IH rewrite is inapplicable. The separate metrics distinguish these cases.

These six theorems were already development examples. Four compare definitional computation with deliberately unnecessary induction; 007 compares direct premise use with induction; 011 compares rewriting with induction on the third binder. Several B references use IH redundantly. This is a small strategy-compliance diagnostic, not an unseen-family benchmark or proof of semantic understanding.

Only the target informal paragraph differs between A and B. Frozen demonstrations, model, decoder, verifier, and greedy 256-token settings are unchanged. No diagnostic evaluation example was used, no existing theorem or seed was edited, and no model output was repaired. Manual reviews are recorded separately and do not change the predeclared scores.

## Artifacts and reproduction

- `argument_switch_v1/variants.json`: both informal paragraphs, formal bodies, strategy requirements, and provenance for each theorem.
- `argument_switch_v1/references/seed_XXX_A.v` and `_B.v`: all 12 standalone reference proof files.
- `argument_switch_v1/preflight.json`: Rocq and decoder acceptance evidence before inference.
- `results.jsonl`, `manifest.json`, `attempts.jsonl`, `summary.json`, `completion.json`: immutable machine-readable generation and scoring evidence.
- `manual_reviews.json` and `assessed_results.jsonl`: separate assistant review and combined records.
- `argument_switch_v1/README.md`: exact reproduction/check commands.
