# Prompt v2: development-only failure audit

**Recommend one next experiment: binder- and scope-aware constrained decoding.** The frozen 3-shot prompt loses most failed outputs before Rocq can check them. This experiment is recommended only; no new samples were generated.

Reviewed all **180 saved development outputs** (118 unique theorem/body pairs) from the 2-, 3-, and 4-shot candidates. Headline counts use only the frozen **short_3shot: 60 outputs on 30 seeds**, both conditions. The 12 diagnostic examples and their results were not read. Prompt, data, verifier, and saved outputs remain unchanged.

## Failure counts

The frozen prompt has **28/60 verified outputs** and **32 failures**. All verified outputs are faithful under the reviewed mathematical-argument rubric. Percentages below use 32 failures or 60 outputs, as labeled.

| Primary category | Count | % of failures | % of all outputs |
| --- | ---: | ---: | ---: |
| formatting parser failure | 2 | 6.25% | 3.33% |
| missing introductions | 7 | 21.88% | 11.67% |
| invalid hypothesis reference | 14 | 43.75% | 23.33% |
| mathematically incorrect proof step | 9 | 28.12% | 15.00% |
| forbidden construct | 0 | 0.00% | 0.00% |
| verified but argument unfaithful | 0 | 0.00% | 0.00% |
| other | 0 | 0.00% | 0.00% |

Each failure gets one primary cause. The 14 excess-introduction failures invent a nonexistent premise H and count as invalid references. The real-but-unintroduced H in seed_014 counts as missing introductions. Malformed `rewrite S n` counts as parser failure. The incomplete symmetry proof counts as an unfinished mathematical argument. These recategorizations do not alter the verifier's original labels.

**23/32 failures (71.88%) occur before Rocq**, versus 9 failures in accepted proofs. The shared 25-seed subset has 22/29 policy failures (75.86%), so the conclusion survives excluding demonstrations and their equivalents. A policy failure can conceal further mathematical errors: for example, seed_027–029 with proof A also mishandle the base-case premise.

| Saved candidate | Verified | Policy rejections | Rocq failures | Faithful despite proxy mismatch |
| --- | ---: | ---: | ---: | ---: |
| short_2shot | 17/60 | 27 | 16 | 4 |
| short_3shot | 28/60 | 23 | 9 | 10 |
| short_4shot | 16/60 | 29 | 15 | 1 |

## Manual fidelity review

These are **assistant reviews, not independent human reviews**. All 61 verified outputs across the three candidates preserve their reference's mathematical argument. The proxy rejects 15 of these, including 10 for the frozen prompt. No verified genuinely different strategy was found; failed attempts with different strategies, omitted induction, or invented premises are flagged separately in JSON.

- Seeds 007–009: `rewrite H; reflexivity` implements the same premise use, symmetry, or successor congruence as `exact H`, `symmetry`, or `f_equal`.
- Seed 012: substitution runs in the opposite direction. It is the same mathematical argument, with an explicit step-direction deviation note.
- Seeds 017, 022, 023: `rewrite IH; reflexivity` and `f_equal; exact IH` implement the same induction-hypothesis congruence.

Parser-unassessable attempts can still contain recognizable partial English alignment; that does not make them verified or faithful complete proofs. Every output has a hash-bound review, original diagnostic, and strategy label in `failure_analysis.json`.

## Informal-proof dependence

**The requested theorem-only / proof-A / proof-B test was not run.** The saved records contain one target informal proof per theorem and zero B outputs. Completing that test requires new generations, which this task forbids. Different few-shot candidates and definitionally related theorems are not substitutes for the missing condition.

The available paired comparison holds the theorem, prompt prefix, model, and generation settings fixed:

| Frozen prompt input | Format accepted | Verified / manually faithful |
| --- | ---: | ---: |
| theorem_only | 20/30 | 12/30 |
| theorem_and_informal | 17/30 | 16/30 |

- Proof A changes **20/30 outputs**. It adds an attempted induction in **13** pairs, but many such outputs still fail binder/reference checks.
- It gains verified faithful proofs on **005, 006, 016, 020, 022, 024**, while losing them on **008 and 014**: 10 pass in both conditions, 12 in neither.
- The successful induction gains are 016, 020, 022, 024. Seed 016 is itself a fixed demonstration. On the common 25-seed subset, proof A verifies 12/25 versus 9/25 theorem-only (five gains, two losses).
- Proof A requests congruence and reverse rewriting in several seeds, yet the output repeats forward rewriting. Many longer induction arguments trigger the same short template with omitted binders or base-case assumptions.

The saved pairs support sensitivity to the supplied text and some coarse induction cues. They do **not** establish faithful following of detailed steps or switching between two valid arguments for the same theorem. Manual faithful counts score the theorem-only output against the same hidden reference, not evidence that it read English.

## Exactly one next experiment

Test **binder- and scope-aware constrained decoding** with the frozen prompt and model on the same development examples. Constrain the existing grammar, binder count, local types/names, and induction-branch scope; leave mathematical strategy selection to the model. Keep the same greedy 256-token budget, one attempt, existing verifier, and two recorded input conditions. No prompt change, repair, data expansion, fine-tuning, or diagnostic-set decisions.

Report all 30 seeds and the common 25-seed subset. Measure format acceptance, binder/reference failures, verification, reviewed fidelity, latency, and tokens. Success requires more **verified faithful proofs** on the common subset as well as fewer policy failures. Merely moving failures into Rocq is insufficient. No claim is made that the 23 rejected outputs would become correct after constraints.

## Reproduce this audit without inference

```sh
python3 -m scripts.audit_prompt_v2_development --check
python3 -m unittest discover -s tests -p test_prompt_v2_audit.py -v
```

`python3 -m scripts.audit_prompt_v2_development` rebuilds only this audit's two reports. Frozen input hashes and saved development evidence are checked; it never loads diagnostic examples or a model. The JSON includes per-candidate/condition totals, the common subset, all reviews, paired evidence, and the explicit missing-B status.
