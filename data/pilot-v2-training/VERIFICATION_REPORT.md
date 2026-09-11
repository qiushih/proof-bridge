# V2 reference verification

PASS: 40/40 unique train/dev/fresh-holdout reference rows compile with Rocq 9.2.0,
including 2/2 new training and 6/6 fresh holdout references before sealing. All target
token paths plus EOS are accepted by frozen constrained-v1 within 256 tokens; all
Prompt v2 plus target serializations fit 768 tokens and pass target-only mask checks.

All 32 negative controls fail within Rocq as intended: remove premise use, remove
IH use, replace the base premise with reflexivity, or attempt direct substitution
without induction, for each of eight new references. These establish useful steps
in the authored script, not global minimality or logical necessity of every premise.

Each new A/B pair has matching own-step contracts and mismatching opposite-step
contracts; both verified variants remain mathematically faithful to the same
induction. Semantic/family judgments are assistant reviews, not independent human
review. Frozen original training/dev text and targets remain byte-identical.

Construction checks found zero cross-split canonical equivalents, duplicate target
pairs, or overlapping family assignments among the 40 rows, and no new statement
equivalent to the public pilot or 30 historical development statements. The manual
family ledger and exact-check limitations are in family_review.json. No comparison
with unopened old holdout/diagnostic statements is claimed. Reserved detailed QA
must remain sealed after construction. No model generation or training occurred.
