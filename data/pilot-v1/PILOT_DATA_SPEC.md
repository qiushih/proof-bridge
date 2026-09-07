# ProofBridge fine-tuning pilot data v1

This release contains **36 assistant-curated proof pairs: 24 train, 6 dev and 6 reserved holdout**. It is a small proof-step learning pilot, not a trained model or a large dataset. All references must pass the existing Rocq 9.2.0 verifier and the frozen constrained-v1 token grammar before the split manifest is frozen. Prompt v2, the decoder, verifier, model, historical seeds, reused diagnostic evaluation and argument-switch-v1 remain unchanged.

## Motivation and reference review

The saved argument-switch probe verified 9/18 outputs but achieved 0/6 verified switches. All six B outputs attempted induction, yet rewrite direction, simplification order and premise use were often wrong. The earlier development audit also found missing introductions and invalid references; those are already addressed by constrained-v1. These findings motivate aligned, useful proof steps. They do not establish that fine-tuning will help.

`historical_reference_review.json` reviews all 12 historical A/B references and rechecks them with Rocq without editing them:

| Historical reference | Quality judgment | Pilot use |
| --- | --- | --- |
| 003–006 A | Direct definitional proofs | Eligible examples of natural reasoning; no copies added |
| 007 A | Direct premise reuse | Eligible; no copy added |
| 011 A | Direct substitution | Eligible; no copy added |
| 003–004 B | Avoidable induction and redundant IH substitution | Exclude from primary targets |
| 005–006 B | Simplify to equal sides, then restore reducible syntax with IH | Exclude from primary targets |
| 007 B | Premise reused in both branches; IH unused | Exclude from primary targets |
| 011 B | Coherent structural proof, but direct substitution suffices | Secondary teaching example only; not copied |

All six B inductions are avoidable for their theorems; five have particularly weak training structure. Their historical PASS results remain valid. The pilot's premise-plus-induction targets instead include a right-zero residual: merely substituting the premise does not complete the general case. Every pilot induction has a useful IH. Construction QA removes each IH reference and each premise reference in turn and requires the altered proof to fail inside Rocq. This tests usefulness in the authored script, not global minimality or logical necessity of every premise.

## Scope and sampling

The language stays within the existing fragment: one to three universally quantified natural numbers, equality, an optional single equality premise, zero/small numerals, successor and addition. There are no new tactics, global lemmas, helper declarations, multiplication or lists. Proofs use explicit introductions and at most one induction with two branches. Every reference including EOS fits the existing 256-token completion budget.

The 36 rows represent 20 explicitly selected theorem statements, including orientation siblings. There are 16 same-theorem A/B groups and four single-argument examples. No paraphrase expansion or random theorem enumeration is performed. Reusable authoring structure only renders the explicitly selected induction cases. The JSONL files themselves are the authoritative curated sources.

| Split | Rows | Theorem IDs | Family allocation |
| --- | ---: | ---: | --- |
| Train | 24 | 14 | 10 equality-transport rows; 14 right-zero rows |
| Dev | 6 | 3 | Successor/increment/reassociation family |
| Holdout | 6 | 3 | Conditional-permutation family; contents reserved |

Training examples cover these decisions:

- Use a premise forwards or backwards, according to the argument's stated substitution direction.
- Rewrite a constructor-headed premise before computation hides its pattern, or simplify first and use an appropriate reverse substitution.
- Compare two valid orders when the needed premise pattern survives simplification.
- Discharge an induction base case with the equality premise instead of reflexivity.
- Apply a useful successor IH through rewriting or through `f_equal` followed by `exact IH`.
- Combine a base-case premise with an induction over a separate prefix variable.

`coverage.json` gives overlapping pattern counts by split. Counts represent reference examples, not independent mathematical facts. Timing labels describe relative command order in one branch; they do not assert that every alternative order is invalid.

## What the A/B pairs mean

The model input is the **formal statement plus the supplied informal proof**; the target is the proof body. Same-theorem variants differ in explicit intermediate reasoning steps. Train pairs `pt01`, `pt02` and `pt03` contrast substitution direction, direction/order together, and order alone. The other pairs contrast IH rewriting with successor congruence.

Rewrite and `f_equal` can implement the same mathematical induction. Opposite equality substitution directions can also be mathematically equivalent. These are **step-level argument contrasts, not claims of different high-level strategies**. A future evaluation should report Rocq verification, broad mathematical faithfulness, and adherence to the requested step contract separately. Exact string matching or a different equivalent tactic alone must not define mathematical unfaithfulness. We deliberately avoid artificial direct-versus-induction contrasts merely to manufacture a strategy switch.

Only `formal_statement` and `informal_proof` enter the exported input. The reference body is an explicitly separate target. IDs, split labels, features, provenance, review judgments, verification output and argument contracts are not input features. A later training experiment must separately freeze its serialization, target-only loss masking, optimizer settings and checkpoint-selection rule; none is implemented here.

## Schema

Each JSONL row uses `schema_version: pilot-1.0`:

| Field | Meaning |
| --- | --- |
| `id`, `theorem_id`, `argument_id` | Unique row, shared exact theorem, and local A/B argument identity |
| `formal_statement` | Fixed verifier-compatible theorem statement, with no declaration wrapper |
| `informal_statement` | Natural-language theorem statement, separate from its proof |
| `informal_proof` | Coherent proof paragraph formed by joining `steps[*].text` with spaces |
| `proof_body` | Canonical target formed by joining `steps[*].code` with newlines |
| `steps` | Ordered informal/formal blocks; a block may contain multiple tactic commands |
| `argument_features` | Inferred induction variable, premise use, rewrite directions, branch-local simplification order, base premise, IH rewrite/congruence and basic tactic features |
| `generation_family` | Conservative family used as the indivisible split unit |
| `split` | Fixed `train`, `dev` or `holdout`; variants inherit their family's assignment |
| `contrast_type`, `argument_contract` | Reviewed type of contrast and requested steps; annotation only, not model input |
| `review` | Assistant alignment/quality judgment and primary-target eligibility; never a claim of human review |
| `provenance` | Assistant-curated informal/formal sources, historical overlap, no model generation, human-reviewed false, Rocq-verified flag and source hash |
| `verification` | Actual verifier status/category, compiler version, stage, kernel/assumption checks, source hash, timing and diagnostics |
| `quality_checks` | Exact frozen-decoder token-path/budget checks and reference-removal ablations |

Rocq checks the formal proof and absence of global axioms. It does not certify English alignment or the manual family/argument labels. The persisted source hash binds verification to the verifier-rendered statement and body. No generated or mutated failure is a training target.

## Splits, freshness and historical exposure

`split_manifest.json` fixes membership and file hashes **before any training**. Related theorems, reversed equalities, definitional variants, repeated contexts and all A/B variants remain within one generation family. The historical equality-transport subfamilies are merged conservatively. Historical successor and reassociation relationships remain merged. Exact/alpha-renamed, definitionally reduced and equality-reversed statement duplicates are checked across partitions, including arbitrary permutations of up to three binders. This is a conservative syntactic audit plus a curated family ledger, not a complete theorem-equivalence decision procedure.

The fresh holdout contains newly authored statements and references, checked automatically against historical development and diagnostic statements for those equivalences. **It is separate from pilot train/dev at family level, but its mathematical family has historical exposure through seed_029.** Thus it is a fresh-instance holdout, not evidence of generalization to wholly unseen mathematical families. Broad induction templates are shared intentionally. No claim is made about model pretraining contamination.

Original development and diagnostic data stay frozen under their old paths. They are not automatically appended to any pilot split. Future training must use the explicit pilot train allowlist; checkpoint selection must use pilot dev. Only the existing three fixed Prompt v2 demonstrations remain available through the frozen prompt. Related historical holdout-family examples must not be imported into training or checkpoint selection. The reused 12-example diagnostic set is not an independent holdout and supplies no selection metric for this pilot. It is read by construction QA solely for an automated duplicate exclusion check.

## Holdout custody

Reference construction, assistant alignment review, Rocq verification, token-path checks and family/duplicate QA necessarily inspect the holdout **before sealing**. These are reference-quality checks, not model evaluation. `preparation.json` records that boundary. Afterward, freeze/check/test commands checksum its bytes without parsing or displaying its contents. Detailed holdout proof diagnostics stay under `reserved/`; the public verification report contains aggregate counts only.

The public loader and export command accept train/dev only. There is no inference or holdout-scoring entry point. Do not open, train on, generate against, or use the reserved files for hyperparameter or checkpoint decisions until a training run and checkpoint selection are complete. After that, authorize a single holdout opening and compare the selected checkpoint with frozen constrained-v1 using the precommitted paired conditions in the split manifest. Subsequent tuning would require a new holdout.

This separation is a workflow safeguard, **not encryption or an access-control boundary**: repository users can still open reserved files. The constructor has necessarily seen their contents. Record any accidental later exposure and stop treating the affected split as untouched.

## Known limitations and readiness

- Only 24 training rows and a few correlated families: this is suitable for a smoke-test learning pilot, not a robust performance estimate.
- Six holdout rows are three correlated theorem pairs in one historically exposed family. Report theorem-level paired outcomes; percentages alone would overstate precision.
- A/B differences mostly test proof-step realization. They do not establish broad strategy switching or independent use of informal prose.
- English uses controlled mathematical language and local naming. There are no noisy handwritten transcriptions or diverse human proof styles.
- All semantic reviews are assistant reviews; human review remains false. Independent human review would improve confidence before a consequential experiment.
- Some targets intentionally overlap older development statements, and right-zero is a frozen prompt demonstration. The historical-overlap audit makes this visible. No such statement duplicate is allowed in the fresh holdout.
- Family membership is conservative but manually chosen; all proofs inhabit a very small common grammar. Proof templates and tactic sequences cross splits by design.
- Reference-removal failures establish a useful step in the current script, not the shortest possible proof or necessity under every alternative strategy.
- Training feasibility, overfitting, improvement and forgetting are unmeasured. No model weights were loaded for generation, no new samples were generated and no training occurred.

See `README.md` for verification and reproduction commands and `VERIFICATION_REPORT.md` for the recorded construction results.
