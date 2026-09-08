# Holdout comparison v1

Verified requested-step adherence improved while verification and mathematical fidelity were preserved.

The pinned base and preselected step-18 adapter each received the same six formal-theorem-plus-informal-proof inputs. Exactly 12 one-attempt generations used unchanged Prompt v2, constrained-v1 and the verifier, with a 256-token budget and no repair.

| Metric | Base | Step 18 | Difference |
| --- | ---: | ---: | ---: |
| Rocq verified | 0/6 (0.0%) | 3/6 (50.0%) | +3 rows |
| Verified + mathematically faithful | 0/6 (0.0%) | 3/6 (50.0%) | +3 rows |
| Verified + requested steps | 0/6 (0.0%) | 3/6 (50.0%) | +3 rows |

## All outputs

| Model | Row | Rocq | Mathematical fidelity | Requested steps |
| --- | --- | --- | --- | --- |
| base | ph01_A | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| base | ph01_B | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| base | ph02_A | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| base | ph02_B | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| base | ph03_A | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| base | ph03_B | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| step18 | ph01_A | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| step18 | ph01_B | PASS / VERIFIED | FAITHFUL | MATCH |
| step18 | ph02_A | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| step18 | ph02_B | PASS / VERIFIED | FAITHFUL | MATCH |
| step18 | ph03_A | FAIL / PROOF_ERROR | NOT_ESTABLISHED | MISMATCH |
| step18 | ph03_B | PASS / VERIFIED | FAITHFUL | MATCH |

## Theorem-pair outcomes

| Theorem | Base: both variants verified and step-adherent | Step 18: both variants verified and step-adherent |
| --- | --- | --- |
| ph01 | False | False |
| ph02 | False | False |
| ph03 | False | False |

The six rows are three correlated A/B pairs. Bodies changing between A and B is not sufficient: both proofs must verify and match their respective step contracts.

## Mathematical review

All twelve proofs were reviewed in shuffled records without model labels, latency, token traces or step scores. The judgments were hash-frozen before the mapping was disclosed for this report. These are assistant reviews, not independent human or guaranteed blind judgments; proof style can suggest model identity. Equivalent successful tactic implementations remain mathematically faithful; failed proofs are NOT_ESTABLISHED.

- **base ph01_A**: The candidate attempts rewrite H before induction. The initial goal has the common prefix n and contains no matching S m + p subterm, so Rocq rejects this first rewrite. The supplied argument instead uses the premise after reducing the induction base case. Subsequent induction commands are unreached.
- **base ph01_B**: The first rewrite H is attempted before induction and fails because the initial prefixed goal contains no matching S m + p subterm. The supplied argument uses H only in the simplified zero branch, then transports the IH by congruence. The subsequent generated induction and IH rewrite are unreached; no complete faithful proof is established.
- **base ph02_A**: The candidate performs induction on n with valid local renamings: the second natural-number binder is k and the equality premise is named m. It nevertheless ignores that premise and fails at base-case reflexivity. SIH is a valid IH name, but its repeated later rewrites are unreached. The failure is mathematical premise omission, not an invalid-reference judgment.
- **base ph02_B**: The candidate attempts a direct premise rewrite and reflexivity, omitting the supplied induction. The goal contains a common n prefix, so the premise pattern m + (p + p) does not occur as the required subterm and Rocq rejects the rewrite. This attempted direct strategy does not produce a verified complete argument.
- **base ph03_A**: The proof attempts the prescribed induction on n and an IH rewrite in the successor branch, using valid local names k, S_k and S_kIH. The base permutation equality is not reflexive; the equality premise S_k is needed but unused. Rocq rejects this base case, so the intended induction is incomplete.
- **base ph03_B**: The candidate inducts on n, but tries reflexivity in the zero case instead of using the supplied equality premise. That equality is not definitional, and Rocq rejects the base case. The unusual local names are valid; the premise is named S_k and the IH is S_kIH. Repeated IH rewrites in the later branch are unreached, and do not establish a complete faithful argument.
- **step18 ph01_A**: The candidate inducts on n but tries to prove the zero equality S (m + p) = p + S m by reflexivity. The premise H would close it by conversion, but is not used. Rocq rejects the base case; the later IH rewrite cannot establish a complete proof.
- **step18 ph01_B**: The proof fixes m, p and H and inducts on n. After zero-case simplification, exact H succeeds by definitional conversion from S m + p to S (m + p). The successor case uses f_equal then exact IH, precisely transporting the IH through successor as the informal argument requests. The complete proof verifies.
- **step18 ph02_A**: The candidate introduces the premise and inducts on the correct variable n, but its base case uses reflexivity on m + (p + p) = (p + p) + m instead of the premise. Rocq rejects that equality as non-definitional. The later forward IH rewrite resembles the requested successor argument, but there is no complete verified proof.
- **step18 ph02_B**: The proof inducts on n while keeping m, p and the permutation premise H fixed. It uses exact H for the simplified zero case and applies successor congruence with f_equal followed by exact IH in the successor case. This follows the supplied mathematical argument and verifies.
- **step18 ph03_A**: The candidate inducts on n but ignores H in the zero case and fails at reflexivity on the permutation equality. Its unreached successor script rewrites the original premise H instead of the induction hypothesis. Rocq stops at the base case, so mathematical fidelity of a complete proof is not established.
- **step18 ph03_B**: With m, p and H fixed, induction is on the supplied prefix variable n. The zero goal is exactly the premise and is closed by exact H. After successor simplification, f_equal reduces to the IH and exact IH finishes. This is the supplied mathematical induction and congruence argument, and Rocq verifies the complete proof.

## Custody and reproduction

The protocol and exact adapter hash were frozen before opening the reserved JSONL. Its contents were parsed once into a separate immutable snapshot. All six gold references were reverified before inference. Base and adapter parameters were hash-compared before/after inference; no weights, prompt, decoder, dataset, verifier or selected checkpoint changed. The original reserved source and historical QA details remain unchanged.

The holdout is now **consumed**. No result here changes the dev-selected step 18 or authorizes further tuning on these rows. The later user-approved 12-generation scope uses argument-conditioned inputs only; the older split-policy proposal also mentioned theorem-only controls, which were not run.

Mean generation latency: base **11.89s**, step 18 **3.93s**. Completion tokens: base **655**, step 18 **192**. Total run wall time: **120.61s**. Sequential worker order and host/cache state limit timing comparisons.

From the repository root, replay without training, new generations or reopening the original reserved contents:

```sh
.venv/bin/python -m pilot_holdout_v1.experiment check --replay-tokens --reverify
.venv/bin/python -m pilot_holdout_v1.assessment report
.venv/bin/python -m unittest discover -s tests -v
```

The one-time command was `.venv/bin/python -m pilot_holdout_v1.experiment run`; it refuses an existing run. `PROTOCOL.md` contains the full precommitted workflow. Machine-readable metrics, all assessed outputs, original raw prompts/token traces, masked reviews, reference evidence and consumed-holdout custody records are saved alongside this report.

## Limits

- Six rows are three correlated theorem pairs in one historically exposed family.
- Fresh-instance transfer under this pipeline; no claim of wholly unseen mathematics or absence of pretraining contamination.
- No theorem-only control, so this comparison alone cannot establish causal informal-proof dependence.
- No further tuning or checkpoint selection may reuse this as an untouched holdout.
