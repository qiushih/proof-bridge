# ProofBridge v2 final holdout comparison

Exactly 18 new one-attempt generations compare the pinned base and the already selected control/intervention step-36 adapters. No checkpoint was reselected and no weights were updated.

| Model | Verified | Verified + mathematically faithful | Verified + requested steps | Verified + faithful + requested steps | Both-argument successes |
| --- | ---: | ---: | ---: | ---: | ---: |
| base | 0/6 | 0/6 | 0/6 | 0/6 | 0/3 |
| control | 5/6 | 5/6 | 5/6 | 5/6 | 2/3 |
| intervention | 6/6 | 6/6 | 6/6 | 6/6 | 3/3 |

Pair success requires both A and B to verify, be mathematically faithful, and match their respective requested steps. All failed proofs remain in the denominator. Raw requested-step matches and percentages are recorded separately in summary.json.

## Per-theorem results

| Theorem | Model | A: verified / faithful / steps | B: verified / faithful / steps |
| --- | --- | --- | --- |
| phv201 | base | no / no / no | no / no / no |
| phv201 | control | no / no / no | yes / yes / yes |
| phv201 | intervention | yes / yes / yes | yes / yes / yes |
| phv202 | base | no / no / no | no / no / no |
| phv202 | control | yes / yes / yes | yes / yes / yes |
| phv202 | intervention | yes / yes / yes | yes / yes / yes |
| phv203 | base | no / no / no | no / no / no |
| phv203 | control | yes / yes / yes | yes / yes / yes |
| phv203 | intervention | yes / yes / yes | yes / yes / yes |

## Interpretation and limits

The frozen development intervention-success decision remains false: both selected arms scored 8/8 on dev. Holdout results do not revise that decision, the checkpoint selections, or any frozen input.

These six rows are three correlated A/B pairs from one family. They are fresh held-out instances under the frozen grouping, not independent unseen mathematics. References were assistant-curated and checked before sealing. Fidelity judgments here are assistant reviews, not independent human review. Equivalent IH tactic implementations may be mathematically faithful while missing the requested proof step. There is no theorem-only control in this final evaluation, so these results do not establish causal informal-proof dependence. Further tuning requires a new holdout.

## Integrity and timing

The original holdout was opened once after the training results, selected hashes, runner, public preflight and protocol were committed. Later assessment/replay reads the hash-identical snapshot. Historical holdout and diagnostic contents remained unopened. All model processes used frozen Prompt v2, constrained-v1, CPU float32, greedy one-beam decoding, seed 1729, 256 tokens, one attempt, no repair, and the unchanged Rocq verifier. Base and adapter hashes match before and after inference.

| Model | Mean generation seconds | Prompt tokens | Completion tokens |
| --- | ---: | ---: | ---: |
| base | 4.14 | 3550 | 204 |
| control | 3.96 | 3550 | 194 |
| intervention | 4.11 | 3550 | 195 |

Timing is descriptive: fresh process/vocabulary per model, shared cache within each six-row run, no reference prewarming.

## Reproduction

From the repository root:

```sh
.venv/bin/python -B -m pilot_holdout_v2.experiment check --replay-tokens --reverify
.venv/bin/python -B -m pilot_holdout_v2.experiment report
.venv/bin/python -B -m unittest discover -s tests -p test_pilot_holdout_v2.py -v
```

These commands replay saved evidence or test public fixtures; they do not generate new samples. Never delete the custody/start markers to repeat the holdout run.

## Saved outputs and manual fidelity reviews

### base / phv201_A

```coq
intros n m p H.
rewrite H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR; mathematical fidelity: NOT_ESTABLISHED; requested steps: MISMATCH.

The proof tries rewrite H before induction, but m + m is not a syntactic subterm of the current goal with variable n. Rocq rejects that first rewrite. The supplied argument instead uses H after simplifying the n = 0 branch. Neither the complete induction nor mathematical fidelity is established by this failed script.

### base / phv201_B

```coq
intros n m p H.
rewrite H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR; mathematical fidelity: NOT_ESTABLISHED; requested steps: MISMATCH.

The proof tries rewrite H before induction, but m + m is not a syntactic subterm of the current goal with variable n. Rocq rejects that first rewrite. The supplied argument instead uses H after simplifying the n = 0 branch. Neither the complete induction nor mathematical fidelity is established by this failed script.

### base / phv202_A

```coq
intros n m p H.
rewrite H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR; mathematical fidelity: NOT_ESTABLISHED; requested steps: MISMATCH.

The proof tries rewrite H before induction, but m + (m + m) is not a syntactic subterm of the current goal with variable n. Rocq rejects that first rewrite. The supplied argument instead uses H after simplifying the n = 0 branch. Neither the complete induction nor mathematical fidelity is established by this failed script.

### base / phv202_B

```coq
intros n m p H.
rewrite H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR; mathematical fidelity: NOT_ESTABLISHED; requested steps: MISMATCH.

The proof tries rewrite H before induction, but m + (m + m) is not a syntactic subterm of the current goal with variable n. Rocq rejects that first rewrite. The supplied argument instead uses H after simplifying the n = 0 branch. Neither the complete induction nor mathematical fidelity is established by this failed script.

### base / phv203_A

```coq
intros n m p H.
rewrite H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR; mathematical fidelity: NOT_ESTABLISHED; requested steps: MISMATCH.

The proof tries rewrite H before induction, but (m + m) + m is not a syntactic subterm of the current goal with variable n. Rocq rejects that first rewrite. The supplied argument instead uses H after simplifying the n = 0 branch. Neither the complete induction nor mathematical fidelity is established by this failed script.

### base / phv203_B

```coq
intros n m p H.
rewrite H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR; mathematical fidelity: NOT_ESTABLISHED; requested steps: MISMATCH.

The proof tries rewrite H before induction, but (m + m) + m is not a syntactic subterm of the current goal with variable n. Rocq rejects that first rewrite. The supplied argument instead uses H after simplifying the n = 0 branch. Neither the complete induction nor mathematical fidelity is established by this failed script.

### control / phv201_A

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR; mathematical fidelity: NOT_ESTABLISHED; requested steps: MISMATCH.

Induction on n is appropriate, but the base goal m + m = p requires the premise H. The generated reflexivity step cannot identify an arbitrary p with m + m, and Rocq rejects it. The later IH rewrite does not remedy the unproved base case; mathematical fidelity remains unestablished.

### control / phv201_B

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  f_equal.
  exact IH.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives m + m = p, which exact H correctly closes. The successor goal is the equality (k + m) + m = k + p lifted under S. Successor congruence via f_equal reduces it to the exact IH, which exact IH supplies. This follows the mathematical argument and the requested congruence step.

### control / phv202_A

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives m + (m + m) = p, which exact H correctly closes. The successor goal is the equality (k + m) + (m + m) = k + p lifted under S. Rewriting IH from left to right inside that successor makes the two sides identical, and reflexivity finishes. This follows the mathematical argument and the requested forward-rewrite step.

### control / phv202_B

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  f_equal.
  exact IH.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives m + (m + m) = p, which exact H correctly closes. The successor goal is the equality (k + m) + (m + m) = k + p lifted under S. Successor congruence via f_equal reduces it to the exact IH, which exact IH supplies. This follows the mathematical argument and the requested congruence step.

### control / phv203_A

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives (m + m) + m = p, which exact H correctly closes. The successor goal is the equality ((k + m) + m) + m = k + p lifted under S. Rewriting IH from left to right inside that successor makes the two sides identical, and reflexivity finishes. This follows the mathematical argument and the requested forward-rewrite step.

### control / phv203_B

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  f_equal.
  exact IH.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives (m + m) + m = p, which exact H correctly closes. The successor goal is the equality ((k + m) + m) + m = k + p lifted under S. Successor congruence via f_equal reduces it to the exact IH, which exact IH supplies. This follows the mathematical argument and the requested congruence step.

### intervention / phv201_A

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives m + m = p, which exact H correctly closes. The successor goal is the equality (k + m) + m = k + p lifted under S. Rewriting IH from left to right inside that successor makes the two sides identical, and reflexivity finishes. This follows the mathematical argument and the requested forward-rewrite step.

### intervention / phv201_B

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  f_equal.
  exact IH.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives m + m = p, which exact H correctly closes. The successor goal is the equality (k + m) + m = k + p lifted under S. Successor congruence via f_equal reduces it to the exact IH, which exact IH supplies. This follows the mathematical argument and the requested congruence step.

### intervention / phv202_A

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives m + (m + m) = p, which exact H correctly closes. The successor goal is the equality (k + m) + (m + m) = k + p lifted under S. Rewriting IH from left to right inside that successor makes the two sides identical, and reflexivity finishes. This follows the mathematical argument and the requested forward-rewrite step.

### intervention / phv202_B

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  f_equal.
  exact IH.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives m + (m + m) = p, which exact H correctly closes. The successor goal is the equality (k + m) + (m + m) = k + p lifted under S. Successor congruence via f_equal reduces it to the exact IH, which exact IH supplies. This follows the mathematical argument and the requested congruence step.

### intervention / phv203_A

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives (m + m) + m = p, which exact H correctly closes. The successor goal is the equality ((k + m) + m) + m = k + p lifted under S. Rewriting IH from left to right inside that successor makes the two sides identical, and reflexivity finishes. This follows the mathematical argument and the requested forward-rewrite step.

### intervention / phv203_B

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  exact H.
- simpl.
  f_equal.
  exact IH.
```

Rocq: PASS / VERIFIED; mathematical fidelity: FAITHFUL; requested steps: MATCH.

The proof fixes m, p and the premise, then inducts on n as supplied. At n = 0, simplification gives (m + m) + m = p, which exact H correctly closes. The successor goal is the equality ((k + m) + m) + m = k + p lifted under S. Successor congruence via f_equal reduces it to the exact IH, which exact IH supplies. This follows the mathematical argument and the requested congruence step.
