# V2 development baseline

Exactly **four new generations** plus **twelve reused records** compare the pinned base model and the fixed step-18 adapter on eight development rows. No training or checkpoint selection occurred.

| Subset | Model | Rocq verified | Verified + faithful | Verified + requested steps | Both-argument successes |
| --- | --- | ---: | ---: | ---: | ---: |
| original_six | base | 6/6 | 6/6 | 3/6 | 0/3 |
| original_six | step18 | 6/6 | 6/6 | 6/6 | 3/3 |
| new_two | base | 0/2 | 0/2 | 0/2 | 0/1 |
| new_two | step18 | 0/2 | 0/2 | 0/2 | 0/1 |
| all_eight | base | 6/8 | 6/8 | 3/8 | 0/4 |
| all_eight | step18 | 6/8 | 6/8 | 6/8 | 3/4 |

Both-argument success requires both proofs to verify, remain mathematically faithful and follow their respective requested steps. Failed proofs count as NOT_ESTABLISHED, never faithful.

## New outputs and assistant reviews

### base / pd04_A

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

Rocq: FAIL / PROOF_ERROR. Mathematical fidelity: NOT_ESTABLISHED. Requested steps: MISMATCH.

Rocq rejects rewrite H before induction: the whole goal has (n + m) + S p, so the required subterm m + S p is not available there. The supplied argument instead uses H after simplification in the zero branch. The later induction/rewrite-IH outline resembles argument A but cannot establish fidelity for a rejected proof; its base branch also substitutes reflexivity for the requested premise use.

### base / pd04_B

```coq
intros n m p H.
rewrite H.
reflexivity.
```

Rocq: FAIL / PROOF_ERROR. Mathematical fidelity: NOT_ESTABLISHED. Requested steps: MISMATCH.

The output attempts direct rewriting with H and reflexivity, omitting the supplied induction and successor-congruence argument. Rocq rejects rewrite H because the needed subterm m + S p is absent in the initial goal. This is an attempted different strategy and is not a verified alternative proof. Mathematical fidelity is not established.

### step18 / pd04_A

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  reflexivity.
```

Rocq: FAIL / PROOF_ERROR. Mathematical fidelity: NOT_ESTABLISHED. Requested steps: MISMATCH.

The induction and successor-branch rewrite match the broad argument A outline, but the zero branch uses reflexivity instead of H. After simplification the goal is m + S p = S (m + p), which requires the premise; Rocq rejects reflexivity. Mathematical fidelity is not established despite the matching successor-branch outline.

### step18 / pd04_B

```coq
intros n m p H.
induction n as [| k IH].
- simpl.
  reflexivity.
- simpl.
  rewrite IH.
  exact H.
```

Rocq: FAIL / PROOF_ERROR. Mathematical fidelity: NOT_ESTABLISHED. Requested steps: MISMATCH.

The model starts the requested induction but again replaces the essential premise use in the zero branch with reflexivity, where Rocq rejects it. The unexecuted successor-branch text also uses rewrite IH followed by exact H instead of the requested congruence and exact IH. Equivalent successful implementations could be faithful despite a step mismatch, but this rejected proof establishes neither mathematical fidelity nor requested-step success.

## Latency and usage

| Source subset | Model | Mean generation seconds | Completion tokens |
| --- | --- | ---: | ---: |
| original_six | base | 6.53 | 187 |
| original_six | step18 | 2.67 | 177 |
| new_two | base | 5.49 | 48 |
| new_two | step18 | 6.60 | 63 |

Historical and new runs have different cache/order/host conditions. Timings are descriptive, not a hardware speed comparison. Full prompt/completion token usage and runtime versions are retained in machine-readable records/manifests.

## Integrity and interpretation

The reuse audit confirms identical prompts, rendered bytes, token IDs, model identities and generation settings for the original six rows. Their original raw records and hash-bound reviews are retained unchanged. The base's historical B proofs use IH rewriting; those verified proofs remain mathematically faithful despite missing the requested congruence steps.

All sixteen saved outputs were replayed through the unchanged Rocq verifier with matching status/category/source hashes. New inference used frozen Prompt v2, constrained-v1, CPU float32, eager attention, seed 1729, greedy one-beam decoding, one attempt, no repair, and 256 tokens. In-memory base/adapter hashes match before/after. No optimizer was constructed and no model weights were updated.

The original six rows were used to select step 18. The two new rows are one theorem pair in the same historically exposed dev family. This is development evidence, not independent generalization or statistical proof of improvement. No new theorem-only controls were run, so no causal claim of informal-proof dependence is supported. Mathematical reviews are assistant reviews, not independent human assessments. Holdout and diagnostic example files were not opened. The v1 checkpoint-selection rule remains unchanged.

## Reproduction

From the repository root:

```sh
.venv/bin/python -B -m baseline_dev_v2.experiment check --replay-tokens --reverify
.venv/bin/python -B -m unittest discover -s tests -p 'test_v2_dev_baseline.py' -v
```

The one-time workflow was `freeze`, `run`, followed by four hash-bound entries in `new_reviews.json` and `assess`. Each command accepts `--output` for its explicit result directory. `run` refuses an existing start marker; do not delete it to retry. `check` never generates samples. `protocol.json` and PROTOCOL.md were frozen before the new attempts; `release.json` seals the final assessment and report. The raw new generations and attempt journals are in base/ and step18/; reused_records.jsonl identifies historical provenance; assessed_results.jsonl contains all sixteen results.
