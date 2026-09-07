# Pre-training dev baseline v1

Exactly **9 generations**: three frozen pilot dev theorems × theorem only / argument A / argument B. One attempt each, no repair, 256 tokens, frozen Prompt v2 and unchanged constrained-v1, model and verifier.

The training protocol was frozen before inference. No adapters were attached to Qwen, no training forward/backward or optimizer step ran, and the base model files remain unchanged. The sealed holdout was not inspected or evaluated; integrity checks used existing byte hashes only.

| Condition | Rocq verified | Verified + mathematically faithful | Requested steps matched | Verified + requested steps | Mean latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| theorem_only | 0/3 | N/A | N/A | N/A | 4.06s |
| argument_A | 3/3 | 3/3 | 3/3 | 3/3 | 8.71s |
| argument_B | 3/3 | 3/3 | 0/3 | 0/3 | 4.36s |

Overall verification: **6/9 (66.7%)**. On the six argument-conditioned rows: **6/6** verified and mathematically faithful; **3/6** verified and step-adherent. The theorem-only controls have no supplied argument and are excluded from both fidelity denominators and checkpoint selection.

## Per-output assessment

| Theorem | Condition | Verification | Mathematical fidelity | Requested steps |
| --- | --- | --- | --- | --- |
| pd01 | theorem_only | FAIL / PROOF_ERROR | NOT_APPLICABLE | NOT_APPLICABLE |
| pd01 | argument_A | PASS / VERIFIED | FAITHFUL | MATCH |
| pd01 | argument_B | PASS / VERIFIED | FAITHFUL | MISMATCH |
| pd02 | theorem_only | FAIL / PROOF_ERROR | NOT_APPLICABLE | NOT_APPLICABLE |
| pd02 | argument_A | PASS / VERIFIED | FAITHFUL | MATCH |
| pd02 | argument_B | PASS / VERIFIED | FAITHFUL | MISMATCH |
| pd03 | theorem_only | FAIL / PROOF_ERROR | NOT_APPLICABLE | NOT_APPLICABLE |
| pd03 | argument_A | PASS / VERIFIED | FAITHFUL | MATCH |
| pd03 | argument_B | PASS / VERIFIED | FAITHFUL | MISMATCH |

## Fidelity review

These are hash-bound assistant reviews, not independent human reviews. Rocq verification, mathematical argument fidelity and strict requested-step adherence are separate fields. Equivalent IH rewriting and successor congruence are accepted as the same mathematical induction. A proof can therefore be mathematically faithful while missing the requested tactic/step realization. Failed proofs cannot establish faithful complete arguments.

- **pd01 argument_A**: The proof inducts on n, computes the zero case and rewrites the successor goal with the IH from left to right. It verifies and follows both the mathematical induction and the requested A steps.
- **pd01 argument_B**: The proof uses the same induction and base case as the supplied argument. Rewriting with the IH transports equality through successor, which is mathematically equivalent to the requested congruence argument. It verifies, but uses rewrite instead of f_equal/exact IH, so the B step contract is missed.
- **pd02 argument_A**: Induction is on n with m and p fixed. The zero equality is computed, and the IH is rewritten under successor after simplification. This is the supplied mathematical argument and the requested A step sequence.
- **pd02 argument_B**: The verified proof performs the prescribed induction on n and solves the successor case using the same IH. IH rewriting is an equivalent implementation of successor congruence, so mathematical fidelity is preserved while the explicit B f_equal/exact step sequence is not followed.
- **pd03 argument_A**: The second theorem binder m is named k locally, and the successor variable and IH are named kIH and kIH2. These are valid local renamings. The proof fixes that second binder, inducts on n, computes the base case and rewrites with its actual IH; it verifies and matches the A steps after name/intros normalization.
- **pd03 argument_B**: The proof fixes m and inducts on n, with the correct computed base and IH-based successor step. Rewriting the IH under successor is mathematically equivalent to the supplied congruence reasoning. It verifies but misses B's requested f_equal/exact realization. Its difference from the A output is only naming and introduction grouping, not a different mathematical argument.

A/B bodies changed on **1/3** theorems. Both variants verified and followed their respective step contracts on **0/3**. This is a step-realization test; A and B share the same high-level induction argument.

The frozen step-zero checkpoint-selection tuple is **[6, 3, 6, 0]**: verified faithful, verified step-adherent, verified, negative optimizer step. This is the comparator for the future fixed 60-update LoRA pilot, not evidence of a training improvement.

## Reproduction and custody

Raw generations, prompts, token IDs, constraint traces, latency and verifier evidence are in `raw_generations.jsonl`; one-attempt events are in `attempts.jsonl`. `assessed_results.jsonl` adds the separate reviewed mathematical judgments. `summary.json` is machine-readable. Protocol and results have separate freezes so the configuration demonstrably precedes inference.

From the repository root:

```sh
.venv/bin/python -m training_protocol_v1.experiment check --replay-tokens --reverify
.venv/bin/python -m unittest discover -s tests -p 'test_training_protocol_v1.py' -v
.venv/bin/python -m unittest discover -s tests -p 'test_pretraining_v1_results.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

These commands check saved artifacts and replay token/verifier evidence; they do not generate additional samples. The original `run` command refuses an existing output directory, and there is no training command. The complete one-time sequence and frozen hyperparameters are in `training_protocol_v1/TRAINING_PROTOCOL.md`.

## Limits

Three correlated dev theorems in one family are a checkpoint-selection set, not an independent test. The protocol deliberately preserves the frozen prose and its minor grammatical defects. The planned Apple M2 / 8 GiB CPU training configuration has serialization and synthetic adapter/loss checks but has not been benchmarked with a Qwen backward pass. No statement about future training gains, memory peak or wall-clock training time is supported yet.
