# ProofBridge baseline v1

Model: [Qwen/Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct), revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`.

30 verified seeds are frozen as development data. The 12 new evaluation reference proofs all pass Rocq 9.2.0. No development examples or reference code were included in model prompts.

Each example received one greedy completion per condition, with no repair: 24 attempts total. Inference used float32 on CPU, four threads, seed 1729, and a 256-new-token limit. Model/tokenizer artifacts, dependencies, prompts, and data are pinned and hashed. Only whitespace and one complete outer code fence may be removed before verification.

| Input condition | Rocq PASS | Structure match | Assistant: faithful | Output tokens | Median generation time |
| --- | --- | --- | --- | --- | --- |
| theorem_and_informal | 0/12 (0.0%) | 0/12 | 0/12 | 743 | 3.74s |
| theorem_only | 0/12 (0.0%) | 0/12 | 0/12 | 2284 | 14.51s |

Paired verification outcomes: neither_pass: 12.

| Failure category | Theorem + informal proof | Theorem only |
| --- | --- | --- |
| FORBIDDEN_COMMAND | 11 | 12 |
| SYNTAX_ERROR | 1 | 0 |

Every candidate was rejected by the existing policy parser before Rocq compilation. These results measure a failure to produce an accepted proof body; they do not establish how often well-formed generated proofs would be mathematically correct.

Manual inspection found 11 outputs that exactly repeated the supplied English proof after `Proof:`, 1 that echoed the tactic instructions, and 12 prose-only reasoning outputs. Formal-fidelity reviews marked UNASSESSABLE: 24/24. Input copying is not credited as successful formalization.

Token-limit stops: theorem + informal proof 0/12; theorem only 8/12. This first fixed-prompt run is dominated by output-contract failures and does not establish the benefit of informal proofs for successful formalization.

Argument fidelity is separate from correctness. The reproducible proxy compares induction/binder position, supplied-premise use, congruence, symmetry, rewrite direction, and branch closing steps. A structure match is not a proof of English fidelity. Assistant reviews assess whether the output follows the supplied reference argument, including for theorem-only outputs where that argument was hidden; they are not independent human judgments.

This small diagnostic shares six argument families with development, although all 12 statements are new under binder renaming and definitional normalization. It cannot establish held-out-family generalization, exclude pretraining contamination, or show that a matching model output causally used the informal proof. Latencies are hardware-dependent; the fixed software/settings do not guarantee bit-identical outputs across platforms.

Files: `results.jsonl` contains immutable raw prompts, tokens, completions, proof bodies, verifier evidence, and the fidelity proxy; `assessed_results.jsonl` adds any hash-bound assistant reviews; `summary.json` contains machine-readable aggregates. `manifest.json`, `completion.json`, and `attempts.jsonl` record provenance and the one-attempt budget.
