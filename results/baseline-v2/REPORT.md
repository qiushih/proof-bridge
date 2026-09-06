# Prompt v2 diagnostic evaluation

Frozen prompt: **short_3shot**, using 3 fixed development demonstrations (seed_002, seed_010, seed_016). Selection used only development metrics; freeze time 2026-09-06T23:05:42.588484+00:00 precedes evaluation start 2026-09-06T23:07:04.074587+00:00.

The same Qwen2.5-Coder-0.5B-Instruct revision, existing verifier, greedy settings, and 256-new-token budget were used. Each of the unchanged 12 examples received one attempt per condition: 24 generations, no repair or fine-tuning.

| Version | Target input | Raw format | Rocq verified | Verified structure match |
| --- | --- | --- | --- | --- |
| v1 | theorem_and_informal | 0/12 | 0/12 | 0/12 |
| v1 | theorem_only | 0/12 | 0/12 | 0/12 |
| v2 | theorem_and_informal | 6/12 | 4/12 | 2/12 |
| v2 | theorem_only | 9/12 | 5/12 | 1/12 |

V2 paired correctness: both_pass=3, informal_only_pass=1, theorem_only_pass=2, neither_pass=6.

| Target input | Failure categories | Median generation time | Completion tokens |
| --- | --- | --- | --- |
| theorem_and_informal | {"SYNTAX_ERROR": 5, "FORBIDDEN_HELPER": 1, "INCOMPLETE_PROOF": 1, "PROOF_ERROR": 1} | 1.58s | 230 |
| theorem_only | {"SYNTAX_ERROR": 3, "PROOF_ERROR": 4} | 1.38s | 188 |

Assistant semantic-fidelity reviews (not independent human review): theorem_and_informal: {"DIVERGENT": 3, "PARTIAL": 5, "FAITHFUL": 4}; theorem_only: {"DIVERGENT": 7, "FAITHFUL": 4, "PARTIAL": 1}.

Format compliance tests raw code with whitespace trimming only; fences, explanations, and declarations fail. Rocq verification uses precisely v1's output extraction and unchanged policy/kernel checks. A format-compliant proof can still fail mathematically.

Verified structure match requires both a verified proof and agreement with the fixed argument-structure proxy. It is stricter than correctness but does not certify English fidelity. A valid alternative can mismatch the proxy. The theorem-only condition is scored against the same hidden reference argument.

Both conditions have identical development demonstrations, including their informal arguments; only the final target's informal proof differs. Evaluation shares arithmetic families with those demonstrations and was already used for the v1 diagnostic. This comparison does not establish unseen-family generalization, exclude pretraining contamination, or demonstrate causal use of every informal step.

`results.jsonl` is the immutable record of every prompt, output, proof body, verifier result, format/fidelity score, token IDs, usage, settings, and latency. `assessed_results.jsonl` attaches available assistant reviews. `report_summary.json` records the comparison; v1 files and all datasets remain unchanged.
