# Constrained decoding with frozen Prompt v2

**Success criterion met:** verified faithful proofs on the common 25-seed subset changed from **21/50 to 34/50** (both conditions pooled).

Exactly 60 new development generations: 30 seeds × two conditions, one attempt each, no repair. Prompt v2, model/tokenizer, verifier, data, greedy configuration, and 256-new-token budget were preserved. No diagnostic examples were used.

Before inference, all 30 exact reference token paths including EOS passed the mask and all 30 reference proofs compiled in Rocq. Nine tests covered grammar, required introductions, invented references, IH/branch scope, malformed commands, and the command limit. `preflight.json` and the run manifest bind this evidence to the decoder sources before generation.

## All 30 development seeds

| Version | Input | Format | Invalid refs | Missing intros | Rocq PASS | Verified + faithful | Median latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unconstrained_v2 | theorem_and_informal | 17/30 | 5/30 | 7/30 | 16/30 | 16/30 | 2.15s |
| unconstrained_v2 | theorem_only | 20/30 | 9/30 | 0/30 | 12/30 | 12/30 | 1.40s |
| constrained_v1 | theorem_and_informal | 30/30 | 0/30 | 0/30 | 23/30 | 23/30 | 2.31s |
| constrained_v1 | theorem_only | 30/30 | 0/30 | 0/30 | 18/30 | 18/30 | 1.30s |

## Common 25 seeds; demonstrations/equivalents excluded

| Version | Input | Format | Invalid refs | Missing intros | Rocq PASS | Verified + faithful | Median latency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| unconstrained_v2 | theorem_and_informal | 13/25 | 5/25 | 6/25 | 12/25 | 12/25 | 2.27s |
| unconstrained_v2 | theorem_only | 15/25 | 9/25 | 0/25 | 9/25 | 9/25 | 1.40s |
| constrained_v1 | theorem_and_informal | 25/25 | 0/25 | 0/25 | 19/25 | 19/25 | 2.31s |
| constrained_v1 | theorem_only | 25/25 | 0/25 | 0/25 | 15/25 | 15/25 | 1.30s |

Format is the unchanged verifier policy parser applied to raw output after edge whitespace trimming. Invalid-reference and missing-introduction counts use the development audit's specific root-cause definitions; invented excess H introductions count as invalid references. Rocq PASS and reviewed fidelity are distinct. Full counts, percentages, mean/median/total latency, token usage, failures, and paired outcomes are in `summary.json`.

There were **13 gains and 0 losses** in verified faithful proofs. All 19 remaining failures reached Rocq: 18 proof errors and one incomplete proof. Three informal-condition attempts (027–029) repeated tactics to the 256-token limit. Their local names were in scope; induction/premise reasoning remained incorrect.

Mean generation latency with informal proofs increased from **2.14s to 4.53s**; theorem-only changed from **1.77s to 1.52s**. The long repeated outputs explain why the mean cost grows more than the median. Vocabulary construction took 0.66s separately; mask computation totaled 41.50s across all attempts.

## Fidelity and limits

All 60 new outputs were manually reviewed by the assistant against the same frozen informal arguments and reference code, including the theorem-only outputs. Equivalent local equality transport and induction-hypothesis congruence count as faithful; different strategies are flagged. These are not independent human reviews. Existing deterministic structure scores are retained separately in every record.

The grammar observes only binder counts and generated code; it does not inspect reference proofs, informal text, goal states, or compiler feedback. It enforces proof-versus-nat reference kinds and branch scope, not mathematical applicability. An in-scope IH may still be unusable in the current goal. Correct syntax can still produce a wrong or unfaithful proof.

Latency includes token-mask computation, but excludes model/vocabulary setup and verifier time. Vocabulary setup is reported separately; cache warm-up and output length affect timing. Unconstrained latency comes from the earlier saved run. These reused development results do not establish held-out generalization or causal use of every informal step.

Reproduction and constraint details: `constrained_v1/README.md`. Raw outputs and mask traces: `results.jsonl`. Manual judgments: `argument_reviews.json`. Combined machine-readable records: `assessed_results.jsonl`. No raw output or previous artifact was repaired or overwritten.
