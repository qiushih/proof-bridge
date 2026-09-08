# First LoRA pilot v1

**Did proof-step adherence improve while preserving verification and mathematical fidelity? Yes, on the six selection-dev rows.**

Selected **optimizer step 18**, using the unchanged lexicographic dev rule. Selected score: `[6, 6, 6, -18]`; step-zero score: `[6, 3, 6, 0]`. All ten checkpoints were evaluated before selection; ties choose the earliest step, including zero.

| Metric | Frozen pre-training baseline | Selected checkpoint |
| --- | ---: | ---: |
| Rocq verification | 6/6 (100.0%) | 6/6 (100.0%) |
| Verified + mathematically faithful | 6/6 (100.0%) | 6/6 (100.0%) |
| Verified + requested proof steps | 3/6 (50.0%) | 6/6 (100.0%) |

## Checkpoint comparison

| Step | Verified | Verified + faithful | Verified + requested steps | Selection score |
| ---: | ---: | ---: | ---: | --- |
| 0 | 6/6 | 6/6 | 3/6 | `[6, 3, 6, 0]` |
| 6 | 6/6 | 6/6 | 3/6 | `[6, 3, 6, -6]` |
| 12 | 6/6 | 6/6 | 5/6 | `[6, 5, 6, -12]` |
| 18 | 6/6 | 6/6 | 6/6 | `[6, 6, 6, -18]` |
| 24 | 5/6 | 5/6 | 4/6 | `[5, 4, 5, -24]` |
| 30 | 6/6 | 6/6 | 6/6 | `[6, 6, 6, -30]` |
| 36 | 6/6 | 6/6 | 6/6 | `[6, 6, 6, -36]` |
| 42 | 6/6 | 6/6 | 5/6 | `[6, 5, 6, -42]` |
| 48 | 6/6 | 6/6 | 5/6 | `[6, 5, 6, -48]` |
| 54 | 6/6 | 6/6 | 5/6 | `[6, 5, 6, -54]` |
| 60 | 5/6 | 5/6 | 4/6 | `[5, 4, 5, -60]` |

## Selected proofs and reviews

| Dev row | Rocq | Mathematical fidelity | Requested steps |
| --- | --- | --- | --- |
| pd01_A | PASS | FAITHFUL | MATCH |
| pd01_B | PASS | FAITHFUL | MATCH |
| pd02_A | PASS | FAITHFUL | MATCH |
| pd02_B | PASS | FAITHFUL | MATCH |
| pd03_A | PASS | FAITHFUL | MATCH |
| pd03_B | PASS | FAITHFUL | MATCH |

Both A/B step contracts were verified on **3/3** selected theorem pairs. Equivalent IH rewriting and `f_equal` followed by `exact`/`apply` can remain mathematically faithful even when the strict step contract differs.

## Training and custody

Fresh seeded rank-8 adapters were attached to the pinned Qwen2.5-Coder-0.5B-Instruct base; feasibility updates were not loaded. The unchanged loop trained on the frozen 24 rows for 10 complete epochs: 240 batch-size-one microbatches, accumulation four, exactly 60 AdamW updates. All 540,672 trainable parameters belong to query/value LoRA modules. Base weights remained unchanged. Ten adapter-only checkpoints retain optimizer/scheduler/RNG state; no base weights were overwritten.

Training-update time: **565.92s**. Total run time including loading, checkpoint I/O, integrity checks and dev evaluation: **771.62s**. Peak process RSS: **2.41 GiB**.

Each checkpoint received exactly six one-attempt argument-conditioned generations, with frozen Prompt v2, constrained-v1, greedy decoding, 256-token budget, and no repair. Evaluation saved/restored training RNG state. Step zero reuses the original six argument-conditioned outputs; no baseline regeneration or theorem-only control enters selection.

All 60 generated candidates have raw prompts, token IDs, constraint traces, latency, Rocq evidence and separate hash-bound assistant fidelity reviews. The sealed holdout was only byte-hashed for integrity; its content was not inspected, generated on or scored. Historical artifacts remain unchanged.

## Reproduction

Run from the repository root in the pinned environment. The completed run is checked without training or new generations:

```sh
.venv/bin/python -m pilot_training_v1.experiment check --replay-tokens --reverify
.venv/bin/python -m pilot_training_v1.report
.venv/bin/python -m unittest discover -s tests -p 'test_pilot_training_v1.py' -v
.venv/bin/python -m unittest discover -s tests -v
```

The original one-time command was `.venv/bin/python -m pilot_training_v1.experiment run`. It refuses an existing run; do not delete evidence to rerun under the same identity. `selected_checkpoint.json` identifies the selected adapter without copying or merging it. Protocol, dataset, prompt, decoder and checkpoint-selection code are unchanged.

## Limits

- Six correlated dev rows, three theorems in one family; selected on these same rows, not an independent estimate.
- A/B distinguish equivalent IH tactic realizations, not different high-level mathematical strategies.
- Assistant fidelity review, not independent human review.
- No new theorem-only or diagnostic generations; no claims about held-out generalization.
