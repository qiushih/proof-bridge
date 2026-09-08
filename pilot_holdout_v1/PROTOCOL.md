# Holdout comparison protocol v1

This protocol and its implementation are frozen before the reserved holdout is parsed or displayed. The user authorized this evaluation after the first pilot completed and selected step 18 using dev only. No holdout result may change that checkpoint, the prompt, decoder, verifier, training configuration or dataset. The holdout is consumed upon opening and cannot be reused as an untouched test after future tuning.

## Fixed scope

Compare the pinned Qwen/Qwen2.5-Coder-0.5B-Instruct base, revision `ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, against the same base with the already selected, unmerged step-18 rank-8 q/v LoRA adapter. The adapter SHA-256 is `004eac24b10840832b2babc290e535f91a5a887d68db5977f58e812c8e57e6f0`. All ten pilot checkpoints and the dev selection remain historical and unchanged; no alternative checkpoint is evaluated here.

Use the six holdout IDs in the existing split manifest, in manifest order, once for each model: **exactly 12 generations**. Each input contains the fixed formal theorem and that row's informal proof. Gold formal code, IDs, features and assessment metadata are excluded from the target input. The unchanged Prompt v2 demonstrations remain in context.

The user's later approved 12-generation plan narrows the older split manifest's post-training proposal, which also mentioned theorem-only controls. This evaluation uses argument-conditioned inputs only. The historical manifest is preserved byte-for-byte. There are no new theorem-only controls and no claim that this comparison alone establishes causal dependence on the informal text.

## Generation and model custody

Run base first and selected step 18 second in separate fresh CPU workers. Reuse the original `prompt_v2.experiment.Inference` loader and `constrained_v1.experiment.generate`. CPU float32, eager attention, four compute threads, one interop thread, seed 1729, deterministic algorithms, greedy decoding, one beam, inference KV cache, maximum 256 new tokens, one attempt and no repair remain identical to the frozen baseline. Each model receives a fresh vocabulary-mask cache with no gold-path prewarming.

For step 18, attach the unchanged LoRA wrapper, load only the selected adapter A/B tensors with exact keys/shapes, and verify loaded tensor hashes. All inference parameters have gradients disabled. No optimizer, training loop, checkpoint merge or weight update is permitted. Compare hashes of base parameters and adapters before/after inference, plus original model/checkpoint file hashes. Loading, latency and token usage are recorded separately; sequential order can affect host/cache timing.

The original reserved JSONL is parsed once by the authorized orchestrator, after writing a custody event. An identical opened snapshot is saved separately for both workers and future replay. The reserved source and its historical QA file remain unchanged; the old train/dev-only loader remains unchanged. Validate schema/membership and recompile all six fixed references with the existing verifier. If reference validation fails, record the consumed/interrupted state and stop; do not fix targets or tune the protocol. The original reserved QA details need not be opened.

Every generated body goes through the unchanged extraction policy and `verifier.verify` with a 10-second timeout. Journal each start/finish before proceeding. Any incomplete run or generation runtime error is retained and stops the workflow without retry, replacement, repair or parameter adjustment. A proof rejection is a valid measured outcome and does not stop the remaining planned attempts.

## Three separate assessments

1. **Rocq verification:** report PASS/FAIL and categories, kernel/assumption evidence, source hashes, and pass counts out of six for each model.
2. **Mathematical argument fidelity:** manually review whether a complete verified proof follows the supplied mathematical argument, including premise use, induction variable, base case and IH transport when applicable. Valid local renamings, definitional conversion, symmetry, equivalent rewrite directions and congruence can be faithful. A genuinely different strategy is UNFAITHFUL with a reason. Failed/incomplete proofs are NOT_ESTABLISHED. No automated induction-on-first-binder heuristic defines this judgment. These are assistant reviews; human-reviewed remains false.
3. **Requested proof-step adherence:** call the unchanged `training_protocol_v1.scoring.step_adherence` against each frozen reference. Its ordered trace ignores local naming, intros grouping, exact/apply of the same local, and optional forward arrow; it retains rewrite direction and simplification order. Report attempted matches and verified matches separately. A mismatch does not automatically imply mathematical unfaithfulness.

Primary comparison: verified requested-step count, alongside verified mathematically faithful and verified counts. Report trained-minus-base differences, all per-row outcomes and per-theorem paired A/B outcomes. Do not select a new checkpoint using these metrics. Missing reviews block the final report. Six rows form three correlated theorem pairs in one family; use descriptive counts, no significance or broad-generalization claim.

## Review masking and limitations

After all 12 attempts finish, shuffle their review records with fixed seed 53017 and assign opaque review IDs. The review packet includes the fixed statement, informal argument, reference proof, generated proof and verifier evidence. It excludes model labels, checkpoint IDs, latency, token traces and strict-step judgments. The mapping stays in a separate file and is not read by the reviewer until all 12 judgments are saved and hash-frozen. This is label-masked assistant review, not independent human or guaranteed blind review; proof style and prior knowledge may suggest identity.

The holdout has fresh statements and a family disjoint from pilot train/dev, but the public data specification documents historical exposure through seed_029. This comparison measures transfer to fresh instances under this pipeline; it does not establish unseen mathematics or absence of pretraining contamination. Subsequent tuning would require a fresh independent holdout.

## Reproduction

From the repository root with the existing pinned environment:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_pilot_holdout_v1.py' -v
.venv/bin/python -m pilot_holdout_v1.experiment freeze-protocol
.venv/bin/python -m pilot_holdout_v1.experiment run
.venv/bin/python -m pilot_holdout_v1.assessment packet
# Read review_packet.jsonl only; save all 12 judgments in masked_reviews.json.
.venv/bin/python -m pilot_holdout_v1.assessment freeze-reviews
.venv/bin/python -m pilot_holdout_v1.experiment check --replay-tokens --reverify
.venv/bin/python -m pilot_holdout_v1.assessment report
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pilot_holdout_v1.experiment freeze-results
```

One-time freeze/open/run commands refuse existing artifacts. After completion, `check`, `report` and tests use saved evidence and the opened snapshot; they do not generate, train or reopen the original reserved contents. Integrity checks may hash original bytes. Results, review decisions and the consumed-holdout custody record are kept under `results/holdout-v1`.
