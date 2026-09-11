# ProofBridge v2 controlled training protocol

This preparation freezes a proposed two-arm experiment; it does not authorize or
perform training. Its hypothesis is that one structurally different right-zero
A/B pair improves premise placement and requested successor steps on development
proofs. It is not a claim that two additional rows will generalize.

## Fixed inputs and hardware

Use Qwen/Qwen2.5-Coder-0.5B-Instruct, revision
`ea3f2471cf1b1f0db85067f1ef93848e38e88c25`, validated by the original model lock.
Base weight SHA256 is
`f9523886352217ded3aeeef552b381af79d568c6d49a4b9e423288cea56b0a44`.
Each arm initializes new rank-8 q/v adapters from that base. Neither the selected
v1 step-18 adapter nor feasibility updates may initialize an arm.

The designated actual machine is the local MacBook Air Mac14,2, Apple M2,
8 CPU cores, 8 GiB unified memory. Use CPU float32, eager attention, four Torch
threads and one interop thread, deterministic algorithms, no MPS/CUDA,
quantization or mixed precision. `hardware.json` records the current observed OS,
architecture and runtime, without device serial identifiers. If the machine or
software changes, revise and freeze a new protocol before training; do not silently
port this experiment to a rented GPU. Current host feasibility is supported by the
previous completed local pilot, not a new forward/backward run in this preparation.

Use Python 3.13.7, Torch 2.9.0, Transformers 4.57.1, tokenizers 0.22.2,
Safetensors 0.8.0 and NumPy 2.3.4; the full existing requirements lock remains pinned.
Prompt v2 short_3shot, its three demonstrations, constrained-v1, the verifier and
all historical releases remain byte-identical. Use the existing prompt builder,
serialization and loss definitions; never include IDs, contracts, reference QA,
family labels or held-out targets in the model input.

## Data arms and exact exposure

Control contains the original 24 training rows, copied byte-for-byte.
Intervention contains those same bytes followed by exactly two new rows,
`ptv201_A/B`, in `right_zero_variants`. They prove
`forall n m p : nat, m = p -> (n + 0) + m = n + p` by induction on n,
using H in the base case and IH rewriting versus successor congruence. The
right-zero residual is now inside the first operand of a larger addition with a
variable suffix. Existing premise-base training rows instead end a sum with + 0.
The new pair is not a new independent training family; it also shares a context
with historical seed_021. A/B are equivalent mathematical inductions with different
requested steps. An arbitrary early rewrite of H is not categorically invalid:
for this theorem it substitutes m, but still leaves a right-zero obligation.

Both arms have **60 optimizer updates, 240 microbatches, 4 examples per update**.
The budget is update-based, not an equal number of epochs. For each arm separately,
shuffle its sorted IDs with `random.Random(1729 + cycle_index)`; concatenate full
cycles and take the first 240 IDs. Control has 10 complete 24-row cycles.
Intervention has nine complete 26-row cycles and the first six IDs of cycle 9.
Carry accumulation across cycle boundaries. Never drop, pad, repeat a failed batch,
shuffle on dev outcomes, or extend a budget. `schedule.json` freezes all 240 IDs,
counts per row, update groups and supervised token totals before training.
Uniform row sampling means additions displace some original-row exposures; this
experiment measures that small dataset change, not pure extra data at fixed
original-row exposure. It is update-matched, not FLOP-, wall-time- or token-matched.

Run control first, then intervention, in separate fresh processes, resetting Python,
NumPy and Torch seeds to 1729 for each arm. The initial adapter tensors must match
across arms. No seed sweep, automatic repair/retry, resumption or early stopping.
An interrupted run is incomplete evidence; preserve it and explicitly version any
later replacement run rather than hiding the interruption.

## SFT and LoRA configuration

Use ordinary unmerged LoRA on `self_attn.q_proj` and `self_attn.v_proj` in all 24
layers: 48 wrappers and 540,672 trainable parameters. Rank 8, alpha 16, dropout 0,
A initialized Kaiming uniform with a=sqrt(5), B zero; freeze all base parameters.
Activation checkpointing is enabled with use_reentrant=false; training KV cache is
disabled. Verify the trainable allowlist and hash base tensors before/after each arm.

Serialize with unchanged Prompt v2 and the pinned tokenizer chat template, then
append the proof body and one EOS. Maximum total length 768; no packing or
truncation. All prompt, demonstration and role tokens have label -100. Supervise
only target proof tokens and EOS; any padding has attention 0 and label -100.
Use the existing per-example mean causal cross-entropy and mean four microbatch
losses for each optimizer update. Batch size is 1, gradient accumulation is 4.

Optimizer: AdamW, adapter parameters only; learning rate 0.0002, betas (0.9, 0.999),
epsilon 1e-8, weight decay 0, foreach=false, fused=false; constant schedule, zero
warmup, gradient norm clipping 1.0. Save adapters, optimizer/scheduler state, RNG
states, update/microbatch position, actual ID schedule and input/protocol hashes
after steps 6, 12, 18, 24, 30, 36, 42, 48, 54 and 60. Preserve per-microbatch loss,
per-update loss/gradient norm, time, memory, exposure IDs and finite-value checks.
Never write or merge base weights. No new optimizer loop is supplied by this task:
a future runner must consume this schedule instead of the v1 fixed-24-row epoch loop.

## Eight-row checkpoint selection

Only the unchanged eight rows `pd01_A/B` through `pd04_A/B` may select checkpoints.
The old v1 six-row scorer stays untouched. `training_protocol_v2.scoring` implements
a separate rule. At each of the ten trained checkpoints per arm, generate one proof
for each dev row, ordered by those IDs: **160 new dev generations total** across
two arms. Step zero reuses the eight base records in the frozen v2 baseline after
prompt/token/model/settings identity checks. Step 18 from v1 remains a historical
comparator only. It is not a candidate checkpoint for either new arm.

Use frozen Prompt v2, constrained_v1.experiment.generate, greedy one-beam decoding,
seed 1729, 256 new tokens including EOS, one attempt, zero repair, unchanged
10-second-timeout verifier. Save raw text, token IDs, prompt messages/rendering,
model/adapter hashes, settings, latency, token usage and verifier diagnostics. All
failures and token limits remain in the denominator.

For each arm select the lexicographic maximum over step zero and its ten checkpoints:
1. Number of verified and mathematically faithful proofs among eight rows.
2. Number of verified proofs matching the requested steps.
3. Number of Rocq-verified proofs.
4. Negative optimizer step (earliest wins ties, including zero).

There is no loss, latency, theorem-only, diagnostic or holdout tie-break. Every
output needs a hash-bound mathematical review under the same rubric before
selection; unresolved reviews cannot count as faithful. Assistant reviews are
labelled as such and are not independent human review. Consistent IH rewriting
and f_equal/exact may express the same valid induction while differing in the
requested step contract. Record failed proofs as NOT_ESTABLISHED. Do not infer
fidelity from exact trace matching alone. The base action must fit the supplied
argument: computation for pd01–03 and use of the premise for pd04, or a valid
mathematically equivalent implementation of that argument.

Report verification, verified mathematical fidelity, attempted step matches,
verified step matches, and their three-way conjunction. Split all metrics into
original-six, new-two and all-eight. Report paired A/B success separately: both
variants must verify, be faithful and match their respective steps.

## Predeclared development success

Compare the selected intervention checkpoint with the selected control checkpoint.
Call the targeted intervention successful on dev only when ALL gates hold:
- More verified, mathematically faithful, step-adherent proofs overall.
- Verification and verified mathematical fidelity are each no lower than control.
- All original six rows remain verified and mathematically faithful.
- Both new pd04 variants verify, are faithful and follow their requested steps.

Report every score and gate even when the intervention fails. These are experiment
decision criteria, not a statistical significance test or a new selection tie-break.
One seed, one added training theorem and one newly covered dev theorem cannot
establish robust causal generalization or independence from the informal input.

## Fresh holdout and custody

Prepare six new rows (three A/B theorem pairs) in the separately allocated
`conditional_repeated_sum_contraction` family. Their exact statements, prose,
reference bodies and detailed QA stay under `data/pilot-v2-training/reserved/`.
All renamed, reversed, definitional, repeated-context and proof variants stay in
that family. Never reassign related variants to training or dev. This family is
excluded from both arms and from checkpoint selection.

The protocol design and public training addition are committed in `design.json`
BEFORE constructing or reviewing those fresh references. Construction necessarily
reads them once for assistant review, Rocq verification, token/loss-mask checks,
reference-removal QA and duplicate/family review. Record that pre-seal exposure.
The final split manifest and release seal are written only after all QA passes.
After sealing, checks may hash reserved bytes but must never parse, print,
recompile, tokenize, export or generate from them during preparation/training.
There is no holdout loader or inference entry point in this release. Old holdout
and diagnostic content is not opened, even for hashes.

The new ledger separates explicit right-zero training, successor/reassociation dev,
old conditional permutation and conditional repeated-sum contraction. These
identities share induction mechanisms and can be connected using associativity.
The fresh holdout is a fresh-instance, family-separated set under a manually
reviewed grouping, NOT independent unseen mathematics. Public train/dev and the
30 historical development statements receive exact/definitional/orientation/binder
checks. No exact hidden-statement scan against unopened old holdout/diagnostic files
is claimed. All freshness and family limitations must accompany future results.
The three fresh holdout theorem pairs are correlated variants in one family.

Only after BOTH arms and selections are complete and hashes are committed may a
separately authorized final evaluation open the holdout once. Predeclare three
models (pinned base, selected control, selected intervention), argument A and B for
each of its three theorems, one attempt, no repair, 256 tokens, same prompt/decoder/
verifier: 18 outputs. Report all three metrics and pair counts for all models.
Holdout outcomes must not revise data, settings, selections, or the success rule.
Even a dev-negative experiment retains this planned final comparison if training
and both selections complete. Further tuning requires another fresh holdout.

## Reproduction and task boundary

From the repository root:

```sh
.venv/bin/python -B -m training_protocol_v2.preparation check
.venv/bin/python -B -m training_protocol_v2.preparation check --reverify-public
.venv/bin/python -B -m unittest discover -s tests -p 'test_training_protocol_v2.py' -v
```

`check` validates the frozen schema, schedules, hashes and public selection
baseline without loading model weights or parsing the reserved holdout.
`--reverify-public` recompiles and replays tokenizer paths for only 26 train plus
8 dev references. Initial construction is recorded in `preparation.json`; it is
not repeated after sealing. The preparer accepts only an unsealed fresh draft for
holdout construction and refuses existing output. Do not use the sealed holdout
as a new draft to reconstruct or revise this experiment. The sealed files and
recorded source/token hashes are the reproduction evidence for reserved QA.

This task ends with frozen data, protocol, tested selection/scheduling definitions
and reference QA. Model generations, model-weight loads and training updates in
this preparation are all zero. Actual future training must be explicitly started
in a later task; the previous v1 runner is not automatically a v2 runner.
