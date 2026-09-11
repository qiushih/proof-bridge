# V2 development baseline protocol

Freeze this protocol, the runner, input hashes and reuse audit before any new generation.

Use the frozen eight-row pilot-v2 development release. Compare the pinned
Qwen/Qwen2.5-Coder-0.5B-Instruct revision ea3f2471cf1b1f0db85067f1ef93848e38e88c25
with the previously selected step-18 adapter, SHA256
004eac24b10840832b2babc290e535f91a5a887d68db5977f58e812c8e57e6f0.
Do not select a different checkpoint, train, or change any existing data or code.

Reuse the six argument-conditioned base outputs from results/pretraining-dev-v1
and the six step-18 outputs from results/lora-pilot-v1/dev-step-0018. Reuse requires
identical theorem, prompt messages, rendered prompt, token IDs, model identity,
settings and hash-bound assessed record. Exclude theorem-only outputs and other
checkpoints. Retain each original source and its mathematical review unchanged.

Run exactly four new generations, ordered base A, base B, step18 A, step18 B,
on pd04_A/pd04_B. Each model uses a fresh process and vocabulary; grammar caches
may be shared between its two attempts. Use frozen Prompt v2 and the unchanged
constrained_v1.experiment.generate. CPU float32, eager attention, four threads,
one interop thread, deterministic algorithms, seed 1729, greedy decoding, one
beam, use_cache=true, 256 new tokens including EOS, one attempt, zero repair.
Use the same effective GenerationConfig as the original baseline. The result has
sixteen model/example records, of which only four are newly generated. No new
theorem-only controls are authorized; do not claim causal informal-proof dependence.

Log an attempt before calling generate and refuse overwrite/resume/retry of a
started run. On a generation exception, preserve the partial output and stop.
Load the selected adapters without merging; disable gradients and use inference
mode. Hash in-memory base and adapter parameters before/after, and verify on-disk
model/adapter hashes. No optimizer, backward pass or adapter update is permitted.

Every new output uses the existing extraction and verifier with a 10-second
compiler timeout. Invalid candidates still count in the denominator. Report raw
verification status and failure category, mathematical fidelity, and requested
proof-step adherence separately. Mathematical fidelity is assistant-reviewed and
hash-bound, not inferred from the exact tactic trace. Equivalent successful tactic
choices may remain faithful while mismatching requested steps. Failed proofs are
NOT_ESTABLISHED; they cannot count as verified faithful. A/B pairs succeed only
when BOTH outputs verify and satisfy the relevant fidelity/step criterion.

Before reporting, replay all sixteen proofs through Rocq without generation.
Replay prompt and completion tokens through the frozen decoder when checking.
Report the original six, new two, and combined eight separately for each model;
include per-theorem paired results. Report latency/token usage by source subset,
without interpreting mixed historical timings as a hardware speed benchmark.
Record current runtime versions; require pinned Python and model-library versions.

No holdout or diagnostic example files may be opened. This is development evidence
in a historically exposed family, with only one new theorem pair. The original six
were used to select step 18; their scores are selection-dev results. No independent
generalization estimate or training intervention is established by this baseline.
No v2 checkpoint-selection rule is created. A later training experiment requires
its own frozen protocol and fresh holdout.

Reproduction from the repository root:

    .venv/bin/python -B -m baseline_dev_v2.experiment check --replay-tokens --reverify
    .venv/bin/python -B -m unittest discover -s tests -p 'test_v2_dev_baseline.py' -v

One-time workflow: `freeze`, `run`, then `assess` after recording four assistant
reviews in new_reviews.json. Those commands refuse an existing output or result.
Each command accepts --output for a separate result directory. Rechecking saved
evidence performs no generation. Never delete the run marker to obtain another attempt.
