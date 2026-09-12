# ProofBridge v2 final holdout runner

The predeclared final comparison uses the pinned Qwen2.5-Coder-0.5B-Instruct
base, control update 36, and intervention update 36. These selections already
scored 8/8 on all three development measures. They cannot be revised here.

Exactly six holdout rows (three A/B pairs) receive one argument-conditioned
generation per model: 18 new outputs total. Run models in base/control/intervention
order and rows in the frozen public manifest order. Each model uses a fresh
process and decoder vocabulary. Use unchanged Prompt v2, constrained-v1,
CPU float32, eager attention, seed 1729, greedy one-beam decoding, 256 new tokens
including EOS, and the unchanged Rocq verifier with a 10-second timeout. No
training, reselection, repair, retries, resumption, or additional conditions.

Before opening, public-fixture tests must pass, both training arms and selections
must be frozen, and their release/selection/adapter hashes must be committed.
The runner, tests, public preflight and frozen protocol must also be committed
before `run`. A custody record precedes the only original holdout content read.
Workers and subsequent reviews use its hash-identical snapshot. Checks may hash
new reserved bytes; old holdout and diagnostic content are denied even for hashing.
The Python I/O guard enforces this workflow for normal file APIs; it is not an
operating-system security boundary.

Only formal_statement and informal_proof from each row enter the model prompt.
Reference code, step contracts and metadata never enter model input. Reference
proof re-verification occurs after generation; no reference token paths prewarm
inference. Every attempt is journaled before generation and saved before checking
for runtime/environment errors. Ordinary proof failures remain in the denominator;
runtime failures stop the experiment without retries. Preserve any interrupted
run. All base and adapter weights are frozen and hashed before/after inference.

Mathematical fidelity requires an explicit assistant review bound to every raw
proof hash. A verified opposite IH tactic or symmetric rewrite may preserve the
mathematical induction but fail the requested-step score. Failed proofs are
NOT_ESTABLISHED. Verified but unresolved reviews may also be NOT_ESTABLISHED and
count as zero faithful proofs. Do not infer fidelity from exact trace matching.
Report each measure separately, their verified/faithful/step conjunction, and
paired success requiring both A and B. Reviews are not independent human review.

The six rows are three correlated pairs in one manually separated family. The
references were assistant-curated and inspected during original construction.
These are fresh held-out instances, not independent unseen mathematics. The
dev-negative intervention decision remains unchanged regardless of these results.
No causal informal-proof-dependence claim follows without theorem-only controls.
Opening consumes this holdout; further tuning requires another fresh holdout.

## Commands

From the repository root, before any holdout opening:

```sh
.venv/bin/python -B -m pilot_holdout_v2.experiment preflight
.venv/bin/python -B -m pilot_holdout_v2.experiment freeze
git add pilot_holdout_v2 tests/test_pilot_holdout_v2.py results/holdout-v2-preflight
git commit -m "Freeze v2 final holdout runner before opening"
.venv/bin/python -B -m pilot_holdout_v2.experiment check-ready
.venv/bin/python -B -m pilot_holdout_v2.experiment run
```

After all 18 outputs finish, create `results/holdout-v2/manual_reviews.json` with
`reviewer: assistant`, `human_reviewed: false`, and `raw_sha256_by_checkpoint`
mapping each `base|control|intervention/raw_generations.jsonl` to its SHA256.
Its `reviews` list must contain all 18 unique keys, each with `key`,
`proof_body_sha256`, `status` and a nonempty mathematical `reason`.

```sh
.venv/bin/python -B -m pilot_holdout_v2.experiment report
.venv/bin/python -B -m pilot_holdout_v2.experiment freeze-results
```

Reproduce saved evidence without generating or updating weights:

```sh
.venv/bin/python -B -m pilot_holdout_v2.experiment check --replay-tokens --reverify
.venv/bin/python -B -m pilot_holdout_v2.experiment report
.venv/bin/python -B -m unittest discover -s tests -p test_pilot_holdout_v2.py -v
```

`freeze-results` replays all 18 token paths and verifier outcomes, recompiles
the six frozen references, then seals raw generations, reviews, metrics, report
and validation. Do not delete markers or output directories to repeat a run.
