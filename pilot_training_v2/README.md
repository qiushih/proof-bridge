# V2 two-arm training runner

This additive runner implements the frozen training_protocol_v2 experiment. The
runner-preparation task executes token-only checks and synthetic tests. It does
not load Qwen weights, run Qwen updates, generate model samples or open holdout
content. All original v1/v2 data, prompts, decoder, verifier and scoring rules stay
unchanged. A future explicit `run` starts the actual experiment.

## Implemented behavior

- Separate fresh processes for control (24 rows), then intervention (26 rows).
  Load only the pinned base; reset the seed before initializing rank-8 q/v LoRA.
  Compare initial base and adapter tensor hashes across arms before intervention
  updates. No feasibility or v1 adapter is loaded or merged.
- Consume each exact 240-ID frozen stream, four microbatches per update, 60 updates.
  Accumulation crosses the 26-row cycle boundary. No truncation, dropping, extra
  epochs or automatic retries. Both original numerical kernels and their config
  are hash protected; shared v1/v2 numerical settings must match before use.
- Reuse the tested completion-only loss, optimizer and update kernels. Audit that
  only 540,672 LoRA parameters are trainable; reject non-finite loss/gradients/weights.
  Log row IDs, loss, clipping, time, process memory and available system swap.
- Save adapters plus optimizer/scheduler/RNG state and the full data stream every
  six updates. Cursor metadata records consumed microbatches, completed cycles and
  the next cycle offset. Base model weights are hashed, never saved or modified.
- Generate exactly eight dev completions at each checkpoint: 80 per arm, 160 total.
  Use the existing frozen Prompt v2, constrained-v1 and verifier, greedy 256 tokens,
  one attempt, no repair. Each arm gets a fresh vocabulary cache; it is shared only
  across that arm's attempts. References never prewarm the live decoder cache.
  Save and restore Python/NumPy/Torch RNG and model mode around evaluation.
- Preserve journals and interruption evidence. Refuse any existing run/arm/checkpoint
  directory. The intervention cannot start until control completes. A verifier environment failure also stops the run after preserving the attempted output. A partial run
  cannot be selected; stopped workers are not silently retried.
- Require all 160 output reviews before separate selection of control and intervention.
  Reuse the eight frozen base outputs for step zero. Use exactly the frozen v2 score
  and success gates, including the new pair and original-six preservation. No v1
  trained checkpoint is eligible for a new arm's selection.

The designated hardware remains the frozen local 8-GiB M2 CPU/float32 setup. The
runner validates machine, OS, Python and pinned packages before model loading.
There are no hardware, hyperparameter, generation-budget or split override flags.
Moving the experiment to a server requires a separately versioned protocol.

## Safe readiness checks

From the repository root:

```sh
.venv/bin/python -B -m pilot_training_v2.experiment check-ready
.venv/bin/python -B -m unittest discover -s tests -p 'test_pilot_training_v2.py' -v
```

The prepared evidence is under results/v2-runner-preflight. It commits all 34 public
serializations, schedules, current hardware/runtime, step-zero prompt/token identity,
code/input hashes, a Rocq 9.2.0 reference smoke check and synthetic test results. pilot_training_v2/release.json freezes
the runner and preflight. It is labelled FROZEN_RUNNER_NOT_TRAINED and is separate
from any future trained-checkpoint release. No results/lora-pilot-v2 run is created
by preparation or tests.

One-time construction commands were `prepare --output <fresh-preflight-directory>`,
synthetic/regression validation, then `freeze-runner --output <same-directory>`.
Both mutation commands refuse existing evidence. Subsequent reproduction uses
check-ready and tests, not another preparation or an implicit training run.

## Future explicit training command — not run during implementation

```sh
.venv/bin/python -B -m pilot_training_v2.experiment run
```

This command performs BOTH arms (120 total optimizer updates and 160 dev generations)
and writes results/lora-pilot-v2. It does not inspect the holdout. An optional --output
changes only the fresh artifact directory; it is not a way to replace a failed run
without explicitly recording a separate experiment. There is no resume or repair.

## Reviews, selection and result freeze after the run

Create manual_reviews.json in the run directory with this structure:

```json
{
  "reviewer": "assistant",
  "human_reviewed": false,
  "raw_sha256_by_checkpoint": {"control/dev-step-0006/raw_generations.jsonl": "<sha256; include all 20 files>"},
  "reviews": [{"key": "<exact raw key>", "proof_body_sha256": "<sha256>", "status": "FAITHFUL", "reason": "<specific mathematical review>"}]
}
```

There must be exactly 160 unique, hash-bound reviews. Valid complete proofs are
FAITHFUL or UNFAITHFUL; an unresolved fidelity review is NOT_ESTABLISHED and counts
as zero for fidelity. Failed proofs must be NOT_ESTABLISHED. Review the supplied
argument and actual Rocq result. Equivalent successful tactics may be faithful
while mismatching requested steps. Neither a parser pass nor a matching tactic
trace alone establishes mathematical fidelity. All judgments in this workflow are
labelled assistant reviews, not independent human reviews.

```sh
.venv/bin/python -B -m pilot_training_v2.experiment check --replay-tokens --reverify
.venv/bin/python -B -m pilot_training_v2.experiment report
.venv/bin/python -B -m pilot_training_v2.experiment freeze-results
```

`report` selects only after both arms and all reviews are complete. It writes
assessed_results.jsonl, summary.json, selected_checkpoints.json and a concise report,
with original-six/new-two/all-eight metrics, paired successes and every success gate.
It refuses differing existing reports. `freeze-results` replays all 160 saved token
paths and Rocq outcomes and seals the results without new sampling. No automatic
fidelity judge or holdout evaluator is implemented.

## Validation limits

Tests use tiny freshly constructed tensors and predetermined reference fixtures;
these are clearly synthetic, live only in temporary directories, and are never
ProofBridge trained checkpoints or model-generated results. Real Qwen training and
160-output integration remain unexecuted in this implementation task. Tests establish
schedule/checkpoint/scoring mechanics, not learning improvement or new throughput.

The I/O guard permits reserved bytes only through the existing hash helper and
blocks old holdout/diagnostic access entirely. This is a workflow guard, not an OS
security sandbox. Frozen manifests and replay checks detect accidental drift; they
are not cryptographic signatures against a malicious editor. The fresh holdout has
correlated theorem pairs and shares algebraic structure with dev. No generalization
claim can be inferred from runner readiness.
