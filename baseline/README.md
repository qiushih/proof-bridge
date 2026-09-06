# First ProofBridge baseline

This is a zero-shot comparison using one released small code model,
[Qwen2.5-Coder-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B-Instruct).
The model has upstream instruction tuning; no ProofBridge fine-tuning, parameter
updates, prompt search, output repair, or candidate selection is performed.

The original 30 examples are frozen as development data in
`data/development/pairs.jsonl`. The 12 new evaluation examples cover three
definitional arguments, four equality-premise arguments, and five single
inductions. Every reference proof passes the existing restricted verifier.

Both conditions receive the identical environment/tactic contract and formal
theorem. `theorem_and_informal` additionally receives the curated informal proof;
`theorem_only` receives no proof, method, title, features, or reference code.
Neither condition includes development demonstrations. Each uses a fresh chat
context and exactly one greedy generation with no repair.

## Reproduce the recorded experiment

The recorded runtime is Python 3.13.7 on Apple Silicon macOS. Rocq remains pinned
at 9.2.0 / Stdlib 9.1.0. Model execution uses PyTorch 2.9.0, Transformers 4.57.1,
float32 CPU inference with eager attention, four compute threads and one interop
thread. All Python dependency versions are pinned in `requirements.lock`.
The model revision and SHA-256 hashes of every downloaded artifact are in
`model.lock.json`. No remote Python code is loaded.

From the repository root:

```sh
python3 scripts/setup_rocq.py --check
python3 scripts/verify_seeds.py --check
python3 -m baseline.dataset --check
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip -r baseline/requirements.lock
.venv/bin/python -m pip check
.venv/bin/python -m baseline.setup_model
.venv/bin/python -m baseline.setup_model --check
.venv/bin/python -m baseline.run --output .cache/baseline/reproduction-001
python3 -m baseline.report --run-dir .cache/baseline/reproduction-001 --reverify
python3 -m unittest discover -s tests -v
```

Only dependency/model setup needs network access. The model is approximately
1 GB on disk; float32 loading requires additional RAM. The runner sets offline
mode and loads only the local, checksum-verified model files. Use the same
Python/runtime/platform for the closest reproduction. Greedy deterministic
decoding and pinned inputs do not promise identical floating-point results
across different hardware or libraries. Latency is inherently machine-dependent.

The output directory must not exist. The runner refuses to overwrite results
or resume an interrupted attempt; use a new directory for an explicitly
separate reproduction run. The checked-in `results/baseline-v1` is the first
run, not the best of several runs. An exception during generation is recorded
as an attempted generation error with an empty candidate sent to the verifier;
it is never silently retried.

To inspect and reverify the **saved outputs without generating again**:

```sh
python3 -m baseline.report --run-dir results/baseline-v1 --reverify
```

That command recomputes the report and attaches the recorded assistant reviews.
It makes no model calls and does not alter raw generation records. Regression
tests also reverify all saved candidates and evaluation references.

`python3 -m baseline.dataset` was the preparation command: it created the
development snapshot once and compiled all evaluation reference bodies. Its
`--check` mode validates the snapshot and evidence and recompiles references
without writes. The experiment fingerprints the resulting reference report;
do not refresh it after a recorded run. An intentional data/protocol revision
belongs in a new experiment, retaining the old artifacts.

## Recorded generation and verification

`config.json` fixes a 256-new-token cap, greedy decoding (`do_sample: false`,
one beam), seed 1729, one attempt per example/condition, no repair, and the
existing ten-second verifier timeout. EOS ends generation; hitting the token
cap is recorded as `max_new_tokens`, with the full truncated text preserved.
The prompt text is fixed in `prompt.txt`, and the tokenizer's chat template is
pinned as part of the model artifacts.

Output extraction strips whitespace and, only if it encloses the entire
response, one outer `coq`/`rocq` code fence. It never removes prose around a
code block, theorem declarations, `Proof`/`Qed`, forbidden tactics, or helper
references. Both raw text and the resulting candidate are saved. This is format
extraction, not semantic repair. Every candidate goes through `verifier.verify`;
policy failures may be rejected before the compiler starts, exactly as in the
existing verifier. All failures remain in the pass-at-one denominator.

| Artifact in `results/baseline-v1` | Contents |
| --- | --- |
| `manifest.json` | Model revision/artifact hashes, full effective generation configuration, runtime/package versions, prompt/data/code fingerprints, parameter count, and model-load latency |
| `attempts.jsonl` | One start/finish event for each of the 24 generation calls |
| `results.jsonl` | Immutable raw results: prompt messages/rendered text/hash/token IDs; generated text/token IDs; candidate body/hash; verifier result and failure category; structure fidelity proxy; latency and token usage |
| `completion.json` | Completion status and hashes of immutable run artifacts |
| `argument_reviews.json` | Assistant judgments and rationales, bound to raw-result and candidate hashes |
| `assessed_results.jsonl` | Complete machine-readable results with the assistant judgments attached to `argument_fidelity` |
| `summary.json` | Per-condition pass-at-one, failure counts, fidelity counts, token totals, latency aggregates, and paired outcomes |
| `REPORT.md` | Concise report and practical limitations |

Generation latency includes the `generate` call and output decoding, excluding
model load, tokenization, and verification. The manifest separately records load
time; verifier results contain their own elapsed time. Completion token counts
include generated special tokens such as EOS. Input/output token IDs make the
counts auditable. There is no API charge or provider-reported billing usage.

## Argument fidelity

Correctness and following the informal argument are distinct measurements.
`argument_fidelity` always includes a deterministic **structural proxy**:
`STRUCTURE_MATCH`, `STRUCTURE_MISMATCH`, or `UNASSESSABLE`. It compares:

- Whether induction is used and the zero-based quantified-binder position,
  allowing local variables to be renamed.
- Whether a supplied equality premise is actually used.
- Congruence/symmetry counts, rewrite reference types and directions, and branch
  closing steps. Direct `apply H` and `exact H` share the same structural role.

The score is the fraction of matching structural criteria, not a probability of
semantic fidelity. Rejected grammar gives a null score. A structure match can
still fail Rocq, and a valid alternative argument can mismatch. In particular,
eval 010 can be proved by directly rewriting its premise, while its supplied
argument deliberately uses induction.

The first run also has an **assistant review**, recorded separately with a
reason for every output: `FAITHFUL` follows the supplied argument with the
essential steps; `PARTIAL` follows meaningful parts but omits or breaks steps;
`DIVERGENT` follows a different argument or contradicts the intended reasoning;
`UNASSESSABLE` contains no sufficiently coherent proof to compare. These are
assistant judgments, not independent human review. For theorem-only outputs,
the comparison is to the same hidden reference argument; matching does not
show that the model used an informal proof it never received. Reproduction runs
have only the automatic proxy until separately reviewed.

## Scope of the comparison

The development/evaluation audit rejects statement duplicates under whitespace
and binder renaming and under the existing definitional normalizer. It finds
no such overlap. The new examples intentionally remain in the same small
fragment and share six generation families with development. This is a
statement-disjoint diagnostic comparison, **not a leakage-safe family holdout**.
The earlier family-boundary rule still applies to a future training/evaluation
split; this experiment does not train on development records or include them
as demonstrations. No claim is made about pretraining contamination, family
generalization, or statistical significance from twelve examples.
