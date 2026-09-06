# Prompt v2: development-only selection

This experiment changes prompting for the existing
`Qwen/Qwen2.5-Coder-0.5B-Instruct` baseline. It uses the same pinned model files,
dependencies, verifier, greedy decoding, seed, CPU/float32 settings, and
256-new-token limit. No dataset, verifier, v1 artifact, or model parameter is
modified. There is no repair or fine-tuning.

The complete v1 release remains intact at `proofbridge-baseline-v1`. The
selected v2 prompt and its development evidence are frozen in `selected.json`
and `frozen.json` before any v2 evaluation records are loaded.

The annotated Git tag `proofbridge-prompt-v2` records this prompt-selection
checkpoint before the diagnostic evaluation run. The selected candidate is
`short_3shot`, with demonstrations 002, 010, and 016.

## Development protocol

`plan.json` and `system.txt` define three candidates before generation:

| Candidate | Fixed demonstration seeds |
| --- | --- |
| `short_2shot` | 002, 016 |
| `short_3shot` | 002, 010, 016 |
| `short_4shot` | 002, 010, 016, 027 |

Each demonstration is copied exactly from the frozen development JSONL,
including its canonical proof body. The shorter system instruction explicitly
requests code only, no Markdown or prose, no declarations/imports, and only the
existing restricted tactics. Demonstrations are ordinary user/assistant chat
turns, not extra theorem declarations in the candidate file.

Both target conditions share identical demonstrations, including the
demonstrations' informal proofs. Only the final target's informal argument is
removed in `theorem_only`. Demonstrations are fixed per candidate and are never
retrieved or adapted using a target or evaluation result.

All three candidates are measured on all 30 development seeds under both
conditions: **180 generations**, one per candidate/seed/condition. Full-set
metrics are reported, but prompt selection uses a shared subset that excludes
the union of demonstration IDs and their definitional equivalents:
**001, 002, 010, 016, 027**. The remaining 25 seeds give 50 selection outputs per
candidate. These are development scores, not independent generalization claims.

The predeclared selection order is lexicographic:

1. Most raw-format-compliant outputs on the common subset.
2. Most Rocq-verified outputs on that subset.
3. Most outputs that verify and match the fixed argument-structure proxy.
4. Fewer demonstrations, then earlier candidate order.

`freeze_selection` applies that rule automatically using development records
only. It writes the selected concrete chat prefix and a checksum manifest
covering the plan, prompt code, model/runtime pins, original development data,
and completed development results. Existing freeze files cannot be replaced.
Evaluation fails before loading examples unless those frozen files exist and
all hashes match. Its manifest records the freeze hash and a later start time.

## Three separate measurements

| Measurement | Meaning |
| --- | --- |
| Raw format compliance | Nonempty generated text, trimmed only at its edges, is accepted by the unchanged restricted proof-body parser. Prose, Markdown fences, declarations, and unsupported syntax fail. This does not imply mathematical correctness. |
| Rocq verification | PASS from the existing verifier, including kernel and assumption checks. Output extraction is identical to v1: whitespace and at most one complete outer code fence. No semantic edits or repair occur. A fenced correct proof can therefore verify but fail raw-format compliance. |
| Argument fidelity | The existing deterministic structure proxy compares induction/binder position, premise use, congruence/symmetry, rewrite references/directions, and branch closing steps. `verified_structure_matches` requires both verification and full proxy agreement. It does not certify English semantics; valid alternative arguments can mismatch. |

All raw completions, extracted bodies, format decisions, verifier results,
failure categories, proxy details, prompts, token IDs/counts, model/settings,
and generation latencies are retained. Any assistant semantic reviews are
attached separately to hash-bound candidates and do not affect selection.
They are not independent human review. The theorem-only output is compared
against the same reference argument, which was hidden from the model.

## Commands and artifacts

The original sequence, before the prompt freeze, is:

```sh
python3 -m unittest discover -s tests -p test_prompt_v2.py -v
.venv/bin/python -m prompt_v2.experiment develop
python3 -m prompt_v2.report development --reverify
python3 -m prompt_v2.experiment freeze
.venv/bin/python -m prompt_v2.experiment evaluate
python3 -m prompt_v2.report evaluation --reverify
python3 -m unittest discover -s tests -v
```

The development and evaluation runners refuse existing output directories and
never resume or retry an attempt. Development is disabled once Prompt v2 is
frozen. The current result files can be checked without model generation:

```sh
python3 -m prompt_v2.experiment check
python3 -m prompt_v2.report development --reverify
python3 -m prompt_v2.report evaluation --reverify
```

The model and dependency setup is unchanged from [v1](../baseline/README.md):

```sh
python3 scripts/setup_rocq.py --check
python3 -m venv .venv
.venv/bin/python -m pip install --cache-dir .cache/pip -r baseline/requirements.lock
.venv/bin/python -m baseline.setup_model
.venv/bin/python -m baseline.setup_model --check
```

To explicitly reproduce the frozen selected prompt without overwriting the
recorded run, choose a new output directory:

```sh
.venv/bin/python - <<'PY'
from baseline.dataset import ROOT
from prompt_v2.experiment import run_phase
run_phase('evaluation', ROOT / '.cache/prompt-v2-reproduction-001')
PY
python3 - <<'PY'
from baseline.dataset import ROOT
from prompt_v2.experiment import load_phase
from prompt_v2.report import reverify
manifest, rows = load_phase(ROOT / '.cache/prompt-v2-reproduction-001')
reverify(manifest, rows)
PY
```

| Location | Contents |
| --- | --- |
| `results/prompt-v2-development/` | All 180 planned development attempts, full-set and common-subset metrics, ranking, and report |
| `prompt_v2/selected.json` | Frozen concrete instructions and fixed demonstration messages |
| `prompt_v2/frozen.json` | Selection evidence, ranking, freeze time, and immutable input checksums |
| `results/baseline-v2/` | Separate 24-attempt diagnostic run, machine-readable outputs/summary, and v1/v2 comparison report |

Within each run, `manifest.json`, `results.jsonl`, `attempts.jsonl`,
`summary.json`, and `completion.json` are immutable. Reporting writes a separate
`REPORT.md`; evaluation additionally writes `report_summary.json` and
`assessed_results.jsonl`, attaching any `argument_reviews.json` entries.

The evaluation uses the same 12 examples only after selection is frozen. It
remains a reused diagnostic with arithmetic-family overlap with development
and the few-shot examples. It is not a family-disjoint test set. The paired
comparison changes only the final target argument, but a matching proof does
not establish causal use of each English step. Reproducibility is strongest
on the pinned runtime/platform; floating-point output and latency can differ
across hardware.
