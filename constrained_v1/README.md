# Constrained decoding with frozen Prompt v2

This is a separate development experiment. It retains the frozen 3-shot
Prompt v2, the pinned Qwen2.5-Coder-0.5B-Instruct weights/tokenizer, and the
original verifier, data, and greedy generation configuration. Only a logits
mask is added. No diagnostic evaluation examples enter this pipeline.

The decoder incrementally parses the generated ASCII code. A vocabulary trie
allows a token only if its entire decoded text leaves a valid grammar prefix,
including tokens that cross command boundaries. Non-ASCII tokens and special
tokens other than EOS are excluded. EOS is permitted only after a complete
grammatical body. The model's highest-scoring allowed token is selected.

The grammar enforces:

- All theorem binders are explicitly introduced before proof tactics. Multiple
  `intros` commands and arbitrary fresh ASCII names up to 32 characters work.
- Local names are unique, nonreserved, and scoped. A theorem binder's position
  determines whether it is a natural-number variable or an equality premise.
- Induction targets are local naturals. At most one induction and exactly two
  nonempty branches are permitted. The successor variable and IH exist only in
  the successor branch; the old induction target is removed from both branches.
- `rewrite`, `exact`, and `apply` refer only to local equality premises or IH,
  never natural-number variables or global helpers. IH may itself have dependent
  premises; the mask does not check its complete Rocq type or applicability.
- Only the existing tactic syntax is allowed. Commands end with periods; a
  separating whitespace prevents qualified-name ambiguities between commands.
  The original 40-command limit is respected.

The mask receives only the theorem's binder counts and the generated prefix.
It never receives a reference proof, informal argument, seed ID, theorem family,
Rocq goal, or compiler feedback. The original informal text still reaches the
model through its unchanged prompt. The model chooses tactics, induction
variable, rewrite direction, and when to end. No proof template is forced.

This is a stricter subset of the verifier's accepted syntax, including complete
introductions, bounded local names, proof-versus-nat reference kinds, whitespace,
and nonempty branches. All 30 exact reference bodies and their token paths pass
preflight, but this does not establish completeness for every possible valid
proof. The 256-token cap can still truncate a valid prefix; there is no automatic
completion, retry, fallback, or repair. Grammar validity does not imply that a
proof is mathematically correct or follows the informal argument.

## Reproduction

Use the unchanged environment described in `baseline/README.md`. The original
experiment sequence is:

```sh
python3 -m unittest discover -s tests -p test_constrained_v1.py -v
.venv/bin/python -m constrained_v1.experiment validate
.venv/bin/python -m constrained_v1.experiment run
.venv/bin/python -m constrained_v1.experiment check --replay-tokens --reverify
python3 -m constrained_v1.report
python3 -m unittest discover -s tests -v
```

`validate` loads the pinned tokenizer but does not run the model. It runs the
grammar tests, checks every reference token and EOS against the actual mask,
and compiles all 30 unchanged reference proofs. `preflight.json` captures this
evidence and source hashes before inference. The runner refuses stale preflight
evidence, an existing output directory, or changed model/settings/prompts.

The default run contains exactly 60 attempts, in fixed seed order with
theorem-plus-informal followed by theorem-only. Per-attempt records contain
the exact prompt, input/output token IDs, proof text, verification result,
format/structure scores, generation settings, latency, and a mask trace.
`attempts.jsonl` records one start and finish per attempt. Checksums bind the
completed raw records. Reporting cannot replace those records.

Check an existing run with the `check` command above; it does not generate
samples. An explicit future reproduction can use a new directory without
overwriting this experiment:

```python
from baseline.dataset import ROOT
from constrained_v1.experiment import run
run(ROOT / '.cache/constrained-v1-reproduction-001')
```

## Comparison and fidelity

Results are saved in `results/constrained-v1/`, separate from all prior runs.
The comparison uses the saved `short_3shot` development outputs and the
hash-bound manual reviews from `results/prompt-v2-development-audit/`.

Report both input conditions on all 30 seeds and on the frozen common 25-seed
subset, excluding 001, 002, 010, 016, and 027. Format, invalid-reference and
missing-introduction failures, Rocq verification, verified faithful proofs,
and latency are distinct measurements. The existing structure proxy remains
available; assistant fidelity reviews accept equivalent mathematical reasoning
and flag genuine strategy differences. They are not independent human review.

The preregistered success condition is more verified faithful proofs on the
common subset, pooled over both conditions; each condition is also reported.
Fewer parser failures alone is not success. These are reused development data,
not a held-out generalization result.

Generation latency includes constraint computation. Vocabulary construction is
reported separately. The grammar cache starts fresh for inference and is shared
across attempts, so cache warm-up and different output lengths affect timing.
No reference paths prewarm the inference cache. Stored unconstrained timings
were measured in the earlier run, not contemporaneously on identical load.
