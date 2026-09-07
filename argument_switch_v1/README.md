# Argument switching on six development theorems

This probe follows the frozen `proofbridge-constrained-v1` checkpoint. It calls
the unchanged `constrained_v1.experiment.generate` and uses the exact frozen
Prompt v2 prefix and target formatter. The theorem-only target omits the
informal paragraph; A and B substitute only that paragraph. Strategy labels,
reference code, and scoring requirements are never included in the prompt.

The selected existing theorems are 003–007 and 011. All are in the common
25-seed development subset, excluding demonstrations and their equivalents.
003–006 compare definitional computation with induction on n; 007 compares
direct use of a premise with induction on n; 011 compares substitution under
addition with induction on p, the third binder. No theorem is added or changed.

`variants.json` stores both informal arguments and reference bodies separately
from the frozen dataset. A reuses the original curated argument. B is newly
curated, and each reference is saved as a standalone `.v` file in `references/`.
`preflight.json` records all 12 actual Rocq results, exact tokenizer-path and EOS
acceptance, and nine passing protocol tests before inference. Some initial B
references required correction during reference preparation; no model outputs
had been generated, and the final references were fixed before inference.

The induction arguments are intentionally unnecessary for some of these simple
goals; several IH uses are redundant. This is a controlled strategy-compliance
test, not a claim about proof efficiency or mathematical generalization.

## Protocol

There are exactly 18 generations, ordered by seed and then theorem-only, A, B.
Each uses one attempt, no repair, the same greedy configuration and 256-token
cap, the same frozen model/tokenizer, and the same verifier. Model errors and
length-limited outputs remain attempts. Existing directories cannot be resumed
or overwritten. No diagnostic evaluation examples enter selection or scoring.

Strategy scoring is separate from verification. The predeclared rubric checks
no induction for A, actual premise use where required, and the correct
induction binder plus specified branch-local premise/IH uses for B. Equivalent
Rocq tactics count equally. A syntactically valid but mathematically wrong proof
may match an attempted strategy, but cannot earn verified strategy credit.
Induction presence alone is insufficient when B requires an IH or premise.

Primary switching accuracy has denominator **all six theorems**. A theorem
counts only if both A and B outputs verify and match their respective
requirements. Verified proofs that ignore B, and unverified proofs with the
right outline, are switching failures. Theorem-only has no requested strategy;
its observed strategy is compared against both references as a control.

## Commands

With the existing pinned environment, the original run sequence was:

```sh
python3 -m checkpoints.freeze_constrained_v1 check
.venv/bin/python -m argument_switch_v1.experiment validate
.venv/bin/python -m argument_switch_v1.experiment run
.venv/bin/python -m argument_switch_v1.experiment check --replay-tokens --reverify
python3 -m argument_switch_v1.report
python3 -m unittest discover -s tests -v
```

Validation refuses an existing preflight or run. The `check` command performs
no model generation; it verifies input hashes, exact prompts, all 18 attempts,
token budgets, scores, and optionally every allowed token and Rocq result.
The original model/dependency setup remains in `constrained_v1/README.md`.

An explicitly requested future reproduction can use a new directory:

```python
from baseline.dataset import ROOT
from argument_switch_v1.experiment import run
run(ROOT / '.cache/argument-switch-v1-reproduction-001')
```

Artifacts are separate in `results/argument-switch-v1/`: immutable raw outputs,
attempt journal, manifest and summary; manual assistant review; combined
machine-readable records; and `ARGUMENT_SWITCH_REPORT.md`. The assistant review
is not independent human review and does not silently replace the predeclared
strategy criteria. These are previously used development theorems with known
proof patterns, not an unseen test set.
