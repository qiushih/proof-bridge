# Twelve baseline evaluation proofs

`curated.json` contains twelve new, assistant-curated informal/formal proofs in
the existing nat/equality/successor/addition fragment. Each `steps` array is the
authored English/code alignment; the loader deterministically joins its text
into `informal_proof` and its code into the reference `proof_body`. Features and
fidelity signatures are derived from the reference code. References are never
included in model prompts.

| IDs | Arguments |
| --- | --- |
| eval_001–003 | Definitional successor/addition reductions |
| eval_004–007 | Symmetry, congruence, and equality-premise rewriting |
| eval_008–012 | One induction, including premise use and induction on the second binder |

All twelve reference proofs pass the unchanged Rocq verifier. Actual compiler
results, input hashes, assistant provenance, and the overlap audit are in
`verification_report.json`. `human_reviewed` remains false. No multiplication,
lists, new tactics, or global helper lemmas were introduced.

The statements are distinct from one another and from all 30 development
statements under binder renaming and definitional normalization. Six generation
families overlap with development, so these are diagnostic evaluation examples,
not an independent family holdout. Related variants retain the existing family
labels. Eval 010 intentionally supplies an induction argument for a theorem
that also has a shorter direct-rewrite proof, separating argument fidelity
from formal correctness.

The reference source, verification report, and development manifest are hashed
in the run manifest before generation. Model outputs did not guide edits to
the evaluation cases or references.

```sh
python3 -m baseline.dataset --check
```

See [baseline/README.md](../../baseline/README.md) for prompts, execution, results,
and the fidelity rubric.
