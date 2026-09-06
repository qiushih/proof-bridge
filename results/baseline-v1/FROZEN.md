# Frozen baseline v1

Release checkpoint: annotated Git tag **`proofbridge-baseline-v1`**.

This release preserves the original experiment: 30 development seeds, 12
evaluation reference proofs, the pinned model/tokenizer and runtime settings,
both prompt conditions, all 24 single-attempt outputs, verifier evidence,
argument-fidelity reviews, summary, and report. The model weights remain an
external download pinned by revision and SHA-256 in `baseline/model.lock.json`.

The recorded outcome remains 0/12 verified in each condition. All 24 generated
candidates were rejected before Rocq compilation, and their formal argument
fidelity is unassessable. The evaluation shares arithmetic families with
development. These limitations are part of the frozen result.

`freeze.json` records release-wide file checksums and the validation performed
at freeze time. It excludes itself to avoid a circular checksum. The Git commit
and annotated tag capture that manifest too. Freezing generated no new model
outputs and did not alter any existing experiment inputs or results.

From the repository root, check the release files without running the model:

```sh
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

manifest = json.loads(Path('results/baseline-v1/freeze.json').read_text())
for name, expected in manifest['files_sha256'].items():
    actual = hashlib.sha256(Path(name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Frozen file changed: {name}')
print(f"PASS: {len(manifest['files_sha256'])} frozen files match")
PY
python3 -m unittest discover -s tests -v
```

Preserve this tag and the v1 artifacts when starting subsequent experiments.
Use separately versioned prompts, configuration, and results. To reproduce v1
after later code changes, start from this tag in a separate checkout and follow
[the reproduction commands](../../baseline/README.md); write reproduction
outputs to a new directory. Reproductions do not replace the recorded first run.
