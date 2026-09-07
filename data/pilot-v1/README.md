# Pilot data v1

Start with [PILOT_DATA_SPEC.md](PILOT_DATA_SPEC.md). Public reviewed examples are `train.jsonl` and `dev.jsonl`. Read `coverage.json`, `family_audit.json`, `historical_reference_review.json` and `verification_report.json` for construction evidence. `split_manifest.json` freezes all three partitions before training. `../../pilot_v1/release.json` binds the final data, documentation, tooling and validation evidence to their hashes. Reserved contents must stay unopened until after training and checkpoint selection.

From the repository root, using the existing pinned environment:

```sh
python3 scripts/setup_rocq.py --check
python3 -m pilot_v1.dataset check
python3 -m pilot_v1.dataset reverify
python3 -m unittest discover -s tests -p 'test_pilot_v1.py' -v
python3 -m unittest discover -s tests -v
```

`check` validates frozen hashes, schema/alignment and public split membership; it does not parse the holdout. `reverify` recompiles all 30 public train/dev references and compares their results with the stored evidence. All six holdout references were compiled during construction; their original evidence stays frozen and checksummed. Full historical regression tests also recheck old diagnostic artifacts; those are not pilot selection or holdout evaluation.

Optional input/target exports, without training or prompt changes:

```sh
python3 -m pilot_v1.dataset export --split train --output .cache/pilot-v1-train-export.jsonl
python3 -m pilot_v1.dataset export --split dev --output .cache/pilot-v1-dev-export.jsonl
```

Exports use exclusive creation and refuse to overwrite existing files. Choose a new output filename to repeat an export. Inputs contain only the fixed theorem and informal proof; targets contain only proof-body code. Holdout is not an accepted export split.

The one-time authoring sequence used before freezing was:

```sh
.venv/bin/python -m pilot_v1.dataset prepare
python3 -m pilot_v1.dataset freeze
```

`prepare` validates the curated JSONL sources, verifies all 36 references and the 12 historical A/B references, runs useful-reference ablations, checks exact target token paths using only the pinned local tokenizer, and seals the holdout. `freeze` uses sealed membership metadata without reopening holdout contents. Both refuse to replace an existing release. Reproducing checks is intentionally distinct from authoring a new dataset version; do not delete manifests to rerun holdout construction. The initial authoring required `.venv` with the existing locked baseline dependencies; public checks and Rocq replay use the Python standard library.
