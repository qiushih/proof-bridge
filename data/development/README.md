# Frozen development examples

`pairs.jsonl` freezes the 30 Rocq-verified `seed-0.2` records for baseline v1.
The only record change from `data/seeds/pairs.jsonl` is `split: "development"`.
Original statements, proof paragraphs, aligned steps, canonical bodies,
families, provenance, and verification evidence are preserved.

`manifest.json` records the freeze time, 30 IDs, and checksums of the snapshot,
original seed artifacts, verifier, and Rocq lock. The baseline refuses changed
inputs. The original seed files retain their historical `unassigned` label;
this snapshot assigns their role in the baseline. No development example
contributes to evaluation metrics or appears in the zero-shot model prompts.

Check the freeze and the evaluation references from the repository root:

```sh
python3 -m baseline.dataset --check
```

See [baseline/README.md](../../baseline/README.md) for the full experiment.
