# Prompt v2 development comparison

All prompt development used the frozen 30 development seeds. The three candidate prompts and selection rule were declared before generation. No evaluation record was loaded by development or prompt selection.

Each candidate received one attempt on each seed in both conditions: 180 generations total, with no repair or fine-tuning. The same Qwen model revision, greedy decoding, 256-new-token cap, CPU/float32 settings, and verifier were retained.

| Candidate | Target input | Raw format | Rocq verified | Verified structure match |
| --- | --- | --- | --- | --- |
| short_2shot | theorem_and_informal | 11/30 | 9/30 | 7/30 |
| short_2shot | theorem_only | 22/30 | 8/30 | 6/30 |
| short_3shot | theorem_and_informal | 17/30 | 16/30 | 11/30 |
| short_3shot | theorem_only | 20/30 | 12/30 | 7/30 |
| short_4shot | theorem_and_informal | 14/30 | 10/30 | 9/30 |
| short_4shot | theorem_only | 17/30 | 6/30 | 6/30 |

The shared selection subset excludes seed_001, seed_002, seed_010, seed_016, seed_027: the union of demonstrations and their definitional duplicates. This leaves 25 seeds, scored under both conditions (50 outputs per candidate). Full-set metrics above include demonstration targets and are not held-out accuracy.

| Selection rank | Candidate | Raw format | Rocq verified | Verified structure match |
| --- | --- | --- | --- | --- |
| 1 | short_3shot | 28/50 | 21/50 | 11/50 |
| 2 | short_2shot | 25/50 | 12/50 | 8/50 |
| 3 | short_4shot | 22/50 | 9/50 | 8/50 |

Selected by the predeclared lexicographic rule: **short_3shot**. Raw format takes precedence, then verification, then verified structure agreement, then fewer demonstrations and candidate order.

Raw format means the nonempty output passes the unchanged restricted parser after trimming edge whitespace only. Markdown fences fail that metric. Verification retains v1's narrowly defined outer-fence extraction, so a fenced valid proof can verify while failing format compliance. All raw and extracted texts are preserved.

Argument fidelity here is a deterministic structure proxy, recorded separately from correctness. It compares induction/binder position, actual premise use, congruence/symmetry, rewrite references/directions, and branch closing steps. Verified structure match requires both kernel verification and a full proxy match; it is not an independent certification of English semantics.

Fixed demonstrations are identical in both conditions and include their informal arguments. Only the final target's informal proof is ablated. The demonstrations are copied exactly from development and are not retrieved or adapted per target.

Files: `manifest.json` preserves prompts, plan, source hashes, and settings; `results.jsonl` preserves all 180 outputs and scores; `attempts.jsonl` and `completion.json` audit the one-attempt budget; `summary.json` preserves the selection metrics. The selected prompt is separately frozen in `prompt_v2/selected.json` and `prompt_v2/frozen.json` before diagnostic evaluation.
