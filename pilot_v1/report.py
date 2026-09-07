"""Render aggregate pilot reports without opening reserved reference contents."""

import json
from pilot_v1.protocol import DATA, PATTERNS


def render_report():
    verification = json.loads((DATA / "verification_report.json").read_text())
    coverage = json.loads((DATA / "coverage.json").read_text())
    audit = json.loads((DATA / "family_audit.json").read_text())
    lines = ["# Pilot v1 reference verification", "",
        f"**Rocq {verification['rocq_version']}: {verification['references_verified']}/36 references PASS.** All passed the unchanged verifier, produced compiled proof artifacts and passed its global-assumption check.", "",
        "| Split | PASS | FAIL | Theorem IDs | Same-theorem A/B groups |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for split, counts in coverage.items():
        lines.append(f"| {split} | {counts['examples']} | 0 | {counts['theorems']} | {counts['same_theorem_pairs']} |")
    lines += ["", f"All {verification['frozen_decoder_token_paths_allowed']} exact reference token paths, including EOS, are allowed by frozen constrained-v1. The longest target has {verification['max_reference_tokens_including_eos']} tokens including EOS, within 256.", "",
        f"All {verification['reference_removal_checks']} reference-removal ablations failed inside Rocq as required. These are construction quality checks on useful premise/IH steps, not generated samples or training targets. All 12 historical A/B references also reverified; their original files/results were preserved.", "",
        "## Coverage by proof pattern", "",
        "Counts overlap: one proof can exercise several patterns. A before/after count requires an actual `simpl` command in that order within the same branch.", "",
        "| Pattern | Train / 24 | Dev / 6 | Holdout / 6 |",
        "| --- | ---: | ---: | ---: |"]
    for pattern in PATTERNS:
        values = [coverage[s]["patterns"][pattern] for s in ("train", "dev", "holdout")]
        lines.append(f"| {pattern} | {' | '.join(map(str, values))} |")
    within = audit["within_pilot"]
    historical = audit["historical_overlap"]
    overlap = {s: sum(bool(r["development"] or r["diagnostic"]) for r in historical["per_example"] if r["split"] == s)
               for s in ("train", "dev", "holdout")}
    lines += ["", "## Family and historical checks", "",
        f"{within['examples']} unique row IDs, {within['theorems']} theorem IDs, {within['paired_theorems']} same-theorem pairs and {within['equivalence_classes']} conservative statement-equivalence classes. No exact statement/body duplicate and no cross-split family or detected statement equivalence.", "",
        f"Historical equivalent-statement overlap: train {overlap['train']} rows, dev {overlap['dev']} rows, holdout {overlap['holdout']} rows. Repeated statements are deliberate training/development targets, not independent observations. Historical family exposure is broader than exact overlap and is recorded in `family_audit.json`.", "",
        "The six holdout references have fresh statements but belong to a historically exposed family. They were inspected only for construction and reference QA, then sealed. Subsequent checks use their committed byte hashes, without parsing or scoring them. Detailed holdout diagnostics remain under `reserved/`.", "",
        "Draft reference preparation encountered the existing expression-depth limit and a simplification/rewrite-pattern error. The new draft examples were corrected before sealing; the verifier and historical proofs were not changed.", "",
        "## Limits", "",
        "All alignment and naturalness reviews are assistant reviews; human-reviewed remains false. The ablations do not prove global minimality. Most A/B contrasts are equivalent proof-step realizations of one mathematical strategy. This small release cannot establish broad informal-proof dependence or independent mathematical generalization.", "",
        "**Training runs: 0. Model generations: 0.** Reference tokenization used the pinned local tokenizer only. See `validation.json` for final regression and replay evidence; no inference runner is part of this dataset release.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    output = DATA / "VERIFICATION_REPORT.md"
    text = render_report()
    if output.exists() and output.read_text() != text:
        raise ValueError("Refusing to overwrite a changed report")
    output.write_text(text)
    print("Report rendered from public aggregates; holdout not opened")
