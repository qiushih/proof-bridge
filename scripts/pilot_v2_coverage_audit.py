"""Audit only frozen pilot-v1 train/dev references; no inference or holdout access.

Run: .venv/bin/python -m scripts.pilot_v2_coverage_audit [--reverify]
The default checks committed evidence; --reverify also compiles the 30 originals.
Neither mode authors examples, mutates references, or loads model weights.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from pilot_v1.protocol import audit_rows, commands, equivalence_key, load_split
from verifier import verify

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/pilot-v2-coverage-audit"
SPLITS = ("train", "dev")
CELLS = (
    "computational_base__ih_rewrite",
    "computational_base__ih_congruence",
    "premise_base__ih_rewrite",
    "premise_base__ih_congruence",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cell(row):
    """Check the concrete branch commands, separately from stored feature flags."""
    cmds = commands(row["proof_body"])
    if not any(c[0] == "induction" for _, c in cmds):
        return "non_inductive"
    branches = {b: [c for branch, c in cmds if branch == b] for b in (1, 2)}
    if branches[1] == [["simpl"], ["reflexivity"]]:
        base = "computational_base"
    elif branches[1] == [["simpl"], ["exact", "H"]]:
        base = "premise_base"
    else:
        raise ValueError(f"Needs manual classification: {row['id']} base")
    if branches[2] in (
        [["simpl"], ["rewrite", "IH"], ["reflexivity"]],
        [["simpl"], ["rewrite", "<-", "IH"], ["reflexivity"]],
    ):
        successor = "ih_rewrite"
    elif branches[2] == [["simpl"], ["f_equal"], ["exact", "IH"]]:
        successor = "ih_congruence"
    else:
        raise ValueError(f"Needs manual classification: {row['id']} successor")
    assert row["argument_contract"]["base"] == (
        "premise" if base == "premise_base" else "reflexivity")
    assert row["argument_contract"]["successor_ih_use"] == (
        "rewrite" if successor == "ih_rewrite" else "congruence")
    return base + "__" + successor


def summary(rows, denominator):
    return {
        "rows": len(rows),
        "percent_of_split": round(100 * len(rows) / denominator, 2),
        "row_ids": [r["id"] for r in rows],
        "theorem_count": len({r["theorem_id"] for r in rows}),
        "theorem_ids": sorted({r["theorem_id"] for r in rows}),
        "equivalence_class_count": len({equivalence_key(r["formal_statement"]) for r in rows}),
        "generation_family_count": len({r["generation_family"] for r in rows}),
        "generation_families": sorted({r["generation_family"] for r in rows}),
    }


def review(row, category):
    """Persist this audit's assistant review; these are not automated English scores."""
    induction = category != "non_inductive"
    issues = []
    if induction:
        issues.append("missing_sentence_boundary_before_This_in_base_case")
    if row["informal_proof"].startswith("Fix n as natural numbers."):
        issues.extend(["singular_n_described_as_plural", "other_introduced_data_boilerplate_without_other_data"])
    if row["id"] in {"pd01_B", "pd03_B"}:
        issues.append("exactly_the_IH_means_up_to_definitional_conversion")
    if row["theorem_id"] == "pd03":
        issues.append("IH_stated_in_definitionally_reduced_form_without_saying_so")
    notes = ["The informal steps match the authored tactic order and intended equality transformations."]
    if induction:
        notes.append("Base action and successor IH method are explicitly requested. Rewrite and congruence preserve the same mathematical induction; assess their requested steps separately.")
    if "premise_base" in category:
        notes.append("The base goal is non-reflexive at symbolic m/p; exact H is explicitly justified. Premise and IH have distinct branch roles.")
    if row["theorem_id"] in {"pt11", "pt12"}:
        notes.append("The premise is useful in this script but is itself a right-zero identity; this does not teach arbitrary premise-dependent equalities.")
    if row["theorem_id"] in {"pt13", "pt14"}:
        notes.append("The premise relates m + 0 to p. Equality reversal produces its orientation sibling; these are not independent family evidence.")
    return {
        "id": row["id"], "split": row["split"], "cell": category,
        "theorem_id": row["theorem_id"], "generation_family": row["generation_family"],
        "proof_body_sha256": hashlib.sha256(row["proof_body"].encode()).hexdigest(),
        "informal_proof_sha256": hashlib.sha256(row["informal_proof"].encode()).hexdigest(),
        "reviewer": "assistant", "human_reviewed": False,
        "mathematical_alignment": "aligned",
        "requested_steps_explicit": True,
        "clarity_issues": issues, "review_notes": notes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reverify", action="store_true")
    args = parser.parse_args()
    inputs = [ROOT / f"data/pilot-v1/{s}.jsonl" for s in SPLITS]
    inputs += [ROOT / p for p in (
        "data/pilot-v1/split_manifest.json", "pilot_v1/protocol.py",
        "scripts/seed_schema.py", "verifier.py", "environment.lock.json",
        "constrained_v1/grammar.py", "scripts/pilot_v2_coverage_audit.py")]
    before = {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    rows = [r for split in SPLITS for r in load_split(split)]
    assert Counter(r["split"] for r in rows) == {"train": 24, "dev": 6}
    audit = audit_rows(rows)
    classified = {r["id"]: cell(r) for r in rows}
    reviews = [review(r, classified[r["id"]]) for r in rows]
    counts = {}
    for split in SPLITS:
        group = [r for r in rows if r["split"] == split]
        counts[split] = summary(group, len(group))
        counts[split]["cells"] = {c: summary(
            [r for r in group if classified[r["id"]] == c], len(group))
            for c in (*CELLS, "non_inductive")}
        counts[split]["ih_rewrite_directions_by_base"] = {
            base: dict(Counter(r["argument_contract"]["ih_direction"] for r in group
                              if classified[r["id"]] == base + "__ih_rewrite"))
            for base in ("computational_base", "premise_base")}
    groups = defaultdict(list)
    for r in rows:
        groups[equivalence_key(r["formal_statement"])] .append(r["id"])
    equivalences = [ids for ids in groups.values() if len(ids) > 1]
    saved = [{"id": r["id"], "verification": r["verification"],
              "reference_removal": r["quality_checks"]["reference_removal"]} for r in rows]
    result = {
        "schema_version": "pilot-v2-coverage-audit-1.0",
        "scope": "Existing pilot-v1 train/dev only; no new data or model outputs.",
        "inputs_sha256": before,
        "counts": counts, "schema_and_family_audit": audit,
        "equivalence_groups": equivalences,
        "counting_notes": [
            "Percentages use all rows in each split, including non-inductive rows.",
            "Family counts follow the existing conservative generation_family ledger; they are not statistically independent samples.",
            "Equivalence counts use the existing definitional/binder-permutation/equality-orientation check, not complete mathematical equivalence.",
            "Cell counts are derived from proof commands and cross-checked against contracts; schema validation recomputes all argument_features.",
        ],
        "alignment_review": reviews,
        "clarity_issue_counts": dict(Counter(i for r in reviews for i in r["clarity_issues"])),
        "saved_reference_evidence": saved,
        "recommendation": {
            "priority": "Add dev coverage for premise-dependent bases before increasing training volume.",
            "minimum_dev_addition": {
                "rows": 2, "theorems": 1, "arguments_per_theorem": 2,
                "required_cells": list(CELLS[2:]),
                "condition": "A naturally useful premise-plus-induction theorem in the existing dev family, after conservative family/duplicate review; no train-family or consumed-holdout-family relatives.",
                "status": "Proposed lower bound, not authored, verified, or guaranteed constructible under the current family ledger.",
            },
            "immediate_training_additions": 0,
            "conditional_training_followup": "Only after dev coverage is established: one natural A/B theorem pair in a training family if a concrete structural gap is demonstrated. Repetition or reversed-equality copies do not add independent coverage.",
            "preservation": "Keep pilot-v1 files, prompt, decoder, verifier and selection rule unchanged. Any new split membership and evaluation protocol must be a separately frozen v2 release.",
            "evaluation": "Use a matched original-data control; report verification, verified mathematical fidelity, requested steps, and both-argument theorem success separately. A fresh holdout is required before a new training experiment; no design or access here.",
        },
        "limitations": [
            "Coverage and assistant review do not establish the cause of model failures or predict an intervention effect.",
            "No independent human language review; minor clarity issues do not imply mathematical unfaithfulness.",
            "Saved ablations establish usefulness in these particular scripts, not globally minimal proofs or logically necessary premises.",
            "A two-row dev addition only fills the empty cells; it is not a robust generalization benchmark.",
        ],
        "activity": {"model_generations": 0, "training_updates": 0,
                     "examples_added": 0, "frozen_files_modified": 0,
                     "holdout_proof_files_opened": 0, "diagnostic_examples_opened": 0},
    }
    if args.reverify:
        fresh = []
        for row in rows:
            evidence = asdict(verify(row["formal_statement"], row["proof_body"]))
            assert evidence["status"] == "PASS", (row["id"], evidence)
            assert evidence["source_sha256"] == row["verification"]["source_sha256"]
            fresh.append({"id": row["id"], "verification": evidence})
            print(f"{row['id']}: PASS", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "rocq_reverification.json").write_text(json.dumps({
            "scope": "Original train/dev references only; no proof mutations.",
            "inputs_sha256": before, "verified": len(fresh), "records": fresh,
        }, indent=2) + "\n")
    assert before == {str(p.relative_to(ROOT)): digest(p) for p in inputs}
    OUT.mkdir(parents=True, exist_ok=True)
    output = OUT / "coverage_audit.json"
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if output.exists():
        assert output.read_text() == encoded, "Audit differs; review before replacing evidence."
    else:
        output.write_text(encoded)
    print(json.dumps({s: {c: v["rows"] for c, v in counts[s]["cells"].items()}
                      for s in SPLITS}, indent=2))
    print("PASS: 30 rows; source hashes unchanged; no model/holdout access.")


if __name__ == "__main__":
    main()
