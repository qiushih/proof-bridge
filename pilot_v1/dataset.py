"""Prepare once, freeze splits, and check/reverify public pilot data. No model calls."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json

from pilot_v1.protocol import (ROOT, DATA, MANIFEST, PATHS, FAMILIES, audit_rows, coverage, digest,
    equivalence_key, load_split, read_rows, training_records, write_json, write_rows)
from verifier import render_source, sha256_file, verify

PREPARATION = DATA / "preparation.json"
PROTECTED = ROOT / "pilot_v1/protected_baselines.json"


def check_protected():
    original = json.loads(PROTECTED.read_text())
    for name, expected in original["files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Historical baseline changed: {name}")
    return len(original["files_sha256"])


def token_evidence(rows):
    # Load ONLY the pinned local tokenizer, never weights or an inference engine.
    from constrained_v1.experiment import tokenizer_only
    from constrained_v1.grammar import initial, feed, can_end
    from constrained_v1.tokens import Vocabulary
    tokenizer = tokenizer_only()
    vocabulary = Vocabulary(tokenizer)
    result = {}
    for row in rows:
        body = row["proof_body"]
        ids = tokenizer.encode(body, add_special_tokens=False)
        state, pieces = initial(row["formal_statement"]), []
        for token in ids:
            if token not in vocabulary.allowed(state):
                raise ValueError(f"Frozen decoder blocks {row['id']}")
            pieces.append(vocabulary.pieces[token])
            state = feed(state, pieces[-1])
        if "".join(pieces) != body or not can_end(state) or vocabulary.eos_id not in vocabulary.allowed(state) or len(ids) + 1 > 256:
            raise ValueError("Invalid token path or target exceeds the 256-token budget including EOS")
        result[row["id"]] = {"all_tokens_and_eos_allowed": True, "body_tokens": len(ids), "with_eos": len(ids) + 1}
    return result


def reference_review():
    reviews = []
    notes = {
        "seed_003": "B inducts on a definitional identity. Its IH rewrites before computation, but reflexivity alone also proves the successor equality. Use the direct A argument instead.",
        "seed_004": "B uses reverse IH substitution inside a goal already true by computation. Induction and this substitution are unnecessary; this is a poor target for learning rewrite direction.",
        "seed_005": "B first simplifies to identical sides, then rewrites both backwards with the IH. This deliberately reintroduces reducible syntax; exclude it from primary training targets.",
        "seed_006": "B reduces to identical sides and then restores a redundant zero expression using the IH. The induction and IH step add no mathematical work.",
        "seed_007": "B repeats the exact premise in both induction branches and never uses the IH. Direct premise reuse is the natural target.",
        "seed_011": "B is a coherent structural derivation with a useful base premise and successor IH. Induction is still avoidable via A's direct substitution. Keep B as secondary pedagogical evidence; use nontrivial right-zero residuals for the primary premise-plus-induction targets.",
    }
    for variant in json.loads((ROOT / "argument_switch_v1/variants.json").read_text()):
        for label, ref in variant["arguments"].items():
            result = asdict(verify(variant["formal_statement"], ref["proof_body"]))
            if result["status"] != "PASS":
                raise ValueError("Historical reference no longer verifies")
            b = label == "B"
            reviews.append({"example_id": variant["example_id"], "argument": label,
                "historical_file": ref["reference_file"], "body_sha256": digest(ref["proof_body"]),
                "reviewer": "assistant", "human_reviewed": False,
                "disposition": ("secondary_only" if variant["example_id"] == "seed_011" else "exclude_primary") if b else "eligible_direct_reference",
                "induction_avoidable_by_direct_proof": b,
                "redundant_ih_use": b and variant["example_id"] in {"seed_003", "seed_004", "seed_005", "seed_006"},
                "unused_ih": b and variant["example_id"] == "seed_007",
                "unnatural_structure": b and variant["example_id"] != "seed_011",
                "copied_into_pilot": False,
                "note": notes[variant["example_id"]] if b else "Direct computation or premise transport is clear and aligned. Preserve this valid historical reference.",
                "verification": result})
    return {"status": "REVIEWED", "historical_files_modified": 0, "references": reviews,
            "counts": {"references": 12, "eligible_direct": 6, "excluded_weak_B": 5, "secondary_only_B": 1},
            "limitation": "Assistant review is not independent human review. Formal validity and training-target quality are different judgments."}


def ablations(row):
    """Quality control only: remove useful references, never model-generated data."""
    tests = []
    for name, enabled, needle in (
        ("remove_ih_use", row["argument_features"]["uses_induction"], "IH."),
        ("remove_premise_use", row["argument_features"]["uses_premise"], "H."),
    ):
        if not enabled:
            continue
        lines = row["proof_body"].splitlines()
        def selected(line):
            # Match reference commands only; never the induction or intros command.
            text = line.strip().removeprefix("- ")
            return text.startswith(("exact ", "rewrite ", "apply ")) and text.split()[-1] == needle
        body = "\n".join(line for line in lines if not selected(line))
        if body == row["proof_body"]:
            raise ValueError("Requested reference-removal ablation found no reference")
        result = asdict(verify(row["formal_statement"], body))
        if result["status"] != "FAIL" or result["stage"] != "rocq" or result["category"] not in {"PROOF_ERROR", "INCOMPLETE_PROOF"}:
            raise ValueError(f"Redundant reference or invalid quality-control check: {row['id']} {name}")
        tests.append({"ablation": name, "mutated_body_sha256": digest(body), "verification": result})
    return tests


def historical_overlap(rows):
    # Automated exclusion audit only. Diagnostic contents are neither printed nor
    # used for example selection, proof design, or any model/prompt decision.
    development = read_rows(ROOT / "data/development/pairs.jsonl")
    diagnostic = json.loads((ROOT / "data/evaluation/curated.json").read_text())
    if not isinstance(diagnostic, list):
        raise ValueError("Unexpected diagnostic schema")
    out = []
    for row in rows:
        key = equivalence_key(row["formal_statement"])
        matches = {"development": [r["id"] for r in development if equivalence_key(r["formal_statement"]) == key],
                   "diagnostic": [r["id"] for r in diagnostic if equivalence_key(r["formal_statement"]) == key]}
        if row["split"] == "holdout" and any(matches.values()):
            raise ValueError("Holdout duplicates a historical statement up to the conservative equivalence check")
        row["provenance"]["historical_statement_overlap"] = matches
        out.append({"id": row["id"], "split": row["split"], **matches})
    ledger = {family: [r["id"] for r in development if r["generation_family"] in info["historical_families"]]
              for family, info in FAMILIES.items()}
    return {"per_example": out, "historical_development_family_exposure": ledger,
        "holdout_historical_statement_duplicates": 0,
        "method": "Automated definitional reduction, arbitrary binder renaming/permutation, and equality orientation. No result-driven diagnostic selection.",
        "caveat": "The holdout has fresh statements, not historically unseen mathematical families. The conditional-permutation family was already exposed through seed_029. All old seeds remain historical artifacts, outside pilot loaders."}


def prepare():
    if PREPARATION.exists() or MANIFEST.exists():
        raise ValueError("Preparation already sealed; refusing to recurate or reread holdout")
    check_protected()
    rows = [r for split in PATHS for r in read_rows(DATA / PATHS[split])]
    audit_rows(rows, verified=False)
    overlap = historical_overlap(rows)
    review = reference_review()
    token_checks = token_evidence(rows)
    records = []
    for row in rows:
        evidence = asdict(verify(row["formal_statement"], row["proof_body"]))
        if evidence["status"] != "PASS":
            raise ValueError(f"Pilot reference failed: {row['id']}: {evidence}")
        row["verification"] = evidence
        row["provenance"]["rocq_verified"] = True
        row["provenance"]["rocq_source_sha256"] = evidence["source_sha256"]
        row["quality_checks"] = {"decoder": token_checks[row["id"]], "reference_removal": ablations(row)}
        records.append({"id": row["id"], "split": row["split"], "proof_body_sha256": digest(row["proof_body"]),
                        "verification": evidence, "quality_checks": row["quality_checks"]})
        print(f"{row['id']}: PASS", flush=True)
    audit = audit_rows(rows)
    counts = coverage(rows)
    for split, name in PATHS.items():
        write_rows(DATA / name, [r for r in rows if r["split"] == split])
    write_json(DATA / "historical_reference_review.json", review)
    write_json(DATA / "coverage.json", counts)
    write_json(DATA / "family_audit.json", {"within_pilot": audit, "historical_overlap": overlap})
    # Keep holdout proof/stdout details inside reserved/, not in the public report.
    public_records = [r for r in records if r["split"] != "holdout"]
    private_records = [r for r in records if r["split"] == "holdout"]
    write_json(DATA / "reserved/verification_details.json", private_records)
    public_report = {"status": "PASS", "rocq_version": "9.2.0", "references_verified": len(rows),
        "kernel_and_assumptions_checked": True, "frozen_decoder_token_paths_allowed": len(rows),
        "max_reference_tokens_including_eos": max(t["with_eos"] for t in token_checks.values()),
        "reference_removal_checks": sum(len(r["quality_checks"]["reference_removal"]) for r in rows),
        "all_removed_reference_proofs_failed_in_rocq": True,
        "historical_references_reverified": 12, "by_split": {s: {"PASS": c["examples"], "FAIL": 0} for s,c in counts.items()},
        "train_dev_records": public_records, "holdout_details": "reserved/verification_details.json",
        "model_generations": 0, "training_runs": 0}
    write_json(DATA / "verification_report.json", public_report)
    files = [p for p in DATA.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl"}]
    write_json(PREPARATION, {"status": "SEALED_AFTER_REFERENCE_QA", "sealed_at_utc": datetime.now(timezone.utc).isoformat(),
        "files_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(files)},
        "partitions": {s: {"path": PATHS[s], "example_ids": [r["id"] for r in rows if r["split"] == s],
                           "theorem_ids": sorted({r["theorem_id"] for r in rows if r["split"] == s})} for s in PATHS},
        "content_access_policy": "Holdout construction and reference QA complete. Subsequent commands may checksum bytes but must not parse, export, generate on, score, or tune against holdout until after an authorized training run.",
        "model_generations": 0, "training_runs": 0})
    print("PREPARED: 36 verified references; holdout sealed; no model generations or training", flush=True)


def prepared_check():
    check_protected()
    preparation = json.loads(PREPARATION.read_text())
    for name, expected in preparation["files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Sealed pilot artifact changed: {name}")
    return preparation


def freeze():
    if MANIFEST.exists():
        raise ValueError("Split manifest already frozen; refusing overwrite")
    prep = prepared_check()
    # No holdout content load here: use its construction-time committed membership.
    public = read_rows(DATA / PATHS["train"]) + read_rows(DATA / PATHS["dev"])
    audit_rows(public)
    write_json(MANIFEST, {"schema_version": "pilot-splits-1.0", "status": "FROZEN_BEFORE_TRAINING",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(), "dataset_version": "pilot-v1",
        "source_baseline_commit": json.loads(PROTECTED.read_text())["source_commit"],
        "partitions": prep["partitions"], "families": FAMILIES,
        "files_sha256": prep["files_sha256"] | {str(PREPARATION.relative_to(ROOT)): sha256_file(PREPARATION)},
        "policy": {"assignment_unit": "generation_family; never theorem/argument rows",
            "training_allowlist": [PATHS["train"]], "selection_allowlist": [PATHS["dev"]],
            "legacy_data": "All original development, diagnostic and probe artifacts stay historical. Do not append them to training or dev; only frozen Prompt v2 demonstrations remain in the prompt.",
            "holdout": "Fresh reference-verified statements; sealed after construction QA. No model use until after training and checkpoint selection. No future tuning after unsealing.",
            "independence_limit": "Family-disjoint from the new train/dev splits, but historical family exposure exists. This is not a test of unseen mathematics or pretraining contamination.",
            "post_training_protocol": "Fix a checkpoint using pilot dev only, then open holdout once. Compare that checkpoint with frozen constrained-v1 under theorem-only and theorem-plus-informal, one attempt, no repair, 256 tokens. Report verified argument following separately from step-contract matching; the pairs are correlated."},
        "model_generations": 0, "training_runs": 0})
    print("FROZEN: train 24 / dev 6 / holdout 6; holdout not parsed during freeze")


def check():
    prepared_check()
    release = ROOT / "pilot_v1/release.json"
    if release.exists():
        for name, expected in json.loads(release.read_text())["files_sha256"].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f"Pilot release file changed: {name}")
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "FROZEN_BEFORE_TRAINING" or manifest["families"] != FAMILIES:
        raise ValueError("Invalid split freeze")
    for name, expected in manifest["files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
    public = load_split("train") + load_split("dev")
    audit_rows(public)
    for split in ("train", "dev"):
        if [r["id"] for r in public if r["split"] == split] != manifest["partitions"][split]["example_ids"]:
            raise ValueError("Split membership changed")
    if manifest["partitions"] != json.loads(PREPARATION.read_text())["partitions"]:
        raise ValueError("Holdout/public membership differs from sealed preparation")
    return {"status": "PASS", "protected_historical_files": check_protected(), "public_examples": len(public),
            "holdout_access": "bytes checked against committed hashes only; no content parsing", "training_runs": 0, "model_generations": 0}


def reverify():
    check()
    outcomes = []
    for split in ("train", "dev"):
        for row in load_split(split):
            actual = asdict(verify(row["formal_statement"], row["proof_body"]))
            for field in ("status", "category", "source_sha256", "kernel_checked", "assumptions_checked"):
                if actual[field] != row["verification"][field]:
                    raise ValueError(f"Rocq replay differs: {row['id']} {field}")
            outcomes.append(row["id"])
    return {"status": "PASS", "train_dev_reverified": len(outcomes), "holdout_reopened": False,
            "note": "All holdout references were verified during construction; retained hash-bound evidence is checked without reopening them."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "freeze", "check", "reverify", "export"])
    parser.add_argument("--split", choices=["train", "dev"], default="train")
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare()
    elif args.action == "freeze":
        freeze()
    elif args.action == "check":
        print(json.dumps(check(), indent=2))
    elif args.action == "reverify":
        print(json.dumps(reverify(), indent=2))
    else:
        if not args.output:
            parser.error("export requires --output")
        check()
        # Exclusive creation prevents an export from overwriting frozen inputs.
        with open(args.output, "x") as stream:
            for record in training_records(args.split):
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Exported {args.split}; no model loaded")


if __name__ == "__main__":
    main()
