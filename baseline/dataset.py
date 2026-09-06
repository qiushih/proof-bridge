"""Prepare frozen development data and verify the diagnostic evaluation references."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.seed_schema import argument_features, authored_body, definitional_key, informal_paragraph
from scripts.verify_seeds import PAIRS, SOURCE, assert_artifacts_current, load_curated, make_record, statement_key
from verifier import LOCK_FILE, render_source, sha256_file, verify

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT = ROOT / "data/development"
DEV_PAIRS = DEVELOPMENT / "pairs.jsonl"
DEV_MANIFEST = DEVELOPMENT / "manifest.json"
EVAL_SOURCE = ROOT / "data/evaluation/curated.json"
EVAL_REPORT = ROOT / "data/evaluation/verification_report.json"


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_development():
    manifest = json.loads(DEV_MANIFEST.read_text())
    for relative, expected in manifest["files_sha256"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"Frozen development input changed: {relative}")
    records = [json.loads(line) for line in DEV_PAIRS.read_text().splitlines()]
    if len(records) != 30 or any(record["split"] != "development" for record in records):
        raise ValueError("Expected 30 frozen development records")
    if [record["id"] for record in records] != manifest["ids"]:
        raise ValueError("Frozen development IDs changed")
    return records


def freeze_development():
    if DEV_MANIFEST.exists():
        return check_development()
    if DEV_PAIRS.exists():
        raise ValueError("Development snapshot already exists without its manifest; refusing overwrite")
    seeds = load_curated()
    assert_artifacts_current([make_record(seed) for seed in seeds])
    records = [json.loads(line) for line in PAIRS.read_text().splitlines()]
    for record in records:
        record["split"] = "development"
    DEVELOPMENT.mkdir(parents=True, exist_ok=True)
    DEV_PAIRS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    files = [DEV_PAIRS, SOURCE, PAIRS, ROOT / "data/seeds/families.json",
             ROOT / "data/seeds/verification_report.json", ROOT / "verifier.py", LOCK_FILE]
    write_json(DEV_MANIFEST, {
        "schema_version": "development-freeze-0.1", "split": "development",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "count": 30, "ids": [r["id"] for r in records],
        "files_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in files},
        "policy": "Immutable baseline-v1 development snapshot; no seed is included in evaluation metrics or model prompts.",
        "fine_tuning_performed": False,
    })
    return check_development()


def load_evaluation():
    examples = json.loads(EVAL_SOURCE.read_text())
    if not isinstance(examples, list) or len(examples) != 12:
        raise ValueError("Baseline v1 requires exactly 12 curated evaluation examples")
    for index, example in enumerate(examples, 1):
        if example["id"] != f"eval_{index:03d}":
            raise ValueError("Expected ordered unique evaluation IDs")
        body = authored_body(example)
        render_source(example["formal_statement"], body)
        example["proof_body"] = body
        example["informal_proof"] = informal_paragraph(example)
        example["argument_features"] = argument_features(example["formal_statement"], body)
        example["split"] = "evaluation_diagnostic"
    return examples


def leakage_audit(development, examples):
    dev_keys = {statement_key(r["formal_statement"]) for r in development}
    dev_definitions = {definitional_key(r["formal_statement"]) for r in development}
    keys, definitions = set(), set()
    for example in examples:
        key = statement_key(example["formal_statement"])
        definition = definitional_key(example["formal_statement"])
        if key in keys or key in dev_keys or definition in definitions or definition in dev_definitions:
            raise ValueError(f"Duplicate/equivalent evaluation statement: {example['id']}")
        keys.add(key)
        definitions.add(definition)
    dev_families = {r["generation_family"] for r in development}
    overlaps = sorted({r["generation_family"] for r in examples} & dev_families)
    return {
        "status": "PASS", "development_count": len(development), "evaluation_count": len(examples),
        "exact_or_alpha_duplicates": 0, "definitional_duplicates": 0,
        "overlapping_generation_families": overlaps,
        "family_disjoint": not overlaps,
        "interpretation": "Statement-disjoint diagnostic evaluation; NOT a leakage-safe held-out-family benchmark. No development examples are provided in prompts and no model is trained.",
    }


def evaluation_fingerprints():
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in
            [EVAL_SOURCE, DEV_MANIFEST, ROOT / "baseline/dataset.py", ROOT / "scripts/seed_schema.py",
             ROOT / "verifier.py", LOCK_FILE]}


def check_evaluation_evidence():
    development = check_development()
    examples = load_evaluation()
    report = json.loads(EVAL_REPORT.read_text())
    if report["fingerprints"] != evaluation_fingerprints():
        raise ValueError("Evaluation reference verification is stale")
    if report["leakage_audit"] != leakage_audit(development, examples):
        raise ValueError("Evaluation leakage audit changed")
    if report["status"] != "PASS" or len(report["results"]) != len(examples):
        raise ValueError("Evaluation references have not all passed")
    for example, evidence in zip(examples, report["results"]):
        result = evidence["verification"]
        if (evidence["id"] != example["id"] or result["status"] != "PASS"
                or result["kernel_checked"] is not True or result["assumptions_checked"] is not True
                or result["source_sha256"] != digest(render_source(example["formal_statement"], example["proof_body"]))):
            raise ValueError("Invalid evaluation reference evidence")
    return examples


def prepare(check=False):
    development = check_development() if check else freeze_development()
    examples = load_evaluation()
    audit = leakage_audit(development, examples)
    if check:
        check_evaluation_evidence()
    outcomes = []
    for example in examples:
        result = verify(example["formal_statement"], example["proof_body"])
        outcomes.append({"id": example["id"], "verification": asdict(result)})
        print(f"{example['id']} reference {result.status} {result.category}", flush=True)
        if result.status != "PASS":
            print(result.message + result.stderr, flush=True)
    success = all(r["verification"]["status"] == "PASS" for r in outcomes)
    if not check:
        write_json(EVAL_REPORT, {
            "schema_version": "evaluation-reference-0.1", "status": "PASS" if success else "FAIL",
            "verified_at_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprints": evaluation_fingerprints(), "leakage_audit": audit,
            "provenance": {"informal_proof_source": "assistant-curated", "formal_proof_source": "assistant-curated",
                           "human_reviewed": False, "rocq_verified": success},
            "results": outcomes,
        })
    print(json.dumps(audit, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Recheck frozen data and references without writing")
    args = parser.parse_args()
    raise SystemExit(prepare(args.check))
