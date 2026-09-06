"""Build or recheck the 30 individually curated seed pairs. Never trains a model."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verifier import (  # noqa: E402
    LOCK_FILE, NO_ARGUMENT_TACTICS, REFERENCE_TACTICS, Rejection,
    StatementParser, render_source, sha256_file, tokenize, verify,
)
from scripts.seed_schema import (  # noqa: E402
    FAMILIES, SCHEMA_VERSION, argument_features, audit_families, authored_body,
    canonical_body, informal_paragraph, steps_sha256,
)

DATA_DIR = ROOT / "data/seeds"
SOURCE = DATA_DIR / "curated.json"
PAIRS = DATA_DIR / "pairs.jsonl"
REPORT = DATA_DIR / "verification_report.json"
DATASET_ID = "mini_nat_addition_seeds_v1"
ALLOWED_TACTICS = sorted(NO_ARGUMENT_TACTICS | REFERENCE_TACTICS | {"intros", "induction"})
METHODS = {"definition", "equality_premise", "induction"}


def statement_key(statement: str) -> tuple[str, ...]:
    """Detect exact structural duplicates despite whitespace or binder renaming."""
    tokens = tokenize(statement)
    parser = StatementParser(tokens)
    parser.parse()
    renamed = {name: f"v{index}" for index, name in enumerate(parser.variables)}
    return tuple(renamed.get(token, token) for token in tokens)


def load_curated(filename: Path = SOURCE) -> list[dict]:
    seeds = json.loads(filename.read_text(encoding="utf-8"))
    if not isinstance(seeds, list) or len(seeds) != 30:
        raise ValueError("This seed release must contain exactly 30 curated records.")
    seen_statements = set()
    for index, seed in enumerate(seeds, start=1):
        if not isinstance(seed, dict) or seed.get("id") != f"seed_{index:03d}":
            raise ValueError("Expected unique, ordered IDs seed_001 through seed_030.")
        for field in ["title", "split_group", "generation_family", "method", "formal_statement", "informal_statement"]:
            if not isinstance(seed.get(field), str) or not seed[field].strip():
                raise ValueError(f"{seed['id']}: missing {field}")
        if seed["method"] not in METHODS:
            raise ValueError(f"{seed['id']}: unsupported method")
        if not isinstance(seed.get("steps"), list) or not seed["steps"]:
            raise ValueError(f"{seed['id']}: no aligned proof steps")
        for step in seed["steps"]:
            if not isinstance(step, dict) or any(not isinstance(step.get(k), str) or not step[k].strip() for k in ("text", "code")):
                raise ValueError(f"{seed['id']}: incomplete step alignment")
        key = statement_key(seed["formal_statement"])
        if key in seen_statements:
            raise ValueError(f"{seed['id']}: duplicate statement after binder normalization")
        seen_statements.add(key)
        body = authored_body(seed)
        # Validate the unchanged verifier policy before creating any data artifact.
        render_source(seed["formal_statement"], body)
        tokens = tokenize(body)
        if ("induction" in tokens) != (seed["method"] == "induction"):
            raise ValueError(f"{seed['id']}: method annotation disagrees with proof")
        if seed["method"] == "induction":
            if seed.get("induction_variable") != tokens[tokens.index("induction") + 1]:
                raise ValueError(f"{seed['id']}: incorrect induction variable annotation")
        if seed.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"{seed['id']}: unsupported schema version")
        if "related_existing_example" in seed or "provenance" in seed:
            raise ValueError(f"{seed['id']}: research fields and provenance belong under metadata")
        metadata = seed.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("proof_normalizations"), list):
            raise ValueError(f"{seed['id']}: missing metadata/proof_normalizations")
        if metadata.get("authored_steps_sha256") != steps_sha256(seed):
            raise ValueError(f"{seed['id']}: original aligned steps changed")
        canonical = canonical_body(seed)
        render_source(seed["formal_statement"], canonical)
        if seed.get("proof_body") != canonical:
            raise ValueError(f"{seed['id']}: proof_body disagrees with steps and recorded normalizations")
        if seed.get("informal_proof") != informal_paragraph(seed):
            raise ValueError(f"{seed['id']}: informal_proof disagrees with aligned text")
        features = argument_features(seed["formal_statement"], canonical)
        # JSON booleans are required, not numeric 0/1 lookalikes.
        if json.dumps(seed.get("argument_features"), sort_keys=True) != json.dumps(features, sort_keys=True):
            raise ValueError(f"{seed['id']}: argument_features disagree with the proof")
        provenance = metadata.get("provenance", {})
        if (provenance.get("informal_proof_source") != "assistant-curated"
                or provenance.get("formal_proof_source") != "assistant-curated"
                or provenance.get("human_reviewed") is not False
                or type(provenance.get("rocq_verified")) is not bool):
            raise ValueError(f"{seed['id']}: missing or incorrect provenance")
        if provenance["rocq_verified"]:
            expected_hash = hashlib.sha256(render_source(seed["formal_statement"], canonical).encode()).hexdigest()
            if provenance.get("rocq_source_sha256") != expected_hash:
                raise ValueError(f"{seed['id']}: stale Rocq provenance hash")
    audit_families(seeds)
    return seeds


def make_record(seed: dict) -> dict:
    lock = json.loads(LOCK_FILE.read_text())
    proof_body = canonical_body(seed)
    informal_proof = informal_paragraph(seed)
    alignment = []
    first_line = 1
    for index, step in enumerate(seed["steps"], start=1):
        last_line = first_line + len(step["code"].splitlines()) - 1
        alignment.append({"step": index, "informal_text": step["text"],
                          "proof_line_start": first_line, "proof_line_end": last_line})
        first_line = last_line + 1
    record = deepcopy(seed) | {
        "schema_version": SCHEMA_VERSION,
        "dataset_id": DATASET_ID,
        "id": seed["id"],
        "title": seed["title"],
        "split": "unassigned",
        "split_group": seed["split_group"],
        "environment_id": lock["environment_id"],
        "environment_lock": {"file": "environment.lock.json", "sha256": sha256_file(LOCK_FILE)},
        "input": {
            "raw_text": seed["informal_statement"] + "\n\n" + informal_proof,
            "normalized_statement": seed["informal_statement"],
            "normalized_proof": informal_proof,
            "formal_statement": seed["formal_statement"],
            "allowed_tactics": ALLOWED_TACTICS,
            "allowed_global_lemmas": [],
        },
        "target": {
            "proof_body": proof_body,
            "rocq_source": render_source(seed["formal_statement"], proof_body),
            "method": seed["method"],
            "induction_variable": seed.get("induction_variable"),
            "tactics_used": sorted(set(tokenize(proof_body)) & set(ALLOWED_TACTICS)),
            "global_lemmas_used": [],
        },
        "alignment": alignment,
    }
    return record


def input_fingerprints() -> dict[str, str]:
    return {
        "curated_source_sha256": sha256_file(SOURCE),
        "environment_lock_sha256": sha256_file(LOCK_FILE),
        "verifier_sha256": sha256_file(ROOT / "verifier.py"),
        "build_script_sha256": sha256_file(Path(__file__)),
        "schema_script_sha256": sha256_file(ROOT / "scripts/seed_schema.py"),
        "family_manifest_sha256": sha256_file(FAMILIES),
    }


def assert_artifacts_current(records: list[dict], pairs_file: Path = PAIRS, report_file: Path = REPORT) -> None:
    stored = [json.loads(line) for line in pairs_file.read_text().splitlines() if line.strip()]
    report = json.loads(report_file.read_text())
    if report.get("fingerprints") != input_fingerprints():
        raise ValueError("Stored verification inputs changed; run verify_seeds.py --write.")
    if report.get("pairs_sha256") != sha256_file(pairs_file):
        raise ValueError("Stored pair file checksum disagrees with its verification report.")
    if len(stored) != len(records):
        raise ValueError("Stored dataset size differs from the curated release.")
    if report.get("status") != "PASS" or report.get("counts", {}).get("passed") != len(records):
        raise ValueError("The stored report does not record a complete successful run.")
    if report.get("family_audit") != audit_families(records):
        raise ValueError("Stored duplicate/family audit disagrees with current records.")
    for actual, expected in zip(stored, records):
        evidence = actual.pop("verification", {})
        if actual != expected:
            raise ValueError(f"Stored pair differs from curated source: {expected['id']}")
        expected_source_hash = hashlib.sha256(expected["target"]["rocq_source"].encode()).hexdigest()
        if (evidence.get("status") != "PASS" or evidence.get("category") != "VERIFIED"
                or evidence.get("kernel_checked") is not True
                or evidence.get("assumptions_checked") is not True
                or evidence.get("source_sha256") != expected_source_hash):
            raise ValueError(f"Missing or stale verification evidence: {expected['id']}")
        if expected["metadata"]["provenance"]["rocq_verified"] is not True:
            raise ValueError(f"Missing verified provenance: {expected['id']}")
        if authored_body(expected) != expected["proof_body"]:
            original = evidence.get("authored_proof", {})
            original_hash = hashlib.sha256(render_source(expected["formal_statement"], authored_body(expected)).encode()).hexdigest()
            if (original.get("status") != "PASS" or original.get("category") != "VERIFIED"
                    or original.get("kernel_checked") is not True or original.get("assumptions_checked") is not True
                    or original.get("source_sha256") != original_hash):
                raise ValueError(f"Missing original proof verification: {expected['id']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Verify and refresh pairs.jsonl plus verification_report.json")
    mode.add_argument("--check", action="store_true", help="Check stored evidence and recompile every pair (default)")
    mode.add_argument("--audit", action="store_true", help="Check schema, duplicates, and families without compilation or writes")
    args = parser.parse_args()
    seeds = load_curated()
    family_audit = audit_families(seeds)
    if args.audit:
        print(json.dumps(family_audit | {"total": len(seeds), "unique_statements_modulo_binder_names": len({statement_key(s['formal_statement']) for s in seeds})}, indent=2))
        return 0
    records = [make_record(seed) for seed in seeds]
    if not args.write:
        assert_artifacts_current(records)
    checked_at = datetime.now(timezone.utc).isoformat()
    outcomes = []
    original_outcomes = []
    for seed, record in zip(seeds, records):
        result = verify(record["input"]["formal_statement"], record["target"]["proof_body"])
        record["verification"] = asdict(result) | {
            "verified_at_utc": checked_at, "training_eligible": result.status == "PASS",
        }
        provenance = record["metadata"]["provenance"]
        provenance["rocq_verified"] = result.status == "PASS"
        provenance["rocq_source_sha256"] = result.source_sha256 if result.status == "PASS" else None
        seed["metadata"]["provenance"] = deepcopy(provenance)
        if authored_body(seed) != record["proof_body"]:
            original = verify(seed["formal_statement"], authored_body(seed))
            record["verification"]["authored_proof"] = asdict(original) | {"verified_at_utc": checked_at}
            original_outcomes.append({"id": seed["id"], "status": original.status, "category": original.category,
                                      "source_sha256": original.source_sha256})
            print(f"{seed['id']} original {original.status} {original.category}", flush=True)
            if original.status != "PASS":
                print(original.message + "\n" + original.stdout + original.stderr, flush=True)
        outcomes.append({"id": record["id"], "status": result.status, "category": result.category,
                         "source_sha256": result.source_sha256})
        print(f"{record['id']} {result.status} {result.category}", flush=True)
        if result.status != "PASS":
            print(result.message + "\n" + result.stdout + result.stderr, flush=True)
    passed = sum(outcome["status"] == "PASS" for outcome in outcomes)
    originals_passed = sum(outcome["status"] == "PASS" for outcome in original_outcomes)
    success = passed == len(records) and originals_passed == len(original_outcomes)
    if args.write:
        # Provenance is refreshed from this run, including false on failures.
        if json.loads(SOURCE.read_text()) != seeds:
            SOURCE.write_text(json.dumps(seeds, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        serialized = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        PAIRS.write_text(serialized, encoding="utf-8")
        lock = json.loads(LOCK_FILE.read_text())
        report = {
            "dataset_id": DATASET_ID,
            "schema_version": SCHEMA_VERSION,
            "status": "PASS" if success else "FAIL",
            "verified_at_utc": checked_at,
            "compiler_version": lock["rocq_package_version"],
            "stdlib_version": lock["stdlib_version"],
            "fingerprints": input_fingerprints(),
            "pairs_sha256": sha256_file(PAIRS),
            "counts": {"total": len(records), "passed": passed, "failed": len(records) - passed,
                       "unique_statements_modulo_binder_names": len({statement_key(s['formal_statement']) for s in seeds}),
                       "by_method": dict(Counter(s["method"] for s in seeds)),
                       "by_split_group": dict(Counter(s["split_group"] for s in seeds)),
                       "by_generation_family": dict(Counter(s["generation_family"] for s in seeds)),
                       "normalized_proof_bodies": len(original_outcomes),
                       "original_bodies_rechecked": len(original_outcomes),
                       "original_bodies_passed": originals_passed},
            "family_audit": family_audit,
            "global_helpers_enabled": [],
            "isolation": "Each pair compiled in a fresh verifier temporary directory.",
            "human_reviewed": False,
            "training_performed": False,
            "results": outcomes,
            "original_proof_results": original_outcomes,
        }
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{passed}/{len(records)} PASS; {len(records) - passed} FAIL")
    print(f"Originals for normalized bodies: {originals_passed}/{len(original_outcomes)} PASS")
    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, Rejection) as error:
        print(f"Seed verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
