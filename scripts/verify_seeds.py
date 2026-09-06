"""Build or recheck the 30 individually curated seed pairs. Never trains a model."""

from __future__ import annotations

import argparse
from collections import Counter
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
        for field in ["title", "split_group", "method", "formal_statement", "informal_statement"]:
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
        body = "\n".join(step["code"] for step in seed["steps"])
        # Validate the unchanged verifier policy before creating any data artifact.
        render_source(seed["formal_statement"], body)
        tokens = tokenize(body)
        if ("induction" in tokens) != (seed["method"] == "induction"):
            raise ValueError(f"{seed['id']}: method annotation disagrees with proof")
        if seed["method"] == "induction":
            if seed.get("induction_variable") != tokens[tokens.index("induction") + 1]:
                raise ValueError(f"{seed['id']}: incorrect induction variable annotation")
    return seeds


def make_record(seed: dict) -> dict:
    lock = json.loads(LOCK_FILE.read_text())
    proof_body = "\n".join(step["code"] for step in seed["steps"])
    informal_proof = " ".join(step["text"] for step in seed["steps"])
    alignment = []
    first_line = 1
    for index, step in enumerate(seed["steps"], start=1):
        last_line = first_line + len(step["code"].splitlines()) - 1
        alignment.append({"step": index, "informal_text": step["text"],
                          "proof_line_start": first_line, "proof_line_end": last_line})
        first_line = last_line + 1
    record = {
        "schema_version": "seed-0.1",
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
        "provenance": {
            "author": "assistant",
            "curation": "Individually selected, authored, and reviewed; no enumeration or paraphrase expansion.",
            "source_file": "data/seeds/curated.json",
            "related_existing_example": seed.get("related_existing_example"),
            "human_reviewed": False,
            "informal_alignment_status": "assistant_reviewed; not kernel-certified",
        },
    }
    return record


def input_fingerprints() -> dict[str, str]:
    return {
        "curated_source_sha256": sha256_file(SOURCE),
        "environment_lock_sha256": sha256_file(LOCK_FILE),
        "verifier_sha256": sha256_file(ROOT / "verifier.py"),
        "build_script_sha256": sha256_file(Path(__file__)),
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="Verify and refresh pairs.jsonl plus verification_report.json")
    mode.add_argument("--check", action="store_true", help="Check stored evidence and recompile every pair (default)")
    args = parser.parse_args()
    seeds = load_curated()
    records = [make_record(seed) for seed in seeds]
    if not args.write:
        assert_artifacts_current(records)
    checked_at = datetime.now(timezone.utc).isoformat()
    outcomes = []
    for record in records:
        result = verify(record["input"]["formal_statement"], record["target"]["proof_body"])
        record["verification"] = asdict(result) | {
            "verified_at_utc": checked_at, "training_eligible": result.status == "PASS",
        }
        outcomes.append({"id": record["id"], "status": result.status, "category": result.category,
                         "source_sha256": result.source_sha256})
        print(f"{record['id']} {result.status} {result.category}", flush=True)
        if result.status != "PASS":
            print(result.message + "\n" + result.stdout + result.stderr, flush=True)
    passed = sum(outcome["status"] == "PASS" for outcome in outcomes)
    if args.write:
        serialized = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
        PAIRS.write_text(serialized, encoding="utf-8")
        lock = json.loads(LOCK_FILE.read_text())
        report = {
            "dataset_id": DATASET_ID,
            "status": "PASS" if passed == len(records) else "FAIL",
            "verified_at_utc": checked_at,
            "compiler_version": lock["rocq_package_version"],
            "stdlib_version": lock["stdlib_version"],
            "fingerprints": input_fingerprints(),
            "pairs_sha256": sha256_file(PAIRS),
            "counts": {"total": len(records), "passed": passed, "failed": len(records) - passed,
                       "unique_statements_modulo_binder_names": len({statement_key(s['formal_statement']) for s in seeds}),
                       "by_method": dict(Counter(s["method"] for s in seeds)),
                       "by_split_group": dict(Counter(s["split_group"] for s in seeds))},
            "global_helpers_enabled": [],
            "isolation": "Each pair compiled in a fresh verifier temporary directory.",
            "human_reviewed": False,
            "training_performed": False,
            "results": outcomes,
        }
        REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{passed}/{len(records)} PASS; {len(records) - passed} FAIL")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, Rejection) as error:
        print(f"Seed verification failed: {error}", file=sys.stderr)
        raise SystemExit(1)
