"""Freeze/check constrained-v1 without modifying its decoder or results."""

import argparse
from datetime import datetime, timezone
import json
import subprocess

from baseline.dataset import ROOT, write_json
from constrained_v1.experiment import check as check_run, fingerprints
from constrained_v1.report import build_report
from verifier import sha256_file

CHECKPOINT = ROOT / "checkpoints/constrained-v1.json"


def check():
    checkpoint = json.loads(CHECKPOINT.read_text())
    if checkpoint["status"] != "FROZEN":
        raise ValueError("Constrained checkpoint is not frozen")
    for name, expected in checkpoint["files_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen constrained-v1 file changed: {name}")
    return checkpoint


def freeze():
    if CHECKPOINT.exists():
        raise ValueError("Refusing to replace an existing checkpoint")
    from baseline.setup_model import check_model
    model = check_model()
    manifest, rows = check_run()
    summary, _ = build_report()
    paths = set()
    for folder in ("constrained_v1", "results/constrained-v1", "results/prompt-v2-development-audit"):
        paths.update(p for p in (ROOT / folder).iterdir() if p.is_file())
    paths.update((ROOT / "tests").glob("test_*.py"))
    paths.update(ROOT / name for name in ("checkpoints/__init__.py", "checkpoints/freeze_constrained_v1.py",
                 "checkpoints/README.md", "scripts/audit_prompt_v2_development.py"))
    hashes = fingerprints() | {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(paths)}
    log = (ROOT / ".cache/constrained-v1-freeze-tests.log").read_text()
    if "Ran 96 tests" not in log or "\nOK\n" not in log:
        raise ValueError("Expected fresh 96-test regression pass before freezing")
    write_json(CHECKPOINT, {
        "schema_version": "constrained-checkpoint-1.0", "status": "FROZEN",
        "release_tag": "proofbridge-constrained-v1", "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "experiment_id": manifest["experiment_id"], "development_examples": 30, "attempts": len(rows),
        "generation_settings": manifest["generation_settings"], "model": model,
        "common_subset_verified_faithful": summary["success"], "files_sha256": hashes,
        "validation": {"status": "PASS", "regression_tests": 96, "saved_run_and_reviews_checked": True,
                       "model_files_checked": True, "new_generations_during_freeze": 0},
        "policy": "Immutable checkpoint. Future probes live separately; do not edit decoder, prompts, existing seeds, or original results. No diagnostic examples used to choose this checkpoint.",
    })
    print(f"FROZEN: {len(hashes)} files; 60 saved attempts; zero new generations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "check"])
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    else:
        checkpoint = check()
        print(f"PASS: {len(checkpoint['files_sha256'])} constrained-v1 checkpoint hashes unchanged")
