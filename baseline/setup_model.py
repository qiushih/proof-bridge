"""Download the pinned public model; inference runs offline after this step."""

import argparse
import json
import os
from pathlib import Path

from baseline.dataset import ROOT, write_json
from verifier import sha256_file

MODEL_LOCK = ROOT / "baseline/model.lock.json"
MODEL_DIR = ROOT / ".cache/baseline/model"


def check_model():
    lock = json.loads(MODEL_LOCK.read_text())
    if set(lock["files_sha256"]) != set(lock["files"]):
        raise ValueError("Model lock does not contain every artifact checksum")
    for name, expected in lock["files_sha256"].items():
        if sha256_file(MODEL_DIR / name) != expected:
            raise ValueError(f"Pinned model file changed: {name}")
    return lock


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check local model hashes without network")
    parser.add_argument("--record-hashes", action="store_true", help="Bootstrap an empty lock once, from its pinned revision")
    args = parser.parse_args()
    if args.check:
        check_model()
    else:
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        from huggingface_hub import snapshot_download
        lock = json.loads(MODEL_LOCK.read_text())
        if args.record_hashes and lock["files_sha256"]:
            raise ValueError("Refusing to replace an existing model lock")
        snapshot_download(repo_id=lock["model_id"], revision=lock["revision"],
                          allow_patterns=lock["files"], local_dir=MODEL_DIR, max_workers=2)
        if args.record_hashes:
            lock["files_sha256"] = {name: sha256_file(MODEL_DIR / name) for name in lock["files"]}
            write_json(MODEL_LOCK, lock)
        check_model()
    print("Pinned model artifacts verified")


if __name__ == "__main__":
    main()
