"""Compile the .v example, verify its JSON proof, and persist actual evidence."""

from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verifier import LOCK_FILE, check_environment, run_process, sha256_file, verify  # noqa: E402


def main() -> int:
    example_file = ROOT / "examples/add_zero_right.v"
    record_file = ROOT / "examples/add_zero_right.json"
    record = json.loads(record_file.read_text())
    result = verify(record["input"]["formal_statement"], record["target"]["proof_body"])
    standalone = {"status": "FAIL", "stdout": "", "stderr": "", "returncode": None}
    try:
        with tempfile.TemporaryDirectory(prefix="proof-bridge-example-") as temporary:
            work = Path(temporary)
            _, compiler = check_environment(work)
            source = work / "add_zero_right.v"
            source.write_text(example_file.read_text())
            process = run_process([str(compiler), "compile", "-q", source.name], work, 10)
            passed = (process.returncode == 0 and source.with_suffix(".vo").is_file()
                      and "Closed under the global context" in process.stdout.splitlines())
            standalone = {"status": "PASS" if passed else "FAIL", "returncode": process.returncode,
                          "stdout": process.stdout, "stderr": process.stderr}
    except Exception as error:
        # Persist a failed result rather than leaving a stale successful record.
        standalone["stderr"] = str(error)
    passed = result.status == "PASS" and standalone["status"] == "PASS"
    lock = json.loads(LOCK_FILE.read_text())
    record["environment_id"] = lock["environment_id"]
    record["environment_lock"] = {"file": "environment.lock.json", "sha256": sha256_file(LOCK_FILE)}
    verification = asdict(result)
    verification.update({
        "status": "PASS" if passed else "FAIL", "training_eligible": passed,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "example_file_sha256": sha256_file(example_file), "standalone_compile": standalone,
    })
    if result.status == "PASS" and not passed:
        verification["category"] = "EXAMPLE_COMPILE_ERROR"
        verification["message"] = "JSON proof passed, but the standalone example failed."
    record["verification"] = verification
    record_file.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(verification, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
