import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from argument_switch_v1.experiment import OUTPUT, PREFLIGHT, check
from argument_switch_v1.report import build_report
from baseline.dataset import ROOT
from checkpoints.freeze_constrained_v1 import check as check_checkpoint
from prompt_v2.experiment import Inference
from verifier import sha256_file


@unittest.skipUnless((OUTPUT / "completion.json").exists(), "Argument-switch run not complete")
class ArgumentSwitchResultTests(unittest.TestCase):
    def test_exactly_18_attempts_after_reference_preflight_and_unchanged_checkpoint(self):
        checkpoint = check_checkpoint()
        manifest, rows = check()
        preflight = json.loads(PREFLIGHT.read_text())
        self.assertEqual(checkpoint["release_tag"], "proofbridge-constrained-v1")
        self.assertEqual(len(rows), 18)
        self.assertLess(preflight["completed_at_utc"], manifest["started_at_utc"])
        self.assertEqual(len(preflight["references"]), 12)
        self.assertTrue(all(r["verification"]["status"] == "PASS" for r in preflight["references"]))
        self.assertTrue(all(r["attempt"] == 1 and r["repair_attempts"] == 0 for r in rows))

    def test_verification_and_strategy_switching_are_distinct(self):
        _, _, extra = build_report()
        summary = extra["summary"]
        self.assertEqual(summary["all_outputs"]["verified"], 9)
        self.assertEqual(summary["switching"]["successful_theorems"], 0)
        self.assertEqual(summary["switching"]["attempted_switches"], 3)
        self.assertEqual(summary["switching"]["theorems"], 6)

    def test_report_and_bound_manual_reviews_reproduce(self):
        text, rows, extra = build_report()
        self.assertEqual((OUTPUT / "ARGUMENT_SWITCH_REPORT.md").read_text(), text)
        self.assertEqual(json.loads((OUTPUT / "report_summary.json").read_text()), extra)
        self.assertEqual(len(rows), 18)
        self.assertTrue(all(r["manual_review"]["proof_body_sha256"] == r["proof_body_sha256"] for r in rows))
        self.assertTrue(all(not r["manual_review"]["human_reviewed"] for r in rows))

    def test_no_probe_inference_or_diagnostic_reads_during_analysis(self):
        original_open = Path.open
        def guarded(path, *args, **kwargs):
            relative = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)
            if any(relative.startswith(prefix) for prefix in ("data/evaluation/", "results/baseline-v1/", "results/baseline-v2/")):
                raise AssertionError("Probe consulted the diagnostic evaluation")
            return original_open(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch.object(Inference, "__init__", side_effect=AssertionError("No new generation")):
            build_report()

    def test_missing_outputs_or_duplicate_attempts_rejected_even_with_updated_checksum(self):
        for name in ("results.jsonl", "attempts.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                for file in ("manifest.json", "results.jsonl", "attempts.jsonl", "summary.json", "completion.json"):
                    (directory / file).write_bytes((OUTPUT / file).read_bytes())
                lines = (directory / name).read_text().splitlines()
                lines = lines[:-1] if name == "results.jsonl" else lines + lines[:1]
                (directory / name).write_text("\n".join(lines) + "\n")
                completion = json.loads((directory / "completion.json").read_text())
                completion["files_sha256"][name] = sha256_file(directory / name)
                (directory / "completion.json").write_text(json.dumps(completion))
                with self.assertRaises(ValueError):
                    check(directory)


if __name__ == "__main__":
    unittest.main()
