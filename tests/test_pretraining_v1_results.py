import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pilot_v1.protocol import ROOT
from prompt_v2.experiment import Inference
from training_protocol_v1.experiment import OUTPUT, check
from training_protocol_v1.report import build_report
from verifier import sha256_file


@unittest.skipUnless((OUTPUT/"manual_reviews.json").exists(), "Pre-training baseline/reviews not complete")
class PretrainingResultTests(unittest.TestCase):
    def test_nine_attempts_and_protocol_before_run(self):
        manifest, rows = check()
        self.assertEqual(len(rows),9)
        self.assertLess(manifest["protocol_frozen_at_utc"],manifest["started_at_utc"])
        self.assertTrue(all(r["attempt"] == 1 and r["repair_attempts"] == 0 for r in rows))

    def test_review_and_report_reproduce(self):
        report,rows,summary = build_report()
        self.assertEqual((OUTPUT/"PRETRAINING_DEV_BASELINE.md").read_text(),report)
        self.assertEqual(json.loads((OUTPUT/"summary.json").read_text()),summary)
        self.assertEqual(len(rows),9)
        self.assertTrue(all(r["mathematical_argument_fidelity"]["proof_body_sha256"] == r["proof_body_sha256"] for r in rows))
        self.assertTrue(all(r["mathematical_argument_fidelity"]["status"] == "NOT_APPLICABLE" for r in rows if r["condition"] == "theorem_only"))

    def test_analysis_does_not_load_a_model_or_read_sealed_content(self):
        original = Path.open
        def guarded(path,*args,**kwargs):
            mode=args[0] if args else kwargs.get("mode","r")
            if str(path.resolve()).startswith((str(ROOT/"data/pilot-v1/reserved"),str(ROOT/"data/evaluation"))) and "b" not in mode:
                raise AssertionError("Sealed/diagnostic content inspected")
            return original(path,*args,**kwargs)
        with patch.object(Path,"open",guarded), patch.object(Inference,"__init__",side_effect=AssertionError("No new inference")):
            check()
            build_report()

    def test_missing_output_and_extra_attempt_rejected_even_after_rehash(self):
        for name in ("raw_generations.jsonl","attempts.jsonl"):
            with tempfile.TemporaryDirectory() as temp:
                directory=Path(temp)
                for f in ("manifest.json","completion.json","raw_generations.jsonl","attempts.jsonl"):
                    (directory/f).write_bytes((OUTPUT/f).read_bytes())
                lines=(directory/name).read_text().splitlines()
                lines=lines[:-1] if name=="raw_generations.jsonl" else lines+lines[:1]
                (directory/name).write_text("\n".join(lines)+"\n")
                completion=json.loads((directory/"completion.json").read_text())
                completion["files_sha256"][name]=sha256_file(directory/name)
                (directory/"completion.json").write_text(json.dumps(completion))
                with self.assertRaises(ValueError):
                    check(directory)


if __name__ == "__main__":
    unittest.main()
