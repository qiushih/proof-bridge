from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline.dataset import ROOT
from constrained_v1.experiment import OUTPUT, PREFLIGHT, check
from constrained_v1.grammar import feed, initial
from constrained_v1.report import build_report, markdown
from constrained_v1.tokens import Vocabulary
from prompt_v2.experiment import Inference
from verifier import sha256_file


class TinyTokenizer:
    eos_token_id = 0
    all_special_ids = [0]
    pieces = ["", "intros n.\n", "induction n as [| k IH].\n- ", "rewrite IH.",
              "simpl.\nreflexivity.\n- rewrite IH.\nreflexivity.", "rewrite H.", "intros n H."]

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids, **kwargs):
        return "".join(self.pieces[i] for i in ids)


class TokenBoundaryTests(unittest.TestCase):
    def test_tokens_spanning_commands_and_branches_preserve_scope(self):
        vocabulary = Vocabulary(TinyTokenizer())
        start = initial("forall n : nat, n + 0 = n")
        self.assertIn(1, vocabulary.allowed(start))
        self.assertNotIn(6, vocabulary.allowed(start))
        intro = feed(start, TinyTokenizer.pieces[1])
        self.assertNotIn(5, vocabulary.allowed(intro))
        self.assertIn(2, vocabulary.allowed(intro))
        base = feed(intro, TinyTokenizer.pieces[2])
        self.assertNotIn(3, vocabulary.allowed(base))
        self.assertIn(4, vocabulary.allowed(base))
        self.assertIn(0, vocabulary.allowed(feed(base, TinyTokenizer.pieces[4])))


@unittest.skipUnless((OUTPUT / "completion.json").exists(), "Experiment has not completed")
class ConstrainedResultTests(unittest.TestCase):
    def test_recorded_budget_and_unchanged_inputs(self):
        manifest, rows = check()
        self.assertEqual(len(rows), 60)
        self.assertEqual(manifest["expected_attempts"], 60)
        self.assertEqual(manifest["generation_settings"]["max_new_tokens"], 256)
        self.assertEqual(Counter(r["condition"] for r in rows), {"theorem_and_informal": 30, "theorem_only": 30})
        self.assertEqual(sum(r["selection_eligible"] for r in rows), 50)
        self.assertFalse(any(r["generation_error"] for r in rows))
        self.assertTrue(all(r["constraint"]["complete_body"] for r in rows))
        self.assertTrue(all(r["constraint"]["mask_calls"] == len(r["completion_token_ids"]) for r in rows))

    def test_preflight_precedes_inference_and_all_references_passed(self):
        preflight = json.loads(PREFLIGHT.read_text())
        manifest, _ = check()
        self.assertLess(preflight["completed_at_utc"], manifest["started_at_utc"])
        self.assertEqual(len(preflight["reference_results"]), 30)
        self.assertEqual(preflight["inference_attempts"], 0)
        self.assertTrue(all(r["all_tokens_allowed"] and r["EOS_allowed"] and r["verification"]["status"] == "PASS" for r in preflight["reference_results"]))

    def test_reports_reproduce_and_measure_real_correctness_gains(self):
        summary, assessed = build_report()
        self.assertEqual(summary, json.loads((OUTPUT / "summary.json").read_text()))
        self.assertEqual(markdown(summary), (OUTPUT / "REPORT.md").read_text())
        self.assertEqual(len(assessed), 60)
        self.assertEqual(summary["success"]["baseline"], 21)
        self.assertEqual(summary["success"]["constrained"], 34)
        self.assertTrue(summary["success"]["met"])
        for subset in summary["comparison"].values():
            for scores in subset["constrained_v1"].values():
                self.assertEqual(scores["invalid_reference_failures"], 0)
                self.assertEqual(scores["missing_introduction_failures"], 0)
                self.assertEqual(scores["format_compliant"], scores["outputs"])

    def test_comparison_does_not_access_diagnostics_or_inference(self):
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            relative = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)
            if any(relative.startswith(prefix) for prefix in ("data/evaluation/", "results/baseline-v1/", "results/baseline-v2/")):
                raise AssertionError("Diagnostic data used by constrained experiment")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), patch.object(Inference, "__init__", side_effect=AssertionError("New inference")), \
             patch("baseline.dataset.load_evaluation", side_effect=AssertionError("Evaluation loader")):
            build_report()

    def test_missing_output_or_duplicate_attempt_is_detected(self):
        for name in ("results.jsonl", "attempts.jsonl"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                for file in ("manifest.json", "results.jsonl", "attempts.jsonl", "completion.json"):
                    (directory / file).write_bytes((OUTPUT / file).read_bytes())
                lines = (directory / name).read_text().splitlines()
                lines = lines[:-1] if name == "results.jsonl" else lines + lines[:1]
                (directory / name).write_text("\n".join(lines) + "\n")
                completion = json.loads((directory / "completion.json").read_text())
                completion["files_sha256"][name] = sha256_file(directory / name)
                (directory / "completion.json").write_text(json.dumps(completion))
                with self.assertRaises(ValueError):
                    check(directory)

    def test_manual_reviews_distinguish_valid_renaming_from_bad_reasoning(self):
        _, assessed = build_report()
        by_key = {r["key"]: r for r in assessed}
        renamed = by_key["constrained_v1:seed_014:theorem_and_informal"]
        self.assertEqual(renamed["verification"]["category"], "PROOF_ERROR")
        self.assertIn("actual theorem premise", renamed["manual_fidelity"]["rationale"])
        equivalent = by_key["constrained_v1:seed_021:theorem_and_informal"]
        self.assertEqual(equivalent["argument_fidelity"]["status"], "STRUCTURE_MISMATCH")
        self.assertEqual(equivalent["manual_fidelity"]["status"], "FAITHFUL")


if __name__ == "__main__":
    unittest.main()
