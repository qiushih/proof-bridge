from datetime import datetime
import json
from pathlib import Path
import tempfile
import unittest

from prompt_v2.experiment import DEV_OUTPUT, EVAL_OUTPUT, load_frozen, load_phase
from prompt_v2.report import reverify
from verifier import sha256_file


@unittest.skipUnless((EVAL_OUTPUT / "completion.json").exists(), "Frozen v2 evaluation has not completed")
class PromptV2ResultTests(unittest.TestCase):
    def test_development_comparison_has_180_attempts_and_fixed_winner(self):
        manifest, rows = load_phase(DEV_OUTPUT)
        freeze, selected = load_frozen()
        self.assertEqual(len(rows), 180)
        self.assertEqual(manifest["phase"], "development")
        self.assertEqual(freeze["evaluation_records_loaded_for_selection"], 0)
        self.assertEqual(selected["id"], freeze["ranking"][0]["candidate_id"])
        self.assertEqual(len(freeze["selection_partition"]["selection_ids"]), 25)

    def test_evaluation_follows_freeze_and_keeps_original_budget(self):
        manifest, rows = load_phase(EVAL_OUTPUT)
        freeze, _ = load_frozen()
        self.assertGreater(datetime.fromisoformat(manifest["started_at_utc"]), datetime.fromisoformat(freeze["frozen_at_utc"]))
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(not r["is_demonstration"] and not r["selection_eligible"] for r in rows))
        self.assertTrue(all(r["attempt"] == 1 and r["repair_attempts"] == 0 for r in rows))
        self.assertEqual(manifest["generation_settings"]["max_new_tokens"], 256)
        self.assertFalse(manifest["generation_settings"]["do_sample"])
        reverify(manifest, rows)

    def test_missing_result_cannot_be_hidden_by_updating_the_checksum(self):
        self.tamper("remove")

    def test_duplicate_attempt_cannot_be_hidden_by_updating_the_checksum(self):
        self.tamper("duplicate")

    def test_generated_candidate_cannot_be_repaired_in_saved_results(self):
        self.tamper("proof")

    def tamper(self, operation):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in ["manifest.json", "results.jsonl", "attempts.jsonl", "summary.json", "completion.json"]:
                (directory / name).write_bytes((EVAL_OUTPUT / name).read_bytes())
            if operation == "duplicate":
                name = "attempts.jsonl"
                path = directory / name
                original = path.read_text()
                path.write_text(original + original.splitlines()[0] + "\n")
            else:
                name = "results.jsonl"
                path = directory / name
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                if operation == "remove":
                    rows.pop()
                else:
                    rows[0]["proof_body"] += "\nreflexivity."
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            completion = json.loads((directory / "completion.json").read_text())
            completion["files_sha256"][name] = sha256_file(directory / name)
            (directory / "completion.json").write_text(json.dumps(completion))
            with self.assertRaises(ValueError):
                load_phase(directory)


if __name__ == "__main__":
    unittest.main()
