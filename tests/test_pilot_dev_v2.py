"""Dev-extension integrity checks; no model, holdout or diagnostic access."""

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from pilot_dev_v2 import release
from pilot_v1.protocol import validate_row
from training_protocol_v1.scoring import selection_score, step_adherence


class DevExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = Path(os.environ.get("PROOFBRIDGE_DEV_V2_TEST_DATA", str(release.DATA)))

    def test_release_validates_and_keeps_old_bytes(self):
        outcome = release.check(self.directory)
        self.assertEqual(outcome["dev_rows"], 8)
        self.assertEqual(outcome["new_rows"], 2)
        self.assertTrue(outcome["original_six_byte_preserved"])

    def test_frozen_data_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "release"
            shutil.copytree(self.directory, target)
            file = target / "dev.jsonl"
            file.write_bytes(file.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "Frozen artifact changed"):
                release.check(target)

    def test_manifest_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "release"
            shutil.copytree(self.directory, target)
            file = target / "dev_manifest.json"
            data = json.loads(file.read_text())
            data["dev_example_ids"] = data["dev_example_ids"][:-1]
            file.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Manifest seal mismatch"):
                release.check(target)

    def test_new_family_cannot_be_relabelled_into_training(self):
        row = copy.deepcopy(release.read_rows(self.directory / "additions.jsonl")[0])
        row["split"] = "train"
        with self.assertRaisesRegex(ValueError, "Family/split mismatch"):
            validate_row(row)

    def test_equivalent_local_use_and_opposite_step_contracts(self):
        a, b = release.read_rows(self.directory / "additions.jsonl")
        self.assertTrue(step_adherence(b, b["proof_body"].replace("exact IH.", "apply IH."), "argument_B")["matched"])
        self.assertFalse(step_adherence(a, b["proof_body"], "argument_A")["matched"])
        self.assertFalse(step_adherence(b, a["proof_body"], "argument_B")["matched"])

    def test_old_selection_rule_is_not_silently_repurposed(self):
        candidates = [{"condition": "argument_A", "dev_example_id": i}
                      for i in [f"pd0{n}_{a}" for n in (1, 2, 3, 4) for a in ("A", "B")]]
        with self.assertRaisesRegex(ValueError, "exactly the six frozen"):
            selection_score(candidates, 0)

    def test_check_does_not_open_holdout_diagnostics_or_weights(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            name = str(path.resolve())
            if any(part in name for part in ("/reserved/", "/data/evaluation/", "/results/holdout-v1/")) or name.endswith(".safetensors"):
                raise AssertionError(f"Prohibited content read: {name}")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            self.assertEqual(release.check(self.directory)["status"], "PASS")

    def test_prepare_refuses_an_existing_release(self):
        with self.assertRaisesRegex(ValueError, "already exists"):
            release.prepare(self.directory)


if __name__ == "__main__":
    unittest.main()
