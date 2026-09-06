import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_seeds import (
    PAIRS, REPORT, SOURCE, assert_artifacts_current, load_curated, make_record, statement_key,
)
from verifier import sha256_file, verify


class SeedDatasetTests(unittest.TestCase):
    def test_exactly_30_distinct_curated_statements(self):
        seeds = load_curated()
        self.assertEqual(len(seeds), 30)
        self.assertEqual(len({statement_key(seed["formal_statement"]) for seed in seeds}), 30)
        self.assertEqual({make_record(seed)["split"] for seed in seeds}, {"unassigned"})
        self.assertTrue(all(not make_record(seed)["provenance"]["human_reviewed"] for seed in seeds))

    def test_all_30_proofs_pass_actual_rocq(self):
        for seed in load_curated():
            with self.subTest(seed=seed["id"]):
                record = make_record(seed)
                result = verify(record["input"]["formal_statement"], record["target"]["proof_body"])
                self.assertEqual(result.status, "PASS", result)
                self.assertTrue(result.kernel_checked)
                self.assertTrue(result.assumptions_checked)
                self.assertEqual(result.source_sha256, hashlib.sha256(record["target"]["rocq_source"].encode()).hexdigest())

    def test_alignment_covers_every_formal_proof_line(self):
        for seed in load_curated():
            with self.subTest(seed=seed["id"]):
                record = make_record(seed)
                lines = record["target"]["proof_body"].splitlines()
                covered = []
                for annotation, authored in zip(record["alignment"], seed["steps"]):
                    first, last = annotation["proof_line_start"], annotation["proof_line_end"]
                    covered.extend(range(first, last + 1))
                    self.assertEqual("\n".join(lines[first - 1:last]), authored["code"])
                    self.assertEqual(annotation["informal_text"], authored["text"])
                self.assertEqual(covered, list(range(1, len(lines) + 1)))

    def test_saved_evidence_matches_sources(self):
        assert_artifacts_current([make_record(seed) for seed in load_curated()])

    def test_binder_renamed_duplicate_rejected(self):
        seeds = json.loads(SOURCE.read_text())
        seeds[1]["formal_statement"] = "forall other : nat, other = other"
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "curated.json"
            filename.write_text(json.dumps(seeds))
            with self.assertRaisesRegex(ValueError, "duplicate statement"):
                load_curated(filename)

    def test_wrong_induction_variable_annotation_rejected(self):
        seeds = json.loads(SOURCE.read_text())
        seeds[15]["induction_variable"] = "m"
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "curated.json"
            filename.write_text(json.dumps(seeds))
            with self.assertRaisesRegex(ValueError, "induction variable"):
                load_curated(filename)

    def test_tampered_proof_cannot_reuse_a_pass_record(self):
        self.check_tampered_artifact("proof")

    def test_stale_source_hash_cannot_reuse_a_pass_record(self):
        self.check_tampered_artifact("evidence")

    def check_tampered_artifact(self, kind):
        stored = [json.loads(line) for line in PAIRS.read_text().splitlines()]
        report = copy.deepcopy(json.loads(REPORT.read_text()))
        if kind == "proof":
            stored[0]["target"]["proof_body"] = "admit."
        else:
            stored[0]["verification"]["source_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            pairs = Path(temporary) / "pairs.jsonl"
            manifest = Path(temporary) / "report.json"
            pairs.write_text("".join(json.dumps(record) + "\n" for record in stored))
            # Even updating the file checksum cannot hide changed proof/evidence.
            report["pairs_sha256"] = sha256_file(pairs)
            manifest.write_text(json.dumps(report))
            with self.assertRaises(ValueError):
                assert_artifacts_current([make_record(seed) for seed in load_curated()], pairs, manifest)


if __name__ == "__main__":
    unittest.main()
