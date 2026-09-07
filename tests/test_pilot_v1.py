import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from pilot_v1 import dataset
from pilot_v1.protocol import (DATA, MANIFEST, ROOT, audit_rows, equivalence_key, features, load_split,
                               read_rows, training_records, validate_row)
from pilot_v1.report import render_report


class PilotSchemaTests(unittest.TestCase):
    def setUp(self):
        self.rows = read_rows(DATA / "train.jsonl")

    def test_alpha_orientation_and_definitional_equivalence(self):
        a = "forall n m : nat, n = m -> S n + 0 = S (m + 0)"
        b = "forall y x : nat, x = y -> S (x + 0) = S y + 0"
        self.assertEqual(equivalence_key(a), equivalence_key(b))
        self.assertNotEqual(equivalence_key("forall n : nat, n + 0 = n"),
                            equivalence_key("forall n : nat, n + 1 = S n"))

    def test_equivalent_variants_cannot_cross_family_or_split(self):
        a = next(r for r in self.rows if r["id"] == "pt08_A")
        b = copy.deepcopy(next(r for r in self.rows if r["id"] == "pt08_B"))
        b["split"], b["generation_family"] = "dev", "successor_reassociation_variants"
        with self.assertRaisesRegex(ValueError, "Equivalent theorems cross"):
            audit_rows([a, b], verified=False)

    def test_duplicate_ids_and_targets_rejected(self):
        a = self.rows[0]
        with self.assertRaisesRegex(ValueError, "Duplicate example IDs"):
            audit_rows([a, a], verified=False)
        b = copy.deepcopy(a)
        b["id"], b["theorem_id"] = "duplicate", "duplicate"
        with self.assertRaisesRegex(ValueError, "Duplicate equivalent statement/proof target"):
            audit_rows([a, b], verified=False)

    def test_alignment_drift_is_rejected(self):
        for field in ("proof_body", "informal_proof"):
            row = copy.deepcopy(self.rows[0])
            row[field] += " drift"
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "alignment"):
                validate_row(row, verified=False)

    def test_stale_features_and_false_provenance_rejected(self):
        row = copy.deepcopy(self.rows[0])
        row["argument_features"]["uses_induction"] = True
        with self.assertRaisesRegex(ValueError, "features"):
            validate_row(row, verified=False)
        row = copy.deepcopy(self.rows[0])
        row["provenance"]["human_reviewed"] = True
        with self.assertRaisesRegex(ValueError, "provenance"):
            validate_row(row, verified=False)

    def test_base_premise_and_successor_ih_are_separate_features(self):
        a = next(r for r in self.rows if r["id"] == "pt11_A")
        b = next(r for r in self.rows if r["id"] == "pt11_B")
        for r in (a, b):
            self.assertTrue(r["argument_features"]["premise_in_base_case"])
            self.assertTrue(r["argument_features"]["uses_induction"])
        self.assertTrue(a["argument_features"]["ih_via_rewrite"])
        self.assertTrue(b["argument_features"]["ih_via_congruence"])
        self.assertFalse(b["argument_features"]["ih_via_rewrite"])

    def test_rewrite_order_and_direction_contrasts_are_real(self):
        a = next(r for r in self.rows if r["id"] == "pt03_A")
        b = next(r for r in self.rows if r["id"] == "pt03_B")
        self.assertEqual(a["formal_statement"], b["formal_statement"])
        self.assertNotEqual(a["informal_proof"], b["informal_proof"])
        self.assertTrue(a["argument_features"]["rewrite_simpl_order"][0]["simpl_after"])
        self.assertTrue(b["argument_features"]["rewrite_simpl_order"][0]["simpl_before"])
        reverse = next(r for r in self.rows if r["id"] == "pt02_B")
        self.assertEqual(reverse["argument_features"]["rewrite_steps"][0]["direction"], "reverse")


@unittest.skipUnless(MANIFEST.exists(), "Pilot split freeze not created yet")
class PilotReleaseTests(unittest.TestCase):
    def test_complete_frozen_membership_and_historical_preservation(self):
        result = dataset.check()
        self.assertEqual(result["protected_historical_files"], 132)
        manifest = json.loads(MANIFEST.read_text())
        self.assertEqual({s: len(p["example_ids"]) for s,p in manifest["partitions"].items()},
                         {"train": 24, "dev": 6, "holdout": 6})

    def test_export_contains_only_intended_inputs_and_target(self):
        records = training_records()
        self.assertEqual(len(records), 24)
        self.assertTrue(all(set(r["input"]) == {"formal_statement", "informal_proof"} for r in records))
        self.assertTrue(all(set(r["target"]) == {"proof_body"} for r in records))
        self.assertTrue(all(r["id"].startswith("pt") for r in records))

    def test_holdout_loader_is_always_rejected_before_io(self):
        with patch.object(Path, "open", side_effect=AssertionError("No file access expected")):
            for name in ("holdout", "reserved/holdout.jsonl", "../reserved/holdout.jsonl"):
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "reserved"):
                    load_split(name)

    def test_public_workflow_does_not_parse_holdout_or_diagnostics_or_infer(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            sensitive = str(path.resolve()).startswith((str(DATA / "reserved"), str(ROOT / "data/evaluation")))
            if sensitive and "b" not in mode:
                raise AssertionError("Sensitive contents opened as text")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded), patch.object(dataset, "token_evidence", side_effect=AssertionError("No token/model work after seal")):
            dataset.check()
            training_records("train")
            training_records("dev")
            render_report()
            with self.assertRaises(ValueError):
                dataset.prepare()
            with self.assertRaises(ValueError):
                dataset.freeze()

    def test_verification_evidence_bound_to_formal_source(self):
        row = copy.deepcopy(load_split("train")[0])
        row["verification"]["source_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Rocq evidence"):
            validate_row(row)

    def test_reference_quality_and_coverage_evidence(self):
        evidence = json.loads((DATA / "verification_report.json").read_text())
        self.assertEqual(evidence["references_verified"], 36)
        self.assertEqual(evidence["frozen_decoder_token_paths_allowed"], 36)
        self.assertLessEqual(evidence["max_reference_tokens_including_eos"], 256)
        self.assertTrue(evidence["all_removed_reference_proofs_failed_in_rocq"])
        self.assertEqual(evidence["model_generations"], 0)
        counts = json.loads((DATA / "coverage.json").read_text())
        self.assertTrue(all(n > 0 for n in counts["train"]["patterns"].values()))
        review = json.loads((DATA / "historical_reference_review.json").read_text())
        self.assertEqual(review["counts"]["excluded_weak_B"], 5)
        self.assertEqual(review["counts"]["secondary_only_B"], 1)

    def test_fresh_statement_holdout_has_explicit_family_exposure(self):
        audit = json.loads((DATA / "family_audit.json").read_text())
        self.assertEqual(audit["historical_overlap"]["holdout_historical_statement_duplicates"], 0)
        self.assertEqual(audit["historical_overlap"]["historical_development_family_exposure"]["conditional_permutation"], ["seed_029"])
        self.assertEqual(audit["within_pilot"]["cross_split_families"], 0)

    def test_report_reproduces_without_reserved_reference_contents(self):
        self.assertEqual((DATA / "VERIFICATION_REPORT.md").read_text(), render_report())


if __name__ == "__main__":
    unittest.main()
