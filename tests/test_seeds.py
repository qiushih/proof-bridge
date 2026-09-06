import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.verify_seeds import (
    PAIRS, REPORT, SOURCE, assert_artifacts_current, load_curated, make_record, statement_key,
)
from scripts.seed_schema import (
    FAMILIES, argument_features, audit_families, authored_body, canonical_body,
    definitional_key, steps_sha256,
)
from verifier import sha256_file, verify


class SeedDatasetTests(unittest.TestCase):
    def test_exactly_30_distinct_curated_statements(self):
        seeds = load_curated()
        self.assertEqual(len(seeds), 30)
        self.assertEqual(len({statement_key(seed["formal_statement"]) for seed in seeds}), 30)
        self.assertEqual({make_record(seed)["split"] for seed in seeds}, {"unassigned"})
        self.assertTrue(all(not make_record(seed)["metadata"]["provenance"]["human_reviewed"] for seed in seeds))

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
                authored_lines = authored_body(seed).splitlines()
                replacements = {edit["proof_line"]: edit for edit in seed["metadata"]["proof_normalizations"]}
                covered = []
                for annotation, authored in zip(record["alignment"], seed["steps"]):
                    first, last = annotation["proof_line_start"], annotation["proof_line_end"]
                    covered.extend(range(first, last + 1))
                    self.assertEqual("\n".join(authored_lines[first - 1:last]), authored["code"])
                    for number in range(first, last + 1):
                        expected = replacements.get(number, {}).get("to", authored_lines[number - 1])
                        self.assertEqual(lines[number - 1], expected)
                    self.assertEqual(annotation["informal_text"], authored["text"])
                self.assertEqual(covered, list(range(1, len(lines) + 1)))
                self.assertEqual(record["steps"], seed["steps"])
                self.assertEqual(seed["metadata"]["authored_steps_sha256"], steps_sha256(seed))

    def test_canonical_fields_and_compatibility_aliases_agree(self):
        for seed in load_curated():
            with self.subTest(seed=seed["id"]):
                record = make_record(seed)
                self.assertEqual(seed["informal_proof"], " ".join(s["text"] for s in seed["steps"]))
                self.assertEqual(seed["proof_body"], canonical_body(seed))
                self.assertEqual(record["input"]["normalized_statement"], seed["informal_statement"])
                self.assertEqual(record["input"]["normalized_proof"], seed["informal_proof"])
                self.assertEqual(record["input"]["formal_statement"], seed["formal_statement"])
                self.assertEqual(record["target"]["proof_body"], seed["proof_body"])
                self.assertEqual(record["input"]["raw_text"], seed["informal_statement"] + "\n\n" + seed["informal_proof"])

    def test_normalizations_preserve_the_three_authored_proofs(self):
        changed = [seed for seed in load_curated() if authored_body(seed) != seed["proof_body"]]
        self.assertEqual([seed["id"] for seed in changed], ["seed_009", "seed_022", "seed_027"])
        for seed in changed:
            with self.subTest(seed=seed["id"]):
                self.assertEqual(len(seed["metadata"]["proof_normalizations"]), 1)
                self.assertIn("apply ", authored_body(seed))
                self.assertNotIn("apply ", seed["proof_body"])
                result = verify(seed["formal_statement"], authored_body(seed))
                self.assertEqual(result.status, "PASS", result)
                self.assertTrue(result.kernel_checked and result.assumptions_checked)

    def test_argument_features_distinguish_premise_from_induction_hypothesis(self):
        features = {seed["id"]: seed["argument_features"] for seed in load_curated()}
        self.assertEqual(sum(f["uses_induction"] for f in features.values()), 15)
        self.assertEqual(sum(f["uses_premise"] for f in features.values()), 12)
        self.assertEqual(features["seed_019"]["induction_variable"], "a")
        self.assertEqual(features["seed_009"], {
            "uses_induction": False, "induction_variable": None, "uses_premise": True,
            "uses_rewrite": False, "uses_f_equal": True, "uses_reflexivity": False,
        })
        self.assertFalse(features["seed_017"]["uses_premise"])
        self.assertTrue(features["seed_027"]["uses_premise"])
        self.assertFalse(features["seed_027"]["uses_reflexivity"])
        self.assertTrue(features["seed_028"]["uses_rewrite"])
        # Introducing an unused premise does not count as using it.
        self.assertFalse(argument_features("forall n : nat, n = n -> n = n", "intros n H. reflexivity.")["uses_premise"])

    def test_schema_rejects_stale_derived_fields_and_provenance(self):
        mutations = [
            (lambda s: s.update(informal_proof="An unrelated argument."), "informal_proof"),
            (lambda s: s.update(proof_body="intros n."), "proof_body"),
            (lambda s: s["argument_features"].update(uses_premise=True), "argument_features"),
            (lambda s: s["argument_features"].update(uses_induction=0), "argument_features"),
            (lambda s: s["argument_features"].update(induction_variable="n"), "argument_features"),
            (lambda s: s["metadata"]["provenance"].update(human_reviewed=True), "provenance"),
            (lambda s: s["metadata"]["provenance"].update(rocq_verified="true"), "provenance"),
            (lambda s: s["metadata"]["provenance"].update(rocq_source_sha256="0" * 64), "provenance hash"),
            (lambda s: s.update(related_existing_example="research-only"), "metadata"),
            (lambda s: s["steps"][0].update(text="Changed original alignment."), "original aligned steps"),
        ]
        for mutate, message in mutations:
            with self.subTest(message=message):
                seeds = json.loads(SOURCE.read_text())
                mutate(seeds[0])
                self.assert_invalid_source(seeds, message)

    def test_normalization_cannot_inject_new_tactics_or_change_reference(self):
        for replacement in ["exact IH.", "admit.", "exact H.\nreflexivity."]:
            with self.subTest(replacement=replacement):
                seeds = json.loads(SOURCE.read_text())
                seeds[8]["metadata"]["proof_normalizations"][0]["to"] = replacement
                self.assert_invalid_source(seeds, "unsupported proof normalization")

    def test_definitional_equivalence_audit(self):
        audit = audit_families(load_curated())
        self.assertEqual(audit["unique_definitional_statements"], 26)
        self.assertEqual(audit["definitional_equivalence_classes"], [
            ["seed_001", "seed_002"], ["seed_005", "seed_006"],
            ["seed_017", "seed_022"], ["seed_018", "seed_026"],
        ])
        self.assertEqual(audit["generation_families"], 7)
        self.assertEqual(audit["split_groups"], 5)
        self.assertEqual(definitional_key("forall n : nat, (1 + n) = S n"),
                         definitional_key("forall x : nat, S (0 + x) = S x"))
        self.assertNotEqual(definitional_key("forall n : nat, n + 0 = n"),
                            definitional_key("forall n : nat, n = n"))

    def test_related_families_cannot_be_silently_reassigned(self):
        for field in ["generation_family", "split_group"]:
            seeds = json.loads(SOURCE.read_text())
            seeds[29][field] = "separate_successor_variant"
            self.assert_invalid_source(seeds, "family manifest")

    def test_definitional_variants_cannot_cross_even_an_edited_family_manifest(self):
        seeds = load_curated()
        seeds[21]["generation_family"] = "renamed_right_one"
        manifest = json.loads(FAMILIES.read_text())
        specification = manifest["families"]["successor_reassociation_variants"]
        specification["members"].remove("seed_022")
        manifest["families"]["renamed_right_one"] = {
            "split_group": specification["split_group"], "members": ["seed_022"],
        }
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "families.json"
            filename.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "Definitionally equivalent"):
                audit_families(seeds, filename)

    def assert_invalid_source(self, seeds, message):
        with tempfile.TemporaryDirectory() as temporary:
            filename = Path(temporary) / "curated.json"
            filename.write_text(json.dumps(seeds))
            with self.assertRaisesRegex(ValueError, message):
                load_curated(filename)

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

    def test_original_proof_evidence_is_required_after_normalization(self):
        self.check_tampered_artifact("original")

    def check_tampered_artifact(self, kind):
        stored = [json.loads(line) for line in PAIRS.read_text().splitlines()]
        report = copy.deepcopy(json.loads(REPORT.read_text()))
        if kind == "proof":
            stored[0]["target"]["proof_body"] = "admit."
        elif kind == "original":
            del stored[8]["verification"]["authored_proof"]
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
