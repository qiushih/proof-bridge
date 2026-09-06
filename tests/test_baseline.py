import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline.dataset import (
    ROOT, check_development, check_evaluation_evidence, digest, leakage_audit,
    load_evaluation,
)
from baseline.protocol import assess_structure, build_messages, evaluate_candidate, extract_body, proof_signature
from baseline.report import load_run, validate_records
from baseline.run import run
from verifier import sha256_file, verify


RUN_DIR = ROOT / "results/baseline-v1"


class BaselineProtocolTests(unittest.TestCase):
    def test_development_snapshot_changes_only_split(self):
        snapshot = check_development()
        original = [json.loads(line) for line in (ROOT / "data/seeds/pairs.jsonl").read_text().splitlines()]
        self.assertEqual(len(snapshot), 30)
        for frozen, source in zip(snapshot, original):
            source["split"] = "development"
            self.assertEqual(frozen, source)

    def test_frozen_input_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary) / "data.json"
            manifest = Path(temporary) / "manifest.json"
            data.write_text("original")
            manifest.write_text(json.dumps({"files_sha256": {str(data): sha256_file(data)}}))
            data.write_text("changed")
            with patch("baseline.dataset.DEV_MANIFEST", manifest):
                with self.assertRaisesRegex(ValueError, "Frozen development input changed"):
                    check_development()

    def test_all_12_evaluation_references_pass_actual_rocq(self):
        examples = check_evaluation_evidence()
        self.assertEqual(len(examples), 12)
        for example in examples:
            with self.subTest(example=example["id"]):
                result = verify(example["formal_statement"], example["proof_body"])
                self.assertEqual(result.status, "PASS", result)
                self.assertTrue(result.kernel_checked and result.assumptions_checked)

    def test_evaluation_is_statement_disjoint_but_honestly_reports_family_overlap(self):
        result = leakage_audit(check_development(), load_evaluation())
        self.assertEqual(result["definitional_duplicates"], 0)
        self.assertEqual(result["exact_or_alpha_duplicates"], 0)
        self.assertFalse(result["family_disjoint"])
        self.assertEqual(len(result["overlapping_generation_families"]), 6)

    def test_definitional_development_copy_is_rejected(self):
        examples = load_evaluation()
        examples[0]["formal_statement"] = "forall z : nat, (0 + (0 + z)) = z"
        with self.assertRaisesRegex(ValueError, "Duplicate/equivalent"):
            leakage_audit(check_development(), examples)

    def test_only_informal_argument_differs_between_prompt_conditions(self):
        example = load_evaluation()[10]
        with_proof = build_messages(example, "theorem_and_informal")
        theorem_only = build_messages(example, "theorem_only")
        self.assertEqual(with_proof[0], theorem_only[0])
        self.assertEqual(with_proof[1]["content"], theorem_only[1]["content"] + "\n\nInformal proof:\n" + example["informal_proof"])
        changed = copy.deepcopy(example)
        changed.update(proof_body="SECRET GOLD CODE", steps=[{"text": "SECRET", "code": "SECRET"}], title="SECRET TITLE", argument_features={"secret": True})
        self.assertEqual(build_messages(changed, "theorem_only"), theorem_only)
        self.assertEqual(build_messages(changed, "theorem_and_informal"), with_proof)

    def test_extraction_removes_only_one_whole_outer_fence(self):
        self.assertEqual(extract_body(" \n```coq\nintros n.\nreflexivity.\n```\n"),
                         ("intros n.\nreflexivity.", "outer_code_fence_removed"))
        for text in ["Explanation\n```coq\nreflexivity.\n```", "```coq\nreflexivity.\n```\nExtra text", "Proof.\nreflexivity.\nQed."]:
            self.assertEqual(extract_body(text), (text, "whitespace_only"))

    def test_generated_admission_is_not_removed_or_repaired(self):
        example = load_evaluation()[0]
        result = evaluate_candidate(example, "```coq\nAdmitted.\n```")
        self.assertEqual(result["proof_body"], "Admitted.")
        self.assertEqual(result["verification"]["category"], "FORBIDDEN_COMMAND")
        self.assertEqual(result["argument_fidelity"]["status"], "UNASSESSABLE")

    def test_valid_alternative_can_fail_argument_fidelity(self):
        example = load_evaluation()[9]
        alternative = "intros n m H.\nrewrite H.\nreflexivity."
        result = evaluate_candidate(example, alternative)
        self.assertEqual(result["verification"]["status"], "PASS")
        self.assertEqual(result["argument_fidelity"]["status"], "STRUCTURE_MISMATCH")
        self.assertFalse(result["argument_fidelity"]["criteria"]["uses_induction"])

    def test_induction_tracks_binder_position_despite_local_renaming(self):
        example = load_evaluation()[10]
        renamed = "intros x y z E.\ninduction y as [| q J].\n- simpl.\n  exact E.\n- simpl.\n  rewrite J.\n  reflexivity."
        signature = proof_signature(example["formal_statement"], renamed)
        self.assertEqual(signature["induction_binder_index"], 1)
        self.assertEqual(assess_structure(example, renamed)["status"], "STRUCTURE_MATCH")
        self.assertEqual(assess_structure(example, renamed.replace("induction y", "induction x"))["status"], "STRUCTURE_MISMATCH")

    def test_existing_output_cannot_trigger_another_generation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "refusing to overwrite or repeat"):
                run(Path(temporary))


@unittest.skipUnless((RUN_DIR / "completion.json").exists(), "Actual baseline run has not completed yet")
class BaselineResultTests(unittest.TestCase):
    def test_recorded_baseline_has_exactly_24_single_attempts(self):
        _, records = load_run(RUN_DIR, reverify=True)
        self.assertEqual(len(records), 24)

    def test_repeated_attempt_or_missing_condition_is_rejected(self):
        manifest, records = load_run(RUN_DIR)
        journal = [json.loads(line) for line in (RUN_DIR / "attempts.jsonl").read_text().splitlines()]
        examples = check_evaluation_evidence()
        with self.assertRaisesRegex(ValueError, "generation records"):
            validate_records(records[:-1], journal, manifest, examples)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            validate_records(records, journal + [journal[0]], manifest, examples)

    def test_generated_proof_cannot_be_replaced_after_generation(self):
        manifest, records = load_run(RUN_DIR)
        journal = [json.loads(line) for line in (RUN_DIR / "attempts.jsonl").read_text().splitlines()]
        records[0]["proof_body"] = "intros n m. reflexivity."
        records[0]["proof_body_sha256"] = digest(records[0]["proof_body"])
        with self.assertRaisesRegex(ValueError, "edited beyond"):
            validate_records(records, journal, manifest, check_evaluation_evidence())


if __name__ == "__main__":
    unittest.main()
