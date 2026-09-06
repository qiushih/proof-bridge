import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline.dataset import ROOT, check_development
from prompt_v2.experiment import freeze_selection, run_phase
from prompt_v2.protocol import (
    assess_output, build_messages, format_compliance, load_plan, make_prompt,
    rank_candidates, selection_partition,
)


class PromptV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.development = check_development()
        cls.plan = load_plan()

    def test_v1_verifier_and_all_frozen_data_are_unchanged(self):
        freeze = json.loads((ROOT / "results/baseline-v1/freeze.json").read_text())
        for name, expected in freeze["files_sha256"].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), expected, name)

    def test_two_to_four_demonstrations_are_exact_development_records(self):
        by_id = {e["id"]: e for e in self.development}
        for candidate in self.plan["candidates"]:
            prompt = make_prompt(candidate, self.development)
            self.assertIn(len(prompt["demonstration_ids"]), [2, 3, 4])
            for index, seed_id in enumerate(prompt["demonstration_ids"]):
                self.assertEqual(prompt["prefix_messages"][2 + 2 * index], {"role": "assistant", "content": by_id[seed_id]["proof_body"]})
                self.assertIn(by_id[seed_id]["informal_proof"], prompt["prefix_messages"][1 + 2 * index]["content"])

    def test_unknown_demonstration_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Only frozen development"):
            make_prompt({"id": "bad", "demonstration_ids": ["not_a_seed"]}, self.development)

    def test_target_argument_is_the_only_difference_between_conditions(self):
        target = self.development[18]
        prompt = make_prompt(self.plan["candidates"][1], self.development)
        full = build_messages(prompt, target, "theorem_and_informal")
        theorem_only = build_messages(prompt, target, "theorem_only")
        self.assertEqual(full[:-1], theorem_only[:-1])
        self.assertEqual(full[-1]["content"].replace("\nInformal proof: " + target["informal_proof"], ""), theorem_only[-1]["content"])
        changed = copy.deepcopy(target)
        changed.update(proof_body="SECRET TARGET CODE", steps=[], argument_features={"secret": True}, title="SECRET TITLE")
        self.assertEqual(build_messages(prompt, changed, "theorem_and_informal"), full)
        self.assertEqual(build_messages(prompt, changed, "theorem_only"), theorem_only)

    def test_demo_and_definitional_duplicates_are_excluded_from_selection(self):
        partition = selection_partition(self.plan, self.development)
        self.assertEqual(partition["excluded_demonstration_or_equivalent_ids"], ["seed_001", "seed_002", "seed_010", "seed_016", "seed_027"])
        self.assertEqual(len(partition["selection_ids"]), 25)

    def test_fences_fail_format_even_if_code_verifies(self):
        example = self.development[15]
        result = assess_output(example, "```coq\n" + example["proof_body"] + "\n```")
        self.assertFalse(result["format_compliance"]["compliant"])
        self.assertEqual(result["verification"]["status"], "PASS")
        self.assertEqual(result["argument_fidelity"]["status"], "STRUCTURE_MATCH")

    def test_format_does_not_imply_mathematical_correctness(self):
        result = assess_output(self.development[15], "intros n.\nreflexivity.")
        self.assertTrue(result["format_compliance"]["compliant"])
        self.assertEqual(result["verification"]["category"], "PROOF_ERROR")

    def test_empty_output_and_wrappers_are_not_compliant(self):
        for text in ["", "Proof. intros n. reflexivity. Qed.", "Here is the proof: intros n."]:
            self.assertFalse(format_compliance(self.development[0]["formal_statement"], text)["compliant"])

    def test_verified_alternative_can_diverge_from_the_reference_argument(self):
        result = assess_output(self.development[6], "intros n m H.\nrewrite H.\nreflexivity.")
        self.assertEqual(result["verification"]["status"], "PASS")
        self.assertEqual(result["argument_fidelity"]["status"], "STRUCTURE_MISMATCH")

    def test_evaluation_cannot_be_loaded_before_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("prompt_v2.experiment.FROZEN", Path(temporary) / "missing.json"):
                with patch("baseline.dataset.check_evaluation_evidence", side_effect=AssertionError("Evaluation accessed")) as loader:
                    with self.assertRaisesRegex(ValueError, "freeze Prompt v2"):
                        run_phase("evaluation", Path(temporary) / "run")
                    loader.assert_not_called()

    def test_development_setup_does_not_access_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch("prompt_v2.experiment.FROZEN", Path(temporary) / "absent.json"):
                with patch("baseline.dataset.check_evaluation_evidence", side_effect=AssertionError("Evaluation accessed")) as loader:
                    with patch("prompt_v2.experiment.Inference", side_effect=RuntimeError("Stopped before generation")):
                        with self.assertRaisesRegex(RuntimeError, "Stopped before generation"):
                            run_phase("development", Path(temporary) / "run")
                    loader.assert_not_called()

    def test_existing_freeze_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "frozen.json"
            existing.write_text("preserve")
            with patch("prompt_v2.experiment.FROZEN", existing):
                with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                    freeze_selection()
            self.assertEqual(existing.read_text(), "preserve")

    def test_selection_prioritizes_format_then_verification_then_fidelity(self):
        def row(candidate, index, format_ok, verified, faithful):
            return {"prompt_id": candidate, "example_id": "seed_x", "condition": str(index),
                    "format_compliance": {"compliant": format_ok},
                    "verification": {"status": "PASS" if verified else "FAIL", "stage": "rocq"},
                    "argument_fidelity": {"status": "STRUCTURE_MATCH" if faithful else "STRUCTURE_MISMATCH"},
                    "failure_category": None if verified else "PROOF_ERROR", "generation_error": None,
                    "finish_reason": "eos", "usage": {"prompt_tokens": 1, "completion_tokens": 1}, "generation_latency_seconds": 1}
        rows = [row("short_2shot", i, True, False, False) for i in range(2)]
        rows += [row("short_3shot", i, i == 0, True, True) for i in range(2)]
        rows += [row("short_4shot", i, True, i == 0, False) for i in range(2)]
        ranking = rank_candidates(self.plan, rows, ["seed_x"])
        self.assertEqual([r["candidate_id"] for r in ranking], ["short_4shot", "short_2shot", "short_3shot"])


if __name__ == "__main__":
    unittest.main()
