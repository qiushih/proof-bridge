import copy
import unittest
from unittest.mock import patch

from argument_switch_v1.protocol import CONDITIONS, assess_strategy, load_variants, messages, strategy_match, summarize
from constrained_v1.grammar import accepts
from prompt_v2.experiment import Inference, load_frozen


class ArgumentSwitchTests(unittest.TestCase):
    def test_six_existing_common_subset_theorems_and_12_allowed_references(self):
        variants = load_variants()
        self.assertEqual(len(variants), 6)
        for variant in variants:
            for key, reference in variant["arguments"].items():
                with self.subTest(seed=variant["example_id"], argument=key):
                    self.assertTrue(accepts(variant["formal_statement"], reference["proof_body"]))
                    self.assertTrue(strategy_match(variant["formal_statement"], reference["proof_body"], reference["requirements"])["matches"])

    def test_frozen_prefix_and_formatter_no_reference_code_leak(self):
        _, selected = load_frozen()
        for variant in load_variants():
            for condition in CONDITIONS:
                actual = messages(variant, condition)
                self.assertEqual(actual[:-1], selected["prefix_messages"])
                target = "Theorem: " + variant["formal_statement"]
                if condition != "theorem_only":
                    target += "\nInformal proof: " + variant["arguments"][condition[-1]]["informal_proof"]
                self.assertEqual(actual[-1]["content"], target + "\nOutput code only.")

    def test_verified_proof_ignoring_B_strategy_is_a_switch_failure(self):
        rows = self.reference_rows(ignore_B=True)
        summary = summarize(rows)
        self.assertEqual(summary["all_outputs"]["verified"], 18)
        self.assertEqual(summary["switching"]["successful_theorems"], 0)
        self.assertEqual(summary["by_condition"]["argument_B"]["verified_strategy_followed"], 0)

    def test_matching_strategy_without_verification_does_not_count_as_success(self):
        rows = self.reference_rows()
        for row in rows:
            if row["condition"] == "argument_B":
                row["verification"]["status"] = "FAIL"
                row["strategy"]["verified_requested_strategy"] = False
        summary = summarize(rows)
        self.assertEqual(summary["switching"]["attempted_switches"], 6)
        self.assertEqual(summary["switching"]["successful_theorems"], 0)

    def test_correct_references_switch_with_all_six_in_denominator(self):
        summary = summarize(self.reference_rows())
        self.assertEqual(summary["switching"]["successful_theorems"], 6)
        self.assertEqual(summary["switching"]["switching_accuracy"], 1)
        self.assertIsNone(summary["by_condition"]["theorem_only"]["attempted_strategy_accuracy"])

    def test_third_binder_induction_and_required_IH_are_not_just_keyword_matches(self):
        variant = load_variants()[-1]
        b = variant["arguments"]["B"]
        wrong_binder = b["proof_body"].replace("induction p", "induction n")
        self.assertFalse(strategy_match(variant["formal_statement"], wrong_binder, b["requirements"])["matches"])
        no_IH = b["proof_body"].replace("f_equal.\n  exact IH.", "reflexivity.")
        self.assertFalse(strategy_match(variant["formal_statement"], no_IH, b["requirements"])["matches"])

    def test_equivalent_premise_tactics_are_accepted(self):
        variant = load_variants()[4]
        body = "intros n m H.\nrewrite H.\nreflexivity."
        self.assertTrue(strategy_match(variant["formal_statement"], body, variant["arguments"]["A"]["requirements"])["matches"])

    def test_no_inference_or_diagnostic_load_during_protocol_checks(self):
        with patch.object(Inference, "__init__", side_effect=AssertionError("No inference")), \
             patch("baseline.dataset.load_evaluation", side_effect=AssertionError("No diagnostic")):
            self.reference_rows()

    def test_missing_or_duplicate_output_cannot_improve_denominator(self):
        rows = self.reference_rows()
        with self.assertRaises(ValueError):
            summarize(rows[:-1])
        rows[-1] = copy.deepcopy(rows[-2])
        with self.assertRaises(ValueError):
            summarize(rows)

    @staticmethod
    def reference_rows(ignore_B=False):
        rows = []
        for variant in load_variants():
            for condition in CONDITIONS:
                label = "B" if condition == "argument_B" and not ignore_B else "A"
                body = variant["arguments"][label]["proof_body"]
                rows.append({"key": variant["example_id"] + ":" + condition, "example_id": variant["example_id"],
                             "condition": condition, "proof_body": body, "verification": {"status": "PASS"},
                             "strategy": assess_strategy(variant, condition, body, True)})
        return rows


if __name__ == "__main__":
    unittest.main()
