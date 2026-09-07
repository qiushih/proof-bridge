"""Check audit accounting and enforce development-only, generation-free access."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

from baseline.dataset import ROOT, digest
from prompt_v2.experiment import Inference
from scripts.audit_prompt_v2_development import OUTPUT, build_analysis, markdown


class PromptV2AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analysis = build_analysis()

    def test_audit_cannot_read_diagnostic_records_or_generate(self):
        original_open = Path.open

        def guarded_open(path, *args, **kwargs):
            relative = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)
            blocked = ("data/evaluation/", "results/baseline-v1/", "results/baseline-v2/")
            if any(relative.startswith(prefix) for prefix in blocked):
                raise AssertionError(f"Audit accessed diagnostic data: {relative}")
            mode = args[0] if args else kwargs.get("mode", "r")
            if any(flag in mode for flag in "wax+"):
                raise AssertionError(f"Analysis attempted to write: {relative}")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "open", guarded_open), \
             patch.object(Inference, "__init__", side_effect=AssertionError("No inference")), \
             patch("prompt_v2.experiment.run_phase", side_effect=AssertionError("No new run")), \
             patch("baseline.dataset.load_evaluation", side_effect=AssertionError("No evaluation")), \
             patch("baseline.dataset.check_evaluation_evidence", side_effect=AssertionError("No evaluation")):
            self.assertEqual(build_analysis(), self.analysis)

    def test_reports_reproduce_exactly(self):
        self.assertEqual(json.loads((OUTPUT / "failure_analysis.json").read_text()), self.analysis)
        self.assertEqual((OUTPUT / "FAILURE_ANALYSIS.md").read_text(), markdown(self.analysis))

    def test_all_outputs_have_bound_manual_reviews(self):
        rows = self.analysis["records"]
        source = {r["key"]: r for r in (json.loads(line) for line in (ROOT / "results/prompt-v2-development/results.jsonl").read_text().splitlines())}
        self.assertEqual(len(rows), 180)
        self.assertEqual({r["key"] for r in rows}, set(source))
        for row in rows:
            with self.subTest(key=row["key"]):
                self.assertEqual(row["source_row_sha256"], digest(json.dumps(source[row["key"]], sort_keys=True)))
                self.assertEqual(row["proof_body_sha256"], digest(row["proof_body"]))
                self.assertFalse(row["manual_fidelity"]["human_reviewed"])
                self.assertTrue(row["manual_fidelity"]["rationale"])

    def test_failure_categories_are_exclusive_and_denominators_explicit(self):
        summaries = [self.analysis[name] for name in ("selected_prompt_summary", "selected_common_subset_summary", "all_candidates_summary")]
        summaries += list(self.analysis["candidate_summaries"].values())
        summaries += list(self.analysis["selected_prompt_by_condition"].values())
        for summary in summaries:
            self.assertEqual(sum(c["count"] for c in summary["primary_categories"].values()), summary["audit_failures"])
            self.assertEqual(sum(summary["manual_fidelity"].values()), summary["outputs"])
        selected = self.analysis["selected_prompt_summary"]
        self.assertEqual(selected["outputs"], 60)
        self.assertEqual(selected["verification_failures"], 32)
        self.assertEqual({k: v["count"] for k, v in selected["primary_categories"].items()}, {
            "formatting_parser_failure": 2, "missing_introductions": 7,
            "invalid_hypothesis_reference": 14, "mathematically_incorrect_proof_step": 9,
            "forbidden_construct": 0, "verified_but_argument_unfaithful": 0, "other": 0,
        })

    def test_equivalent_arguments_are_separate_from_different_attempted_strategies(self):
        disagreements = self.analysis["fidelity_disagreement_notes"]
        self.assertEqual(len(disagreements), 15)
        self.assertTrue(all(d["manual_status"] == "FAITHFUL" for d in disagreements))
        rows = {r["key"]: r for r in self.analysis["records"]}
        equivalent = rows["short_3shot:seed_017:theorem_and_informal"]
        self.assertEqual(equivalent["manual_fidelity"]["status"], "FAITHFUL")
        different = rows["short_3shot:seed_004:theorem_only"]
        self.assertEqual(different["manual_fidelity"]["strategy_relation"], "different_strategy")
        self.assertEqual(different["verification"]["status"], "FAIL")
        self.assertIn("direction", rows["short_3shot:seed_012:theorem_and_informal"]["manual_fidelity"]["rationale"])

    def test_no_proof_B_observation_is_invented(self):
        dependence = self.analysis["informal_proof_dependence"]
        self.assertEqual(dependence["saved_target_argument_inventory"]["theorems_with_A_and_B"], 0)
        selected = dependence["selected_prompt"]
        self.assertEqual(selected["three_way_test"]["status"], "NOT_RUN_NO_SAVED_B")
        self.assertEqual(selected["three_way_test"]["complete_triplets"], 0)
        self.assertTrue(all(p["proof_B_key"] is None for p in selected["per_example"]))
        self.assertEqual(selected["body_changed"], 20)
        self.assertEqual(selected["induction_added_with_A"], 13)
        self.assertEqual(selected["verification_transitions"], {"PASS->PASS": 10, "FAIL->FAIL": 12, "FAIL->PASS": 6, "PASS->FAIL": 2})

    def test_exactly_one_future_experiment_and_no_new_work_claimed(self):
        decision = self.analysis["next_experiment"]
        self.assertEqual(decision["number_of_recommended_experiments"], 1)
        self.assertEqual(decision["experiment"], "binder_and_scope_aware_constrained_decoding")
        self.assertEqual(decision["status"], "RECOMMENDED_NOT_RUN")
        self.assertEqual(self.analysis["scope"]["new_generations"], 0)
        self.assertEqual(self.analysis["scope"]["diagnostic_evaluation_records_read"], 0)


if __name__ == "__main__":
    unittest.main()
