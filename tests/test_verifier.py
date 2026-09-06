import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import verifier


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = json.loads((ROOT / "examples/add_zero_right.json").read_text())
STATEMENT = EXAMPLE["input"]["formal_statement"]
PROOF = EXAMPLE["target"]["proof_body"]


class VerifierTests(unittest.TestCase):
    def assert_category(self, body, category, statement=STATEMENT):
        result = verifier.verify(statement, body)
        self.assertEqual(result.status, "FAIL", result)
        self.assertEqual(result.category, category, result)
        self.assertFalse(result.kernel_checked)
        return result

    def test_valid_induction_compiles(self):
        result = verifier.verify(STATEMENT, PROOF)
        self.assertEqual(result.status, "PASS", result)
        self.assertEqual(result.category, "VERIFIED")
        self.assertEqual(result.compiler_version, "9.2.0")
        self.assertTrue(result.kernel_checked)
        self.assertTrue(result.assumptions_checked)
        self.assertIn("Closed under the global context", result.stdout)

    def test_standalone_example_compiles(self):
        # Compile a fresh copy of the actual .v example, not just its JSON body.
        with tempfile.TemporaryDirectory(prefix="proof-bridge-test-") as temporary:
            work = Path(temporary)
            _, compiler = verifier.check_environment(work)
            (work / "Example.v").write_text((ROOT / "examples/add_zero_right.v").read_text())
            process = verifier.run_process([str(compiler), "compile", "-q", "Example.v"], work, 10)
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            self.assertTrue((work / "Example.vo").is_file())
            self.assertIn("Closed under the global context", process.stdout)

    def test_invalid_proof_reaches_rocq(self):
        result = self.assert_category("intros n. reflexivity.", "PROOF_ERROR")
        self.assertEqual(result.stage, "rocq")
        self.assertIn("Error:", result.stderr)

    def test_incomplete_proof_reaches_rocq(self):
        result = self.assert_category("intros n.", "INCOMPLETE_PROOF")
        self.assertEqual(result.stage, "rocq")

    def test_empty_proof_is_incomplete(self):
        self.assert_category("", "INCOMPLETE_PROOF")

    def test_syntax_error_missing_reference(self):
        result = self.assert_category("intros n. rewrite .", "SYNTAX_ERROR")
        self.assertEqual(result.stage, "policy")

    def test_syntax_error_missing_period(self):
        self.assert_category("intros n", "SYNTAX_ERROR")

    def test_admissions_and_declarations_rejected(self):
        for body in ["Admitted.", "admit.", "Abort.", "Axiom cheat : False.", "Parameter cheat : False."]:
            with self.subTest(body=body):
                result = self.assert_category(body, "FORBIDDEN_COMMAND")
                self.assertEqual(result.stage, "policy")

    def test_forbidden_tactics(self):
        for tactic in ["auto", "eauto", "lia", "nia", "ring"]:
            with self.subTest(tactic=tactic):
                self.assert_category(f"intros n. {tactic}.", "FORBIDDEN_TACTIC")

    def test_forbidden_helper_qualified_and_unqualified(self):
        for body in ["intros n. apply Nat.add_0_r.", "intros n. exact plus_n_O.", "intros n. rewrite Nat.add_0_r. reflexivity."]:
            with self.subTest(body=body):
                self.assert_category(body, "FORBIDDEN_HELPER")

    def test_import_and_theorem_modification_rejected(self):
        for injection in ["From Stdlib Require Import Lia.", "Require Import Arith.", "Qed. Theorem other : True.", "Abort. Goal True.", "Unset Guard Checking.", "Ltac escape := auto."]:
            with self.subTest(injection=injection):
                self.assert_category(injection, "FORBIDDEN_COMMAND")

    def test_statement_cannot_inject_commands(self):
        self.assert_category("reflexivity.", "FORBIDDEN_COMMAND", "forall n : nat, n = n. Axiom cheat : False")

    def test_arbitrary_term_and_tactical_rejected(self):
        self.assert_category("intros n. exact (Nat.add_0_r n).", "SYNTAX_ERROR")
        self.assert_category("intros n; reflexivity.", "SYNTAX_ERROR")

    def test_successor_hypothesis_not_visible_in_base_branch(self):
        self.assert_category(
            "intros n. induction n as [| k plus_n_O]. - exact plus_n_O. - simpl. rewrite plus_n_O. reflexivity.",
            "FORBIDDEN_HELPER",
        )

    def test_local_names_are_renamed_to_prevent_global_fallback(self):
        source = verifier.render_source(STATEMENT, PROOF)
        self.assertIn("rewrite pb_local_2.", source)
        self.assertNotIn("rewrite IH.", source)

    def test_comments_are_lexed_and_cannot_smuggle_commands(self):
        result = verifier.verify(STATEMENT, "(* Admitted (* auto *) are forbidden *)\n" + PROOF)
        self.assertEqual(result.status, "PASS", result)
        self.assert_category("ad(* hidden *)mit.", "FORBIDDEN_TACTIC")
        self.assert_category("(* unterminated", "SYNTAX_ERROR")

    def test_equality_premise_and_reverse_rewrite(self):
        result = verifier.verify("forall n m : nat, n = m -> m = n", "intros a b H. rewrite <- H. reflexivity.")
        self.assertEqual(result.status, "PASS", result)

    def test_different_false_target_is_not_replaced(self):
        self.assert_category(PROOF, "PROOF_ERROR", "forall n : nat, n + 0 = S n")

    def test_addition_only_statement_grammar(self):
        self.assert_category(PROOF, "SYNTAX_ERROR", "forall n : nat, n * 0 = 0")

    def test_timeout_kills_real_compiler(self):
        started = time.monotonic()
        result = verifier.verify(STATEMENT, PROOF, timeout=0.000001)
        self.assertEqual(result.category, "TIMEOUT", result)
        self.assertEqual(result.stage, "rocq")
        self.assertLess(time.monotonic() - started, 6)
        self.assertFalse(result.kernel_checked)

    def test_invalid_timeout(self):
        for timeout in [0, -1, float("nan"), float("inf"), True, "10"]:
            with self.subTest(timeout=timeout):
                self.assertEqual(verifier.verify(STATEMENT, PROOF, timeout).category, "INPUT_ERROR")

    def test_input_limit(self):
        self.assert_category(" " * (verifier.MAX_INPUT_CHARS + 1), "LIMIT_EXCEEDED")

    def test_missing_environment_fails_closed(self):
        with patch.object(verifier, "LOCK_FILE", ROOT / "does-not-exist.json"):
            self.assert_category(PROOF, "ENVIRONMENT_ERROR")

    def test_cli_pass_and_fail_exit_codes(self):
        process = subprocess.run([sys.executable, str(ROOT / "verifier.py"), "--example", str(ROOT / "examples/add_zero_right.json")], capture_output=True, text=True, timeout=15)
        self.assertEqual(process.returncode, 0, process.stderr + process.stdout)
        self.assertEqual(json.loads(process.stdout)["status"], "PASS")
        with tempfile.TemporaryDirectory(prefix="proof-bridge-cli-") as temporary:
            candidate = Path(temporary) / "body.txt"
            candidate.write_text("Admitted.")
            process = subprocess.run([sys.executable, str(ROOT / "verifier.py"), "--statement", STATEMENT, "--proof-file", str(candidate)], capture_output=True, text=True, timeout=15)
            self.assertEqual(process.returncode, 1)
            self.assertEqual(json.loads(process.stdout)["category"], "FORBIDDEN_COMMAND")


if __name__ == "__main__":
    unittest.main()
