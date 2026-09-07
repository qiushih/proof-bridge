import unittest

from baseline.dataset import check_development
from constrained_v1.grammar import accepts, can_end, feed, initial
from verifier import render_source

N = "forall n : nat, n + 0 = n"
NM = "forall n m : nat, n = m -> S n = S m"
INDUCTION = "intros n.\ninduction n as [| k IH].\n- simpl.\nreflexivity.\n- simpl.\nrewrite IH.\nreflexivity."


class GrammarTests(unittest.TestCase):
    def test_all_30_reference_proofs_allowed_and_unchanged_parser_accepts(self):
        examples = check_development()
        self.assertEqual(len(examples), 30)
        for example in examples:
            with self.subTest(seed=example["id"]):
                self.assertTrue(accepts(example["formal_statement"], example["proof_body"]))
                render_source(example["formal_statement"], example["proof_body"])

    def test_invented_hypotheses_and_wrong_kind_references_blocked(self):
        for body in ["intros n H.", "intros n.\nrewrite H.", "intros n.\nexact n.",
                     "intros n.\nrewrite IH.", "intros n.\napply Nat.add_0_r."]:
            with self.subTest(body=body):
                self.assertIsNone(feed(initial(N), body))

    def test_all_introductions_required(self):
        self.assertIsNone(feed(initial(NM), "intros n.\ninduction n "))
        self.assertIsNone(feed(initial(NM), "intros n m.\nrewrite H."))
        self.assertTrue(accepts(NM, "intros x.\nintros y.\nintros E.\nrewrite E.\nreflexivity."))

    def test_malformed_or_forbidden_commands_blocked(self):
        bodies = ["```coq", "Theorem foo", "Admitted.", "intros n.\nlia.",
                  "intros n.\nsimpl n.", "intros n.\nrewrite S n.",
                  "intros n.\ninduction n.", "intros n.\nreflexivity; reflexivity.",
                  "intros n.reflexivity.", "intros n.\n- reflexivity."]
        for body in bodies:
            with self.subTest(body=body):
                self.assertIsNone(feed(initial(N), body))

    def test_IH_only_in_successor_branch_and_branch_local_names_unique(self):
        prefix = "intros n.\ninduction n as [| k IH].\n- "
        self.assertIsNone(feed(initial(N), prefix + "rewrite IH."))
        self.assertIsNone(feed(initial(N), prefix + "exact k."))
        self.assertTrue(accepts(N, INDUCTION))
        self.assertIsNone(feed(initial(N), INDUCTION + "\n- reflexivity."))
        for names in ["n IH", "k k", "k n"]:
            self.assertIsNone(feed(initial(N), "intros n.\ninduction n as [| " + names + "]."))

    def test_premise_available_in_each_branch_but_nat_is_not_a_proof(self):
        statement = "forall n m : nat, m + 0 = m -> (n + m) + 0 = n + m"
        body = "intros n m H.\ninduction n as [| k IH].\n- exact H.\n- exact IH."
        self.assertTrue(accepts(statement, body))
        self.assertIsNone(feed(initial(statement), body.replace("exact IH", "exact k")))

    def test_EOS_requires_complete_body_and_two_nonempty_branches(self):
        for body in ["", "intros n.", "intros n.\nreflexivity", INDUCTION.split("- simpl.")[0],
                     "intros n.\ninduction n as [| k IH].\n- reflexivity.\n- "]:
            self.assertFalse(can_end(feed(initial(N), body)))
        self.assertTrue(can_end(feed(initial(N), "intros n.\nreflexivity.")))

    def test_no_mathematical_or_gold_strategy_filter(self):
        # These may fail in Rocq; scope constraints must not choose the proof.
        self.assertTrue(accepts(N, "intros n.\nsimpl.\nreflexivity."))
        self.assertTrue(accepts(NM, "intros n m H.\nrewrite <- H.\nreflexivity."))
        second = "intros n m H.\ninduction m as [| k IH].\n- exact H.\n- exact IH."
        self.assertTrue(accepts(NM, second))

    def test_command_limit_reserves_successor_branch(self):
        statement = "forall n : nat, n = n"
        self.assertTrue(accepts(statement, "intros n.\n" + "simpl.\n" * 38 + "reflexivity."))
        self.assertFalse(accepts(statement, "intros n.\n" + "simpl.\n" * 39 + "reflexivity."))
        body = "intros n.\ninduction n as [| k IH].\n- " + "simpl.\n" * 37
        self.assertIsNone(feed(initial(statement), body + "simpl."))
        self.assertTrue(accepts(statement, body + "- reflexivity."))


if __name__ == "__main__":
    unittest.main()
