import copy
from functools import lru_cache
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from pilot_v1.protocol import ROOT, load_split
from prompt_v2.experiment import load_frozen
from training_protocol_v1.data import config, encode_example, epoch_order
from training_protocol_v1.experiment import dev_groups, messages
from training_protocol_v1.scoring import mathematical_structure, selection_score, step_adherence


class ProtocolTests(unittest.TestCase):
    def test_three_theorems_and_exact_prompt_prefix(self):
        _, prompt = load_frozen()
        groups = dev_groups()
        self.assertEqual([g["theorem_id"] for g in groups], ["pd01", "pd02", "pd03"])
        for g in groups:
            for c in ("theorem_only", "argument_A", "argument_B"):
                msg = messages(g, c)
                self.assertEqual(msg[:-1], prompt["prefix_messages"])
                self.assertNotIn("proof_body", msg[-1]["content"])
                self.assertEqual("Informal proof:" in msg[-1]["content"], c != "theorem_only")

    def test_equivalent_ih_tactics_do_not_automatically_diverge_mathematically(self):
        for g in dev_groups():
            a, b = g["arguments"]["A"], g["arguments"]["B"]
            self.assertEqual(mathematical_structure(a["formal_statement"], b["proof_body"])["status"], "COMPATIBLE")
            self.assertFalse(step_adherence(a, b["proof_body"], "argument_A")["matched"])
            self.assertTrue(step_adherence(b, b["proof_body"].replace("exact IH.", "apply IH."), "argument_B")["matched"])

    def test_alpha_renaming_intros_grouping_and_explicit_forward_rewrite(self):
        a = dev_groups()[1]["arguments"]["A"]
        body = a["proof_body"].replace("intros n m p.", "intros n.\nintros m p.").replace("IH", "Hyp").replace("rewrite Hyp.", "rewrite -> Hyp.")
        self.assertTrue(step_adherence(a, body, "argument_A")["matched"])

    def test_controls_are_not_scored_as_argument_following(self):
        a = dev_groups()[0]["arguments"]["A"]
        self.assertIsNone(step_adherence(a, a["proof_body"], "theorem_only")["matched"])

    def test_selection_uses_only_six_dev_rows_and_earliest_tie(self):
        rows = [{"condition":"argument_" + r["argument_id"], "dev_example_id":r["id"], "verification":{"status":"PASS"},
                 "requested_proof_steps":{"matched":True}, "mathematical_argument_fidelity":{"status":"FAITHFUL"}}
                for r in load_split("dev")]
        control = {"condition":"theorem_only", "verification":{"status":"PASS"}}
        self.assertEqual(selection_score(rows + [control], 0), [6,6,6,0])
        self.assertGreater(selection_score(rows, 0), selection_score(rows, 6))
        with self.assertRaises(ValueError):
            selection_score(rows[:-1], 0)

    def test_frozen_budget_and_shuffle_cover_each_training_row_once_per_epoch(self):
        c = config()["training"]
        self.assertEqual(c["epochs"] * c["examples_per_epoch"] // c["effective_batch_size"], 60)
        expected = sorted(r["id"] for r in load_split("train"))
        for epoch in range(10):
            self.assertEqual(sorted(epoch_order(epoch)), expected)
            self.assertEqual(epoch_order(epoch), epoch_order(epoch))
        self.assertNotEqual(epoch_order(0), epoch_order(1))

    def test_no_holdout_or_diagnostic_content_reads(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            mode = args[0] if args else kwargs.get("mode", "r")
            if str(path.resolve()).startswith((str(ROOT/"data/pilot-v1/reserved"),str(ROOT/"data/evaluation"))) and "b" not in mode:
                raise AssertionError("Forbidden sealed/diagnostic content read")
            return original(path, *args, **kwargs)
        with patch.object(Path, "open", guarded):
            dev_groups()
            epoch_order(0)
            messages(dev_groups()[0], "argument_B")


@unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("transformers"), "Use the pinned .venv for tensor/tokenizer preflight")
class LossAndAdapterTests(unittest.TestCase):
    @staticmethod
    @lru_cache(maxsize=1)
    def tokenizer():
        from constrained_v1.experiment import tokenizer_only
        return tokenizer_only()

    def test_all_public_rows_fit_and_only_final_completion_is_supervised(self):
        tokenizer = self.tokenizer()
        _, prompt = load_frozen()
        for split in ("train", "dev"):
            for row in load_split(split):
                encoded = encode_example(tokenizer, prompt, row)
                p = encoded["prompt_length"]
                self.assertEqual(encoded["labels"][:p], [-100]*p)
                self.assertEqual(encoded["labels"][p:], tokenizer.encode(row["proof_body"],add_special_tokens=False)+[tokenizer.eos_token_id])
                self.assertLessEqual(encoded["sequence_length"],768)
                self.assertEqual(encoded["labels"][-1],tokenizer.eos_token_id)

    def test_oversized_input_errors_without_truncation(self):
        row = copy.deepcopy(load_split("train")[0])
        row["informal_proof"] = "This proof must not be truncated. " * 1000
        with self.assertRaisesRegex(ValueError,"truncation"):
            encode_example(self.tokenizer(),load_frozen()[1],row)

    def test_completion_loss_matches_full_causal_mask_and_supervises_eos(self):
        import torch
        import torch.nn.functional as F
        from training_protocol_v1.data import completion_loss
        logits = torch.randn(1,7,11, generator=torch.Generator().manual_seed(1729))
        encoded = {"prompt_length":4,"sequence_length":7,"input_ids":[1,2,3,4,5,6,10],
                   "labels":[-100,-100,-100,-100,5,6,10],"attention_mask":[1]*7}
        class Dummy:
            def __call__(self, **kwargs):
                self.positions = kwargs["logits_to_keep"].tolist()
                self.no_labels = "labels" not in kwargs
                return SimpleNamespace(logits=logits[:,kwargs["logits_to_keep"],:])
        dummy = Dummy()
        actual = completion_loss(dummy,encoded)
        expected = F.cross_entropy(logits[:,:-1,:].reshape(-1,11),torch.tensor(encoded["labels"][1:]),ignore_index=-100)
        torch.testing.assert_close(actual,expected)
        self.assertEqual(dummy.positions,[3,4,5])
        self.assertTrue(dummy.no_labels)
        encoded["labels"][0] = 1
        with self.assertRaises(ValueError):
            completion_loss(dummy,encoded)

    def test_zero_lora_preserves_base_and_only_adapters_receive_gradients(self):
        import torch
        from training_protocol_v1.adaptation import LoRALinear
        torch.manual_seed(1729)
        base = torch.nn.Linear(12,9)
        weight, bias = base.weight.detach().clone(), base.bias.detach().clone()
        layer = LoRALinear(base)
        x = torch.randn(2,12)
        torch.testing.assert_close(layer(x),base(x),rtol=0,atol=0)
        layer(x).square().mean().backward()
        self.assertIsNone(base.weight.grad)
        self.assertIsNone(base.bias.grad)
        self.assertIsNotNone(layer.B.grad)
        self.assertGreater(float(layer.B.grad.abs().sum()),0)
        torch.testing.assert_close(base.weight,weight,rtol=0,atol=0)
        torch.testing.assert_close(base.bias,bias,rtol=0,atol=0)
        # No optimizer step, even on this synthetic layer.


if __name__ == "__main__":
    unittest.main()
