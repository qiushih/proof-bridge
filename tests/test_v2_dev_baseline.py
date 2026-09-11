import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from baseline_dev_v2 import experiment as e


def row(argument='A', passed=True, faithful=True, steps=True):
    body = 'intros n. reflexivity.'
    return {'key': argument, 'theorem_id': 't', 'proof_body': body,
            'verification': {'status': 'PASS' if passed else 'FAIL'},
            'mathematical_argument_fidelity': {'status': 'FAITHFUL' if faithful else 'NOT_ESTABLISHED'},
            'requested_proof_steps': {'matched': steps}, 'failure_category': None if passed else 'PROOF_ERROR',
            'generation_latency_seconds': 1.0, 'usage': {'prompt_tokens': 3, 'completion_tokens': 2}}


class MetricTests(unittest.TestCase):
    def test_one_verified_argument_does_not_make_a_successful_pair(self):
        m = e.metrics([row(), row('B', False, False, False)])
        self.assertEqual(m['verified'], 1)
        self.assertEqual(m['successful_pairs'], 0)

    def test_faithful_equivalent_tactic_can_fail_requested_steps(self):
        m = e.metrics([row(), row('B', steps=False)])
        self.assertEqual(m['verified_faithful'], 2)
        self.assertEqual(m['verified_requested_steps'], 1)
        self.assertEqual(m['successful_pairs'], 0)

    def test_both_correct_variants_make_one_pair(self):
        self.assertEqual(e.metrics([row(), row('B')])['successful_pairs'], 1)

    def test_failed_proof_cannot_be_marked_faithful(self):
        r = row(passed=False)
        review = {'key': r['key'], 'proof_body_sha256': e.digest(r['proof_body']), 'reviewer': 'assistant',
                  'human_reviewed': False, 'status': 'FAITHFUL', 'reason': 'A review'}
        with self.assertRaisesRegex(ValueError, 'cannot establish'):
            e.attach_review(r, review)

    def test_unbound_review_rejected(self):
        r = row()
        review = {'key': r['key'], 'proof_body_sha256': 'incorrect'}
        with self.assertRaisesRegex(ValueError, 'not bound'):
            e.attach_review(r, review)

    def test_prompt_drift_prevents_reuse(self):
        class Tokenizer:
            def apply_chat_template(self, messages, **kwargs): return str(messages)
            def encode(self, text, **kwargs): return [len(text)]
        example = {'formal_statement': 'forall n : nat, n = n'}
        with patch.object(e, 'prompt_for', return_value=[{'role': 'user', 'content': 'fixed'}]):
            with self.assertRaisesRegex(ValueError, 'identity differs'):
                e.check_prompt({'formal_statement': example['formal_statement'], 'prompt': {}}, example, Tokenizer())


OUTPUT = Path(os.environ.get('PROOFBRIDGE_V2_BASELINE_TEST_OUTPUT', str(e.OUT)))

@unittest.skipUnless((OUTPUT / 'release.json').exists(), 'Completed baseline not yet available')
class SavedEvidenceTests(unittest.TestCase):
    def test_final_evidence_and_exact_counts(self):
        result = e.check(OUTPUT)
        self.assertEqual(result['new_attempts'], 4)
        self.assertEqual(result['reused_records'], 12)
        self.assertEqual(result['total_records'], 16)

    def test_check_never_opens_holdout_or_diagnostic_examples(self):
        original = Path.open
        def guarded(path, *args, **kwargs):
            name = str(path.resolve())
            if any(part in name for part in ('/reserved/', '/data/evaluation/', '/results/holdout-v1/')):
                raise AssertionError('Prohibited access')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'open', guarded):
            e.check(OUTPUT)

    def test_run_refuses_repeating_attempts(self):
        with self.assertRaises(FileExistsError):
            e.run(OUTPUT)

    def test_protocol_was_frozen_before_any_attempt(self):
        p = e.read(OUTPUT / 'protocol.json')
        for model in e.MODELS:
            for event in e.read_rows(OUTPUT / model / 'attempts.jsonl'):
                if event['event'] == 'started':
                    self.assertGreater(event['at_utc'], p['frozen_at_utc'])


if __name__ == '__main__':
    unittest.main()
