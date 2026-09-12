"""Public-fixture tests only: never read the real holdout or load Qwen weights."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from constrained_v1.grammar import initial, feed, can_end
from constrained_v1.tokens import Vocabulary
from constrained_v1.experiment import tokenizer_only
from baseline_dev_v2.experiment import metrics
from pilot_training_v2.report import reviewed
from pilot_holdout_v2 import protocol as p, experiment as e, assessment as a


def fixtures():
    rows = copy.deepcopy(p.dev_rows()[:6])
    for row, ident in zip(rows, p.IDS):
        row.update(id=ident, theorem_id=ident.rsplit('_', 1)[0], split='holdout')
    return rows


def fake_generation(error=None):
    return {'generation_error': error, 'generation_latency_seconds': 0.0,
            'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2}}


class BoundaryTests(unittest.TestCase):
    def test_guard_denies_original_text_and_direct_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            original = Path(tmp) / 'reserved/holdout.jsonl'
            original.parent.mkdir(); original.write_text('public fixture')
            with p.holdout_io(original):
                with self.assertRaisesRegex(ValueError, 'sealed'): original.read_text()
                with self.assertRaisesRegex(ValueError, 'sealed'): original.read_bytes()
                self.assertEqual(len(p.sha256_file(original)), 64)

    def test_guard_rejects_historical_sets_even_for_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            for relative in ('data/pilot-v1/reserved/a', 'results/holdout-v1/a', 'data/evaluation/a'):
                path = Path(tmp) / relative
                path.parent.mkdir(parents=True); path.write_text('public fixture')
                with p.holdout_io(root=Path(tmp)), self.assertRaises(PermissionError): p.sha256_file(path)

    def test_exclusive_custody_precedes_single_original_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); original = root / 'reserved/holdout.jsonl'
            original.parent.mkdir(); original.write_text(''.join(json.dumps(r) + '\n' for r in fixtures()))
            frozen = root / 'frozen.json'; frozen.write_text('{}')
            out = root / 'run'; out.mkdir()
            config = {'holdout_sha256': p.sha256_file(original)}
            actual_read = Path.read_bytes
            openings = []
            def checked(path):
                if path == original:
                    self.assertTrue((out / 'custody.json').exists())
                    openings.append(path)
                return actual_read(path)
            with patch.object(p, 'FROZEN', frozen), patch.object(Path, 'read_bytes', checked), p.holdout_io(original):
                self.assertEqual(len(p.open_original_once(out, config, original)), 6)
                with self.assertRaises(FileExistsError): p.open_original_once(out, config, original)
                self.assertEqual(len(openings), 1)
                self.assertFalse(p.ORIGINAL_OPEN_ALLOWED)
                (out / 'opened_holdout.jsonl').write_text('tampered')
                with self.assertRaisesRegex(ValueError, 'snapshot changed'): p.opened_rows(out, config)

    def test_run_refuses_existing_directory_before_opening(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'OUTPUT', Path(tmp)), patch.object(e, 'open_original_once') as opened:
            with self.assertRaisesRegex(ValueError, 'already exists'): e.run()
            opened.assert_not_called()

    def test_missing_row_rejected(self):
        with self.assertRaisesRegex(ValueError, 'six ordered'): p.validate_rows(fixtures()[:-1])

    def test_duplicate_row_rejected(self):
        rows = fixtures(); rows[-1] = rows[-2]
        with self.assertRaisesRegex(ValueError, 'six ordered'): p.validate_rows(rows)

    def test_mismatched_same_theorem_pair_rejected(self):
        rows = fixtures(); rows[1]['formal_statement'] = 'forall n : nat, n = n'
        with self.assertRaisesRegex(ValueError, 'A/B pair'): p.validate_rows(rows)

    def test_reference_metadata_does_not_enter_prompt(self):
        row = fixtures()[0]
        before = e.prompt_for(row)
        row['proof_body'] = 'SENTINEL_TARGET_MUST_NOT_ENTER_PROMPT'
        row['steps'] = [{'code': 'SENTINEL_CODE', 'text': 'SENTINEL_ALIGNMENT'}]
        row['argument_contract'] = {'base': 'SENTINEL_CONTRACT'}
        row['generation_family'] = 'SENTINEL_FAMILY'
        self.assertEqual(before, e.prompt_for(row))
        self.assertNotIn('SENTINEL', json.dumps(before))

    def test_checkpoint_reselection_is_rejected(self):
        selections = copy.deepcopy(p.selected()); selections['control']['selected_step'] = 42
        with patch.object(p, 'read', return_value=selections), self.assertRaisesRegex(ValueError, 'checkpoint changed'):
            p.selected()

    def test_review_file_hash_binding_required(self):
        with self.assertRaisesRegex(ValueError, 'not bound'):
            reviewed([], {'raw_sha256_by_checkpoint': {}}, {'base/raw_generations.jsonl': 'expected'})


class GenerationTests(unittest.TestCase):
    def test_six_attempts_and_exact_journal_no_retry(self):
        rows = fixtures()
        values = [(row['proof_body'], fake_generation()) for row in rows]
        # The first candidate is intentionally wrong; the other five are public references.
        values[0] = ('intros n. reflexivity.', fake_generation())
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'generate', side_effect=values) as generated:
            folder = Path(tmp)
            e.generate_rows(folder, 'base', rows, None, None, None)
            records = a.read_rows(folder / 'raw_generations.jsonl')
            self.assertEqual(generated.call_count, 6)
            self.assertEqual(len(records), 6)
            self.assertEqual(records[0]['verification']['status'], 'FAIL')
            self.assertTrue(all(r['verification']['status'] == 'PASS' for r in records[1:]))
            a.check_identity(records, 'base', None)
            a.check_journal(records, a.read_rows(folder / 'attempts.jsonl'))
            with self.assertRaises(FileExistsError): e.generate_rows(folder, 'base', rows, None, None, None)
            self.assertEqual(generated.call_count, 6)

    def test_runtime_error_preserved_and_stops_without_next_attempt(self):
        rows = fixtures()
        value = (rows[0]['proof_body'], fake_generation({'type': 'RuntimeError', 'message': 'fixture'}))
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'generate', return_value=value) as generated:
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'no retry'): e.generate_rows(folder, 'base', rows, None, None, None)
            self.assertEqual(generated.call_count, 1)
            self.assertEqual(len(a.read_rows(folder / 'raw_generations.jsonl')), 1)
            self.assertEqual(len(a.read_rows(folder / 'attempts.jsonl')), 2)

    def test_verifier_environment_failure_preserved_and_stops(self):
        rows = fixtures()
        failure = replace(e.verify(rows[0]['formal_statement'], rows[0]['proof_body']), status='FAIL', category='ENVIRONMENT_ERROR')
        with tempfile.TemporaryDirectory() as tmp, patch.object(e, 'generate', return_value=(rows[0]['proof_body'], fake_generation())) as generated, patch.object(e, 'verify', return_value=failure):
            folder = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'environment failure'): e.generate_rows(folder, 'base', rows, None, None, None)
            self.assertEqual(generated.call_count, 1)
            self.assertEqual(a.read_rows(folder / 'raw_generations.jsonl')[0]['failure_category'], 'ENVIRONMENT_ERROR')

    def test_missing_finished_journal_event_rejected(self):
        with self.assertRaisesRegex(ValueError, 'journal'):
            a.check_journal([{'key': 'k'}], [{'key': 'k', 'event': 'started', 'attempt': 1}])

    def test_second_attempt_rejected(self):
        events = [{'key': 'k', 'event': x, 'attempt': 2} for x in ('started', 'finished')]
        with self.assertRaisesRegex(ValueError, 'journal'): a.check_journal([{'key': 'k'}], events)

    def test_wrong_adapter_or_order_rejected(self):
        rows = [{'example_id': ident, 'theorem_id': ident.rsplit('_', 1)[0], 'model_key': 'control',
                 'key': 'holdout_v2:control:' + ident, 'condition': 'argument_' + ident[-1],
                 'checkpoint_adapter_sha256': 'correct'} for ident in p.IDS]
        with self.assertRaisesRegex(ValueError, 'adapter differs'): a.check_identity(rows, 'control', 'wrong')
        with self.assertRaisesRegex(ValueError, 'ordered unique'): a.check_identity(list(reversed(rows)), 'control', 'correct')


class ReviewTests(unittest.TestCase):
    def records(self):
        records = []
        for row in fixtures():
            records.append(e.generation_record('base', row, row['proof_body'], fake_generation(), None))
        return records

    def reviews(self, rows, status='FAITHFUL'):
        return {'reviewer': 'assistant', 'human_reviewed': False, 'raw_sha256_by_checkpoint': {},
                'reviews': [{'key': r['key'], 'proof_body_sha256': r['proof_body_sha256'], 'status': status,
                             'reason': 'Public fixture judgment'} for r in rows]}

    def test_equivalent_tactic_faithful_but_step_mismatch(self):
        examples = fixtures(); row = examples[1]
        # B supplies congruence; the mathematically equivalent A body rewrites IH.
        r = e.generation_record('base', row, examples[0]['proof_body'], fake_generation(), None)
        judged = reviewed([r], self.reviews([r]), {})[0]
        self.assertEqual(judged['verification']['status'], 'PASS')
        self.assertEqual(judged['mathematical_argument_fidelity']['status'], 'FAITHFUL')
        self.assertFalse(judged['requested_proof_steps']['matched'])

    def test_failed_proof_cannot_count_as_faithful(self):
        row = fixtures()[0]
        r = e.generation_record('base', row, 'intros n. reflexivity.', fake_generation(), None)
        with self.assertRaisesRegex(ValueError, 'Failed proof'): reviewed([r], self.reviews([r]), {})

    def test_unresolved_verified_proof_does_not_count_as_faithful(self):
        rows = self.records()
        judged = reviewed(rows, self.reviews(rows, 'NOT_ESTABLISHED'), {})
        m = metrics(judged)
        self.assertEqual(m['verified'], 6)
        self.assertEqual(m['verified_faithful'], 0)
        self.assertEqual(m['successful_pairs'], 0)

    def test_missing_or_duplicate_reviews_rejected(self):
        rows = self.records(); reviews = self.reviews(rows)
        reviews['reviews'][-1] = reviews['reviews'][0]
        with self.assertRaisesRegex(ValueError, 'unique raw output'): reviewed(rows, reviews, {})

    def test_pair_success_requires_both_step_variants(self):
        rows = self.records(); judged = reviewed(rows, self.reviews(rows), {})
        self.assertEqual(metrics(judged)['successful_pairs'], 3)
        judged[1]['requested_proof_steps']['matched'] = False
        self.assertEqual(metrics(judged)['verified_faithful'], 6)
        self.assertEqual(metrics(judged)['successful_pairs'], 2)


class PublicTokenAndRocqTests(unittest.TestCase):
    def test_all_eight_public_references_replay_through_existing_decoder_and_rocq(self):
        tokenizer = tokenizer_only(); vocabulary = Vocabulary(tokenizer)
        for row in p.dev_rows():
            with self.subTest(example=row['id']):
                state = initial(row['formal_statement'])
                for token in tokenizer.encode(row['proof_body'], add_special_tokens=False):
                    self.assertIn(token, vocabulary.allowed(state))
                    state = feed(state, vocabulary.pieces[token])
                self.assertTrue(can_end(state))
                self.assertIn(vocabulary.eos_id, vocabulary.allowed(state))
                result = e.verify(row['formal_statement'], row['proof_body'], timeout=10.0)
                self.assertEqual(result.status, 'PASS')
                self.assertTrue(result.kernel_checked and result.assumptions_checked)


if __name__ == '__main__':
    with p.holdout_io():
        unittest.main()
