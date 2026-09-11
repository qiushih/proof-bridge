"""Protocol/data safety checks; no model weights, generation or holdout parsing."""
import copy
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from training_protocol_v2 import preparation as p
from training_protocol_v2 import scoring as s
from training_protocol_v1.scoring import selection_score as old_selection_score


def records():
    return [{'dev_example_id': i, 'condition': 'argument_' + i[-1],
             'verification': {'status': 'PASS'},
             'mathematical_argument_fidelity': {'status': 'FAITHFUL'},
             'requested_proof_steps': {'matched': True}} for i in s.DEV_IDS]


class SelectionTests(unittest.TestCase):
    def test_exact_eight_rows_and_earliest_tie(self):
        r = records()
        self.assertEqual(s.selection_score(r, 0), [8, 8, 8, 0])
        self.assertGreater(s.selection_score(r, 6), s.selection_score(r, 12))

    def test_missing_duplicate_wrong_condition_or_unplanned_step_rejected(self):
        bad_condition = records(); bad_condition[-1]['condition'] = 'theorem_only'
        for rows in (records()[:-1], records()[:-1] + [records()[0]], bad_condition):
            with self.assertRaises(ValueError): s.selection_score(rows, 6)
        with self.assertRaises(ValueError): s.selection_score(records(), 7)

    def test_old_scorer_still_rejects_eight_rows(self):
        with self.assertRaisesRegex(ValueError, 'exactly the six frozen'):
            old_selection_score(records(), 0)

    def test_equivalent_verified_tactic_can_be_faithful_without_step_match(self):
        r = records(); r[-1]['requested_proof_steps']['matched'] = False
        score = s.selection_score(r, 6)
        self.assertEqual(score[:3], [8, 7, 8])
        self.assertEqual(s.metrics(r)['new_two']['both_argument_successes'], 0)

    def test_failed_proof_is_not_counted_despite_faithful_label_or_trace(self):
        r = records(); r[-1]['verification']['status'] = 'FAIL'
        self.assertEqual(s.selection_score(r, 6)[:3], [7, 7, 7])

    def test_unresolved_review_cannot_count_as_faithful(self):
        r = records(); r[-1]['mathematical_argument_fidelity']['status'] = 'NOT_ESTABLISHED'
        self.assertEqual(s.selection_score(r, 6)[0], 7)

    def test_gates_require_new_pair_and_preserve_original_six(self):
        c, t = records(), records()
        for r in c:
            if r['dev_example_id'].endswith('B'): r['requested_proof_steps']['matched'] = False
        self.assertTrue(s.success_criteria(c, t)['success'])
        t[-1]['verification']['status'] = 'FAIL'
        self.assertFalse(s.success_criteria(c, t)['success'])
        t = records(); t[0]['verification']['status'] = 'FAIL'
        self.assertFalse(s.success_criteria(c, t)['success'])


class ScheduleTests(unittest.TestCase):
    def test_same_seed_produces_exact_update_budget_and_control_epochs(self):
        ids = [r['id'] for r in p.read_rows(p.ROOT / 'data/pilot-v1/train.jsonl')]
        a, b = p.schedule(ids), p.schedule(list(reversed(ids)))
        self.assertEqual(a, b)
        self.assertEqual(len(a['updates']), 60)
        self.assertTrue(all(len(g) == 4 for g in a['updates']))
        self.assertEqual(set(a['counts_by_id'].values()), {10})

    def test_intervention_budget_crosses_cycle_boundaries_without_dropping(self):
        ids = [r['id'] for r in p.read_rows(p.ROOT / 'data/pilot-v1/train.jsonl')] + p.NEW_IDS
        result = p.schedule(ids)
        self.assertEqual(result['complete_cycles'], 9)
        self.assertEqual(result['final_cycle_rows'], 6)
        self.assertEqual(sum(result['counts_by_id'].values()), 240)
        self.assertEqual(set(result['counts_by_id']), set(ids))
        self.assertTrue(set(result['counts_by_id'].values()) <= {9, 10})
        self.assertEqual(sum(result['updates'], []), result['microbatches'])

    def test_unexpected_or_duplicate_training_membership_rejected(self):
        for ids in ([str(i) for i in range(25)], ['duplicate'] * 24):
            with self.assertRaises(ValueError): p.schedule(ids)


OUTPUT = Path(os.environ.get('PROOFBRIDGE_V2_PREPARATION_TEST_DATA', str(p.DATA)))

@unittest.skipUnless((OUTPUT / 'release.json').exists(), 'Prepared release not yet available')
class ReleaseTests(unittest.TestCase):
    def test_frozen_release_and_original_bytes(self):
        result = p.check(OUTPUT)
        self.assertEqual(result['control_rows'], 24)
        self.assertEqual(result['intervention_rows'], 26)
        self.assertEqual(result['dev_rows'], 8)
        self.assertEqual(result['sealed_holdout_rows'], 6)
        self.assertFalse(result['holdout_content_parsed'])

    def test_checker_never_reads_private_text_or_old_holdout_diagnostics(self):
        original_text, original_open = Path.read_text, Path.open
        def guarded_text(path, *args, **kwargs):
            if '/reserved/' in str(path.resolve()):
                raise AssertionError('Sealed content parsed')
            return original_text(path, *args, **kwargs)
        def guarded_open(path, *args, **kwargs):
            name = str(path.resolve())
            if any(part in name for part in ('/data/pilot-v1/reserved/', '/results/holdout-v1/', '/data/evaluation/')):
                raise AssertionError('Historical protected content opened')
            return original_open(path, *args, **kwargs)
        with patch.object(Path, 'read_text', guarded_text), patch.object(Path, 'open', guarded_open):
            self.assertEqual(p.check(OUTPUT)['status'], 'PASS')

    def test_public_loader_refuses_holdout_before_opening_any_file(self):
        with patch.object(p, 'read', side_effect=AssertionError('Unexpected file read')):
            with self.assertRaisesRegex(ValueError, 'holdout is sealed'):
                p.load_public('holdout', OUTPUT)

    def test_prepare_refuses_existing_release_without_reading_draft(self):
        with patch.object(p, 'read', side_effect=AssertionError('Unexpected file read')):
            with self.assertRaisesRegex(ValueError, 'already exists'):
                p.prepare(OUTPUT, None)

    def test_tampered_schedule_detected(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / 'copy'
            shutil.copytree(OUTPUT, target)
            path = target / 'schedule.json'
            path.write_bytes(path.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'Frozen artifact changed'):
                p.check(target)

    def test_family_reassignment_rejected(self):
        r = copy.deepcopy(p.load_public('intervention', OUTPUT)[-1])
        r['split'] = 'holdout'
        with self.assertRaisesRegex(ValueError, 'Family/split mismatch'):
            p.validate_row(r)

    def test_canonical_alignment_tampering_rejected(self):
        r = copy.deepcopy(p.load_public('intervention', OUTPUT)[-1])
        r['informal_proof'] += ' Extra text.'
        with self.assertRaisesRegex(ValueError, 'Canonical informal'):
            p.validate_row(r)

    def test_design_precedes_fresh_holdout_and_final_seal(self):
        design, prep, manifest = p.check_design(), p.read(OUTPUT / 'preparation.json'), p.read(OUTPUT / 'split_manifest.json')
        self.assertLess(design['precommitted_at_utc'], prep['draft_created_at_utc'])
        self.assertLess(prep['draft_created_at_utc'], manifest['frozen_at_utc'])


if __name__ == '__main__':
    unittest.main()
