"""Synthetic runner tests: no Qwen loading, real generations, or holdout content."""
import copy
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import load_file

from pilot_training_v2 import common as c, core, evaluation as ev, experiment as e, audit, report
from training_protocol_v1.adaptation import LoRALinear
from training_protocol_v1.data import completion_loss
from training_protocol_v2 import preparation
from training_protocol_v2.scoring import DEV_IDS, STEPS


def fixture_model():
    torch.manual_seed(1729)
    return torch.nn.Sequential(LoRALinear(torch.nn.Linear(3, 2)))


def fixture_encoded(arm):
    ids = [r['id'] for r in preparation.load_public(arm)]
    return {i: {'id': i, 'value': (n % 3 + 1) / 10, 'sequence_length': 7, 'target_length': 3}
            for n, i in enumerate(ids)}


def fixture_loss(model, row):
    return (model(torch.full((1, 3), row['value'])) - 0.2).square().mean()


def judged_rows():
    return [{'key': i, 'dev_example_id': i, 'condition': 'argument_' + i[-1],
             'proof_body_sha256': 'test-hash', 'verification': {'status': 'PASS'},
             'mathematical_argument_fidelity': {'status': 'FAITHFUL'},
             'requested_proof_steps': {'matched': True}} for i in DEV_IDS]


class GuardAndEntryTests(unittest.TestCase):
    def test_frozen_inputs_and_shared_numerical_configuration(self):
        with c.sealed_io():
            self.assertTrue(c.inputs())
        self.assertEqual(c.config()['training']['max_optimizer_steps'], 60)

    def test_new_reserved_text_and_direct_binary_reads_blocked_but_hash_allowed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'reserved/fixture.txt'; path.parent.mkdir(); path.write_text('synthetic secret')
            with c.sealed_io():
                with self.assertRaisesRegex(ValueError, 'only be hashed'): path.read_text()
                with self.assertRaisesRegex(ValueError, 'only be hashed'): path.read_bytes()
                self.assertEqual(len(c.sha256_file(path)), 64)

    def test_old_holdout_and_diagnostic_access_blocked_even_for_hashes(self):
        with c.sealed_io():
            for path in ('data/pilot-v1/reserved/holdout.jsonl', 'data/evaluation/never_open.jsonl', 'results/holdout-v1/never_open.json'):
                with self.assertRaises(PermissionError): c.sha256_file(c.ROOT / path)

    def test_guard_restores_open_functions_after_exception(self):
        import builtins, io
        a, b = builtins.open, io.open
        with self.assertRaises(RuntimeError):
            with c.sealed_io(): raise RuntimeError('fixture')
        self.assertIs(builtins.open, a); self.assertIs(io.open, b)

    def test_existing_run_rejected_before_inputs_or_model(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(e, 'check_ready', side_effect=AssertionError('Too late')):
            with self.assertRaisesRegex(ValueError, 'already exists'): e.run(Path(temp))

    def test_existing_arm_rejected_before_inputs_or_model(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(e, 'check_ready', side_effect=AssertionError('Too late')):
            folder = Path(temp); (folder / 'control').mkdir()
            with self.assertRaisesRegex(ValueError, 'already started'): e.worker(folder, 'control')

    def test_invalid_arm_rejected_and_holdout_loader_absent(self):
        with self.assertRaises(ValueError): c.public_rows('holdout')
        with self.assertRaises(ValueError): e.worker(Path('/unused'), 'holdout')

    def test_failed_control_stops_dispatch_and_preserves_interruption(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'run'; module = Path(temp) / 'module'; module.mkdir()
            (module / 'release.json').write_text('{}')
            with patch.object(e, 'HERE', module), patch.object(e, 'check_ready', return_value={'code_sha256': {}}), \
                 patch.object(e, 'hardware', return_value={}), patch.object(e, 'check_model', return_value={}), \
                 patch.object(e, 'inputs', return_value={}), patch.object(e, 'launch_arm', side_effect=RuntimeError('synthetic worker failure')) as launch:
                with self.assertRaises(RuntimeError): e.run(folder)
                self.assertEqual(launch.call_count, 1)
                self.assertEqual(launch.call_args.args[1], 'control')
            self.assertTrue((folder / 'interruption.json').exists())
            self.assertFalse((folder / 'completion.json').exists())
            self.assertFalse((folder / 'intervention').exists())

    def test_incomplete_run_cannot_be_assessed(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, 'No complete'): audit.check(Path(temp))


class StreamAndCheckpointTests(unittest.TestCase):
    def test_identical_initialization_check_rejects_adapter_or_base_drift(self):
        a = {'initial_adapter_hashes': {'A': 'x'}, 'base_hashes_before': {'w': 'y'}}
        core.matching_initial(a, copy.deepcopy(a))
        for field in ('initial_adapter_hashes', 'base_hashes_before'):
            b = copy.deepcopy(a); b[field] = {'bad': 'z'}
            with self.assertRaisesRegex(ValueError, 'identical'): core.matching_initial(a, b)

    def test_actual_tiny_tensor_loops_consume_both_full_frozen_streams(self):
        for arm in c.ARMS:
            model, encoded = fixture_model(), fixture_encoded(arm)
            before = core.base_hashes(model)
            optimizer, scheduler = core.make_optimizer(model, strict=False)
            schedule = c.schedules()[arm]
            callbacks = []
            with tempfile.TemporaryDirectory() as temp:
                output = Path(temp) / 'training'
                def callback(model, path, step):
                    callbacks.append(step)
                    cp = c.read(path / 'checkpoint.json')
                    self.assertEqual(cp['completed_microbatches'], step * 4)
                    self.assertEqual(cp['next_cycle_offset'], step * 4 % len(encoded))
                    # These are locally authored, temporary synthetic checkpoints.
                    saved = torch.load(path / 'trainer_state.pt', weights_only=False)
                    self.assertEqual(saved['microbatch_stream'], schedule['microbatches'])
                    self.assertEqual(saved['next_microbatch_index'], step * 4)
                    self.assertEqual(saved['scheduler']['last_epoch'], step)
                    self.assertEqual(saved['torch_rng'].dtype, torch.uint8)
                    self.assertIn('python_rng', saved); self.assertIn('numpy_rng', saved)
                    tensors = load_file(str(path / 'adapter.safetensors'))
                    named, _ = core.audit_trainable(model, strict=False)
                    self.assertEqual(set(tensors), set(named))
                    for name, value in tensors.items(): torch.testing.assert_close(value, named[name])
                    c.check_files(path, cp['files_sha256'])
                with patch.object(core, 'process_memory', return_value={'synthetic': True}), \
                     patch.object(core, 'swap_snapshot', return_value={'available': False, 'synthetic': True}):
                    result = core.run_updates(model, optimizer, scheduler, encoded, schedule, output, arm,
                                              before, {'synthetic_fixture': True}, callback, strict=False, loss_fn=fixture_loss)
                self.assertEqual(result['optimizer_updates'], 60)
                self.assertEqual(callbacks, STEPS[1:])
                events = c.read_rows(output / 'training_metrics.jsonl')
                starts = [r for r in events if r['event'] == 'forward_started']
                self.assertEqual([r['example_id'] for r in starts], schedule['microbatches'])
                updates = [r for r in events if r['event'] == 'optimizer_finished']
                self.assertEqual(len(updates), 60)
                self.assertEqual([r['example_ids'] for r in updates], schedule['updates'])
                self.assertEqual(before, core.base_hashes(model))
                # The production checker must not accept tiny fixture parameter counts.
                with self.assertRaisesRegex(ValueError, 'trainable'): audit.check_events(events, schedule['microbatches'])
                checks = copy.deepcopy(events)
                for r in checks:
                    if r['event'] == 'optimizer_finished': r['trainable_audit']['trainable_parameters'] = 540672
                self.assertEqual(len(audit.check_events(checks, schedule['microbatches'])), 60)
                with self.assertRaisesRegex(ValueError, 'event order'): audit.check_events(checks[:-1], schedule['microbatches'])
                with self.assertRaisesRegex(ValueError, 'stream differs'): audit.check_events(checks, list(reversed(schedule['microbatches'])))
                with self.assertRaises(FileExistsError):
                    core.save_checkpoint(model, optimizer, scheduler, output, 6, schedule['microbatches'], arm, {}, strict=False)

    def test_nonfinite_loss_stops_before_any_update(self):
        model = fixture_model(); optimizer, scheduler = core.make_optimizer(model, strict=False)
        with tempfile.TemporaryDirectory() as temp, patch.object(optimizer, 'step', side_effect=AssertionError('Optimizer must not run')):
            with self.assertRaises(FloatingPointError):
                core.run_updates(model, optimizer, scheduler, fixture_encoded('control'), c.schedules()['control'],
                                 Path(temp) / 'training', 'control', core.base_hashes(model), {}, lambda *a: None,
                                 strict=False, loss_fn=lambda *a: torch.tensor(float('nan'), requires_grad=True))
            self.assertFalse(list(Path(temp).rglob('adapter.safetensors')))

    def test_mutated_base_or_trainability_rejected(self):
        model = fixture_model()
        model[0].base.weight.requires_grad_(True)
        with self.assertRaises(ValueError): core.make_optimizer(model, strict=False)

    def test_partial_groups_or_reused_optimizer_rejected(self):
        model = fixture_model(); optimizer, scheduler = core.make_optimizer(model, strict=False)
        encoded = fixture_encoded('control'); schedule = c.schedules()['control']
        with tempfile.TemporaryDirectory() as temp:
            bad = copy.deepcopy(schedule); bad['updates'][-1] = bad['updates'][-1][:-1]
            with self.assertRaisesRegex(ValueError, 'Invalid stream'):
                core.run_updates(model, optimizer, scheduler, encoded, bad, Path(temp) / 'a', 'control', {}, {}, lambda *a: None, strict=False)
            scheduler.last_epoch = 2
            with self.assertRaisesRegex(ValueError, 'fresh'):
                core.run_updates(model, optimizer, scheduler, encoded, schedule, Path(temp) / 'b', 'control', {}, {}, lambda *a: None, strict=False)

    def test_completion_loss_matches_full_shifted_target_mask(self):
        logits = torch.randn(1, 7, 11)
        row = {'prompt_length': 4, 'sequence_length': 7, 'input_ids': [1,2,3,4,5,6,10],
               'labels': [-100,-100,-100,-100,5,6,10], 'attention_mask': [1]*7}
        def model(**kwargs): return SimpleNamespace(logits=logits[:, kwargs['logits_to_keep'], :])
        actual = completion_loss(model, row)
        expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), torch.tensor(row['labels'][1:]), ignore_index=-100)
        torch.testing.assert_close(actual, expected)
        row['labels'][0] = 1
        with self.assertRaises(ValueError): completion_loss(model, row)


class EvaluationTests(unittest.TestCase):
    def setup_fixture(self, folder):
        model = fixture_model().train()
        checkpoint = folder / 'checkpoint'; checkpoint.mkdir()
        (checkpoint / 'adapter.safetensors').write_bytes(b'synthetic placeholder only')
        named, _ = core.audit_trainable(model, strict=False)
        c.write(checkpoint / 'checkpoint.json', {'adapter_tensor_hashes': {n: core.tensor_digest(p) for n, p in named.items()}})
        callback = ev.CheckpointEvaluator(folder, 'control', object())
        return model, checkpoint, callback

    def patches(self):
        return (patch.object(ev, 'audit_trainable', side_effect=lambda m: core.audit_trainable(m, strict=False)),
                patch.object(ev, 'make_engine', return_value=object()), patch.object(ev, 'Vocabulary', return_value=object()))

    def test_exact_eight_reference_fixtures_reach_real_rocq_and_restore_rng(self):
        examples = c.dev_rows()
        outputs = iter([(r['proof_body'], {'generation_error': None}) for r in examples])
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); model, checkpoint, callback = self.setup_fixture(folder)
            states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
            a, b, d = self.patches()
            with a, b, d, patch.object(ev, 'generate', side_effect=lambda *args: next(outputs)) as mock_generate:
                callback(model, checkpoint, 6)
                self.assertEqual(mock_generate.call_count, 8)
                with self.assertRaisesRegex(ValueError, 'repeated'): callback(model, checkpoint, 6)
            rows = c.read_rows(folder / 'dev-step-0006/raw_generations.jsonl')
            self.assertEqual([r['dev_example_id'] for r in rows], DEV_IDS)
            self.assertTrue(all(r['verification']['status'] == 'PASS' and r['requested_proof_steps']['matched'] for r in rows))
            self.assertTrue(model.training)
            self.assertEqual(random.getstate(), states[0])
            np.testing.assert_equal(np.random.get_state(), states[1])
            self.assertTrue(torch.equal(torch.get_rng_state(), states[2]))
            self.assertEqual(len(c.read_rows(folder / 'dev-step-0006/attempts.jsonl')), 16)

    def test_generation_exception_preserves_attempt_and_restores_state_without_retry(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); model, checkpoint, callback = self.setup_fixture(folder)
            rng = torch.get_rng_state().clone()
            a, b, d = self.patches()
            with a, b, d, patch.object(ev, 'generate', side_effect=RuntimeError('fixture interrupted')) as mock_generate:
                with self.assertRaises(RuntimeError): callback(model, checkpoint, 6)
                self.assertEqual(mock_generate.call_count, 1)
            self.assertTrue(model.training)
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            path = folder / 'dev-step-0006'
            self.assertEqual(len(c.read_rows(path / 'attempts.jsonl')), 1)
            self.assertFalse((path / 'completion.json').exists())

    def test_recorded_generation_error_cannot_become_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); model, checkpoint, callback = self.setup_fixture(folder)
            a, b, d = self.patches()
            with a, b, d, patch.object(ev, 'generate', return_value=('', {'generation_error': {'type': 'fixture'}})) as mock_generate:
                with self.assertRaisesRegex(ValueError, 'no retry'): callback(model, checkpoint, 6)
                self.assertEqual(mock_generate.call_count, 1)
            path = folder / 'dev-step-0006'
            self.assertEqual(len(c.read_rows(path / 'raw_generations.jsonl')), 1)
            self.assertFalse((path / 'completion.json').exists())


    def test_verifier_environment_error_stops_after_one_recorded_attempt(self):
        from verifier import VerificationResult
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); model, checkpoint, callback = self.setup_fixture(folder)
            a, b, d = self.patches()
            failed = VerificationResult('FAIL', 'ENVIRONMENT_ERROR', 'Synthetic environment failure', 'environment')
            with a, b, d, patch.object(ev, 'generate', return_value=(c.dev_rows()[0]['proof_body'], {'generation_error': None})) as mock_generate, patch.object(ev, 'verify_candidate', return_value=failed):
                with self.assertRaisesRegex(ValueError, 'environment failure'): callback(model, checkpoint, 6)
                self.assertEqual(mock_generate.call_count, 1)
            self.assertFalse((folder / 'dev-step-0006/completion.json').exists())



class ReviewAndSelectionTests(unittest.TestCase):
    def test_all_checkpoints_required_and_zero_wins_ties(self):
        group = {s: copy.deepcopy(judged_rows()) for s in STEPS}
        self.assertEqual(report.select(group)[0], 0)
        del group[60]
        with self.assertRaisesRegex(ValueError, 'All ten'): report.select(group)

    def test_fidelity_precedes_step_adherence_and_earliest_wins_tie(self):
        group = {s: copy.deepcopy(judged_rows()) for s in STEPS}
        for rows in group.values(): rows[-1]['requested_proof_steps']['matched'] = False
        group[6][-1]['requested_proof_steps']['matched'] = True
        group[12] = copy.deepcopy(group[6])
        self.assertEqual(report.select(group)[0], 6)
        group[6][-1]['mathematical_argument_fidelity']['status'] = 'UNFAITHFUL'
        self.assertEqual(report.select(group)[0], 12)

    def test_missing_or_unbound_reviews_rejected(self):
        raw = judged_rows()
        reviews = [{'key': r['key'], 'proof_body_sha256': r['proof_body_sha256'], 'status': 'FAITHFUL',
                    'reason': 'Synthetic reviewer fixture.'} for r in raw]
        file = {'reviewer': 'assistant', 'human_reviewed': False, 'raw_sha256_by_checkpoint': {'file': 'hash'}, 'reviews': reviews}
        self.assertEqual(len(report.reviewed(raw, file, {'file': 'hash'})), 8)
        with self.assertRaisesRegex(ValueError, 'not bound'): report.reviewed(raw, file, {'file': 'changed'})
        bad = copy.deepcopy(file); bad['reviews'] = bad['reviews'][:-1]
        with self.assertRaises(ValueError): report.reviewed(raw, bad, {'file': 'hash'})
        bad = copy.deepcopy(file); bad['reviews'][0]['proof_body_sha256'] = 'wrong'
        with self.assertRaises(ValueError): report.reviewed(raw, bad, {'file': 'hash'})

    def test_failed_proof_cannot_be_faithful_or_unfaithful_complete_proof(self):
        raw = judged_rows(); raw[0]['verification']['status'] = 'FAIL'
        reviews = [{'key': r['key'], 'proof_body_sha256': r['proof_body_sha256'], 'status': 'FAITHFUL', 'reason': 'Fixture'} for r in raw]
        file = {'reviewer': 'assistant', 'human_reviewed': False, 'raw_sha256_by_checkpoint': {}, 'reviews': reviews}
        with self.assertRaises(ValueError): report.reviewed(raw, file, {})
        file['reviews'][0]['status'] = 'UNFAITHFUL'
        with self.assertRaises(ValueError): report.reviewed(raw, file, {})
        file['reviews'][0]['status'] = 'NOT_ESTABLISHED'
        self.assertEqual(len(report.reviewed(raw, file, {})), 8)

    def test_unresolved_verified_proof_counts_zero_fidelity_without_forced_judgment(self):
        raw = judged_rows()
        reviews = [{'key': r['key'], 'proof_body_sha256': r['proof_body_sha256'],
                    'status': 'FAITHFUL', 'reason': 'Synthetic fixture review.'} for r in raw]
        reviews[0]['status'] = 'NOT_ESTABLISHED'
        reviews[0]['reason'] = 'Synthetic unresolved fidelity review; do not assert unfaithfulness.'
        file = {'reviewer': 'assistant', 'human_reviewed': False, 'raw_sha256_by_checkpoint': {}, 'reviews': reviews}
        assessed = report.reviewed(raw, file, {})
        self.assertEqual(report.selection_score(assessed, 6), [7, 8, 8, -6])

    def test_report_refuses_overwrite_with_different_contents(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'report.md'
            report.persist_once(path, 'same')
            report.persist_once(path, 'same')
            with self.assertRaisesRegex(ValueError, 'refusing overwrite'): report.persist_once(path, 'different')

    def test_frozen_results_status_not_overwritten_by_assessment_status(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(report, 'write_report', return_value={'status': 'ASSESSED', 'selected_steps': {'control': 0, 'intervention': 6}}), \
             patch.object(report, 'check', return_value={'status': 'PASS', 'dev_token_replays': 160, 'dev_rocq_replays': 160}):
            self.assertEqual(report.freeze_results(Path(temp))['status'], 'FROZEN_TRAINED_DEV_RESULTS')


if __name__ == '__main__':
    unittest.main()
