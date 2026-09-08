import copy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from pilot_training_v1.experiment import OUTPUT, ROOT, check_inputs, run
from pilot_training_v1.report import select


class PilotTrainingTests(unittest.TestCase):
    def baseline(self):
        return [json.loads(s) for s in (ROOT/'results/pretraining-dev-v1/assessed_results.jsonl').read_text().splitlines() if json.loads(s)['condition']!='theorem_only']

    def test_selection_keeps_step_zero_on_ties_and_fidelity_before_step_match(self):
        baseline=self.baseline()
        checkpoints={s:copy.deepcopy(baseline) for s in range(0,61,6)}
        self.assertEqual(select(checkpoints)[0],0)
        for r in checkpoints[6]: r['requested_proof_steps']['matched']=True
        self.assertEqual(select(checkpoints)[0],6)
        checkpoints[12]=copy.deepcopy(checkpoints[6])
        self.assertEqual(select(checkpoints)[0],6)
        for step in (6,12): checkpoints[step][0]['mathematical_argument_fidelity']['status']='UNFAITHFUL'
        self.assertEqual(select(checkpoints)[0],0)
        del checkpoints[60]
        with self.assertRaises(ValueError): select(checkpoints)

    def test_existing_run_rejected_before_model_or_inputs(self):
        with tempfile.TemporaryDirectory() as folder, patch('pilot_training_v1.experiment.OUTPUT',Path(folder)), patch('pilot_training_v1.experiment.check_inputs',side_effect=AssertionError('Too late')):
            with self.assertRaisesRegex(ValueError,'refusing'): run()

    def test_protected_input_check_never_parses_holdout(self):
        original=Path.open
        def guard(path,*args,**kwargs):
            mode=args[0] if args else kwargs.get('mode','r')
            if 'data/pilot-v1/reserved' in str(path) and 'b' not in mode: raise AssertionError('Holdout read')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guard): self.assertEqual(check_inputs(),199)

    def test_evaluation_restores_rng_and_mode_even_on_error(self):
        import numpy as np
        import torch
        from pilot_training_v1.evaluation import evaluation_scope
        model=torch.nn.Linear(2,2).train()
        states=random.getstate(),np.random.get_state(),torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with evaluation_scope(model):
                self.assertFalse(model.training)
                random.random(); np.random.rand(); torch.rand(3)
                raise RuntimeError('synthetic interruption')
        self.assertTrue(model.training)
        self.assertEqual(random.getstate(),states[0])
        np.testing.assert_equal(np.random.get_state(),states[1])
        self.assertTrue(torch.equal(torch.get_rng_state(),states[2]))

    def test_callback_six_attempts_real_verifier_and_no_duplicate_checkpoint(self):
        import torch
        from lora_training_v1.core import audit_trainable
        from training_protocol_v1.adaptation import LoRALinear
        from training_protocol_v1.experiment import dev_groups
        from pilot_training_v1.evaluation import CheckpointEvaluator
        examples=[g['arguments'][a] for g in dev_groups() for a in ('A','B')]
        model=torch.nn.Sequential(LoRALinear(torch.nn.Linear(2,2)))
        generated=iter([(r['proof_body'],{'generation_error':None}) for r in examples])
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); cp=folder/'checkpoint'; cp.mkdir(); (cp/'adapter.safetensors').write_bytes(b'synthetic placeholder, not model weights')
            callback=CheckpointEvaluator(folder)
            with patch('pilot_training_v1.evaluation.audit_trainable',side_effect=lambda m:audit_trainable(m,strict=False)), patch('pilot_training_v1.evaluation.make_engine',return_value=object()), patch('pilot_training_v1.evaluation.Vocabulary',return_value=object()), patch('pilot_training_v1.evaluation.generate',side_effect=lambda *args:next(generated)) as generate:
                callback(model,object(),cp,6)
                self.assertEqual(generate.call_count,6)
                with self.assertRaises(ValueError): callback(model,object(),cp,6)
            rows=[json.loads(s) for s in (folder/'dev-step-0006/raw_generations.jsonl').read_text().splitlines()]
            self.assertEqual([r['dev_example_id'] for r in rows],[r['id'] for r in examples])
            self.assertTrue(all(r['verification']['status']=='PASS' and r['requested_proof_steps']['matched'] for r in rows))
            self.assertTrue(model.training)

    @unittest.skipUnless((OUTPUT/'summary.json').exists(),'Pilot report not yet complete')
    def test_saved_pilot_and_selection_reproduce(self):
        from pilot_training_v1.report import build_report
        text,summary,rows=build_report()
        self.assertEqual(len(rows),60)
        self.assertEqual((OUTPUT/'LORA_PILOT_REPORT.md').read_text(),text)
        self.assertEqual(json.loads((OUTPUT/'summary.json').read_text()),summary)
        self.assertFalse(summary['holdout_evaluated'])


if __name__=='__main__': unittest.main()
