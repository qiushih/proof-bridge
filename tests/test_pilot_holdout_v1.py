import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pilot_holdout_v1 import protocol
from pilot_holdout_v1.assessment import masked_records,validate_reviews,build_report


class HoldoutTests(unittest.TestCase):
    def test_original_content_stays_unread_in_input_checks(self):
        original=Path.open
        def guard(path,*args,**kwargs):
            mode=args[0] if args else kwargs.get('mode','r')
            if 'data/pilot-v1/reserved' in str(path) and 'b' not in mode: raise AssertionError('Original content read')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guard): self.assertEqual(protocol.check_inputs(),293)

    def test_existing_run_rejected_before_opening_or_model_load(self):
        from pilot_holdout_v1.experiment import run
        with tempfile.TemporaryDirectory() as folder,patch('pilot_holdout_v1.experiment.OUTPUT',Path(folder)),patch('pilot_holdout_v1.experiment.check_protocol',side_effect=AssertionError('Too late')):
            with self.assertRaisesRegex(ValueError,'no reopening'): run()

    def test_single_opening_records_custody_before_content_and_rejects_second(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp); source=folder/'synthetic.jsonl';source.write_text('synthetic test only\n')
            frozen={'frozen_at_utc':'2020-01-01T00:00:00+00:00','holdout_sha256':'synthetic'}
            with patch.object(protocol,'OUTPUT',folder),patch.object(protocol,'RESERVED',source),patch.object(protocol,'FROZEN',source),patch.object(protocol,'check_protocol',return_value=frozen),patch.object(protocol,'opened_rows',return_value=[]):
                original=Path.read_text
                calls=[]
                def read(path,*args,**kwargs):
                    if path==source:
                        self.assertTrue((folder/'custody.json').exists()); calls.append(path)
                    return original(path,*args,**kwargs)
                with patch.object(Path,'read_text',read):
                    protocol.open_original_once()
                    with self.assertRaises(FileExistsError): protocol.open_original_once()
                self.assertEqual(calls,[source])

    def test_masked_packet_has_no_model_or_step_score_fields(self):
        examples={'x':{'id':'x','formal_statement':'forall n : nat, n = n','informal_proof':'By reflexivity.','proof_body':'intros n. reflexivity.'}}
        rows=[{'key':f'{model}:x','model_key':model,'example_id':'x','proof_body':'intros n. reflexivity.','proof_body_sha256':'hash','verification':{'status':'PASS'},'requested_proof_steps':{'matched':True}} for model in ('base','step18')]
        packet,mapping=masked_records(rows,examples,53017)
        self.assertEqual(len(packet),2);self.assertEqual(set(mapping.values()),{'base:x','step18:x'})
        for r in packet:
            self.assertTrue(set(r).isdisjoint({'model_key','key','requested_proof_steps','generation_latency_seconds','usage'}))
        self.assertEqual(masked_records(rows,examples,53017),(packet,mapping))

    def test_failed_or_unbound_reviews_cannot_be_faithful(self):
        packet=[{'review_id':'R01','proof_body_sha256':'hash','verification':{'status':'FAIL'}}]
        decisions={'reviewer':'assistant','human_reviewed':False,'reviews':[{'review_id':'R01','proof_body_sha256':'hash','status':'FAITHFUL','reason':'test'}]}
        with self.assertRaises(ValueError): validate_reviews(packet,decisions)
        decisions['reviews'][0]['status']='NOT_ESTABLISHED'
        validate_reviews(packet,decisions)
        decisions['reviews'][0]['proof_body_sha256']='other'
        with self.assertRaises(ValueError): validate_reviews(packet,decisions)

    def test_gold_code_never_enters_target_prompt(self):
        from prompt_v2.experiment import load_frozen
        from prompt_v2.protocol import build_messages
        _,prompt=load_frozen()
        example={'formal_statement':'forall n : nat, n = n','informal_proof':'By reflexivity.','proof_body':'SECRET_TARGET_SENTINEL','id':'SECRET_ID_SENTINEL'}
        text=json.dumps(build_messages(prompt,example,'theorem_and_informal'))
        self.assertNotIn('SECRET_TARGET_SENTINEL',text);self.assertNotIn('SECRET_ID_SENTINEL',text)
        self.assertIn(example['informal_proof'],text)

    def test_exact_adapter_loading_rejects_wrong_shape_without_partial_copy(self):
        import torch
        from training_protocol_v1.adaptation import LoRALinear
        from pilot_holdout_v1.worker import load_adapter
        from lora_training_v1.core import base_hashes
        model=torch.nn.Sequential(LoRALinear(torch.nn.Linear(3,2)))
        base=base_hashes(model)
        tensors={n:torch.full_like(p,0.125) for n,p in model.named_parameters() if n.endswith(('.A','.B'))}
        load_adapter(model,tensors,strict=False)
        self.assertEqual(base_hashes(model),base)
        invalid=copy.deepcopy(tensors);invalid['0.B']=torch.ones(7,7)
        with self.assertRaises(ValueError): load_adapter(model,invalid,strict=False)
        for n,p in model.named_parameters():
            if n in tensors: torch.testing.assert_close(p,tensors[n])

    @unittest.skipUnless((protocol.OUTPUT/'summary.json').exists(),'Holdout report not yet created')
    def test_saved_report_reproduces_without_original_content_reopening(self):
        original=Path.open
        def guard(path,*args,**kwargs):
            mode=args[0] if args else kwargs.get('mode','r')
            if 'data/pilot-v1/reserved' in str(path) and 'b' not in mode: raise AssertionError('Original reopened')
            return original(path,*args,**kwargs)
        with patch.object(Path,'open',guard): text,summary,rows=build_report()
        self.assertEqual((protocol.OUTPUT/'HOLDOUT_REPORT.md').read_text(),text)
        self.assertEqual(json.loads((protocol.OUTPUT/'summary.json').read_text()),summary)
        self.assertEqual(len(rows),12);self.assertEqual(summary['optimizer_updates'],0)
        self.assertEqual(summary['holdout_status'],'CONSUMED')


if __name__=='__main__': unittest.main()
