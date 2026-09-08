import copy
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from lora_training_v1.preflight import OUTPUT, check_inputs
from lora_training_v1.report import classify
from training_protocol_v1.data import config


class FeasibilityPlanTests(unittest.TestCase):
    def test_preserved_historical_inputs(self):
        self.assertEqual(check_inputs(),178)

    def test_classification_boundary_and_no_false_memory_label(self):
        plan={"slow_threshold_native_update_seconds":120}
        result={"status":"PASS","base_parameters_unchanged":True,"updates":[
            {"pad_to":None,"update_wall_seconds":100},{"pad_to":768,"update_wall_seconds":500},
            {"pad_to":None,"update_wall_seconds":120}]}
        self.assertEqual(classify(result,{"timed_out":False},plan),("feasible",110))
        result["updates"][-1]["update_wall_seconds"]=200
        self.assertEqual(classify(result,{"timed_out":False},plan)[0],"feasible but impractically slow")
        self.assertEqual(classify({"failure_kind":"MEMORY"},{"timed_out":False},plan)[0],"infeasible due to memory")
        with self.assertRaises(ValueError):
            classify({"status":"FAIL","failure_kind":"IMPLEMENTATION_OR_RUNTIME"},{"timed_out":False},plan)

    def test_input_checks_only_hash_reserved_contents(self):
        original=Path.open
        def guarded(path,*args,**kwargs):
            mode=args[0] if args else kwargs.get("mode","r")
            if "data/pilot-v1/reserved" in str(path) and "b" not in mode:
                raise AssertionError("Holdout contents must remain sealed")
            return original(path,*args,**kwargs)
        with patch.object(Path,"open",guarded):
            check_inputs()


@unittest.skipUnless(importlib.util.find_spec("torch"),"Run tensor tests in pinned .venv")
class TrainingLoopTests(unittest.TestCase):
    def make_model(self):
        import torch
        from training_protocol_v1.adaptation import LoRALinear
        torch.manual_seed(1729)
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj=LoRALinear(torch.nn.Linear(12,9))
            def forward(self,x): return self.proj(x)
        return Toy()

    def examples(self):
        return [{"id":str(i),"x":i+1,"sequence_length":7,"target_length":3} for i in range(4)]

    def loss(self,model,example):
        import torch
        return model(torch.full((1,12),float(example["x"]))).square().mean()

    def test_actual_accumulation_matches_four_example_average_and_preserves_base(self):
        import torch
        from lora_training_v1.core import optimizer_update,make_optimizer,base_hashes,audit_trainable
        first=self.make_model(); second=copy.deepcopy(first)
        before=base_hashes(first)
        opt,sched=make_optimizer(first,strict=False)
        control,csched=make_optimizer(second,strict=False)
        events=[]
        result=optimizer_update(first,opt,sched,self.examples(),lambda e,**v:events.append((e,v)),1,
                                strict=False,loss_fn=self.loss)
        loss=sum(self.loss(second,e) for e in self.examples())/4
        loss.backward()
        named,_=audit_trainable(second,strict=False)
        torch.nn.utils.clip_grad_norm_(list(named.values()),config()["training"]["max_grad_norm"])
        control.step(); csched.step(); control.zero_grad(set_to_none=True)
        for (a,p),(b,q) in zip(first.named_parameters(),second.named_parameters()):
            self.assertEqual(a,b); torch.testing.assert_close(p,q)
        self.assertEqual(before,base_hashes(first))
        self.assertEqual(result["microbatches"],4)
        self.assertEqual(sum(e=="forward_finished" for e,_ in events),4)
        self.assertEqual(sum(e=="backward_finished" for e,_ in events),4)
        self.assertEqual(sum(e=="optimizer_finished" for e,_ in events),1)

    def test_nonfinite_loss_stops_before_optimizer(self):
        import torch
        from lora_training_v1.core import optimizer_update,make_optimizer
        model=self.make_model(); opt,sched=make_optimizer(model,strict=False)
        with patch.object(opt,"step",side_effect=AssertionError("No step on invalid loss")):
            with self.assertRaises(FloatingPointError):
                optimizer_update(model,opt,sched,self.examples(),lambda *a,**k:None,1,strict=False,
                    loss_fn=lambda m,e:torch.tensor(float('nan'),requires_grad=True))

    def test_base_trainability_is_rejected_and_optimizer_has_only_adapters(self):
        from lora_training_v1.core import audit_trainable,make_optimizer
        model=self.make_model()
        named,_=audit_trainable(model,strict=False)
        opt,_=make_optimizer(model,strict=False)
        self.assertEqual({id(p) for g in opt.param_groups for p in g["params"]},{id(p) for p in named.values()})
        model.proj.base.weight.requires_grad_(True)
        with self.assertRaises(ValueError): audit_trainable(model,strict=False)

    def test_real_loss_interface_handles_padded_768_without_supervising_padding(self):
        import torch
        from lora_training_v1.core import RightPaddedForward
        from training_protocol_v1.data import completion_loss
        class Dummy(torch.nn.Module):
            def forward(self,input_ids,attention_mask,**kwargs):
                self.ids=input_ids; self.mask=attention_mask; self.positions=kwargs["logits_to_keep"]
                return SimpleNamespace(logits=torch.zeros(1,len(self.positions),11))
        model=Dummy()
        example={"prompt_length":4,"sequence_length":7,"input_ids":[1,2,3,4,5,6,10],
                 "labels":[-100,-100,-100,-100,5,6,10],"attention_mask":[1]*7}
        before=copy.deepcopy(example)
        loss=completion_loss(RightPaddedForward(model,768,10),example)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(tuple(model.ids.shape),(1,768))
        self.assertEqual(model.mask[0,7:].sum().item(),0)
        self.assertTrue((model.ids[0,7:]==10).all())
        self.assertEqual(model.positions.tolist(),[3,4,5])
        self.assertEqual(example,before)

    def test_checkpoint_roundtrip_is_adapter_only_and_feasibility_save_forbidden(self):
        import torch
        from safetensors.torch import load_file
        from lora_training_v1.core import make_optimizer,optimizer_update,save_checkpoint,audit_trainable
        model=self.make_model(); opt,sched=make_optimizer(model,strict=False)
        optimizer_update(model,opt,sched,self.examples(),lambda *a,**k:None,1,strict=False,loss_fn=self.loss)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                save_checkpoint(model,opt,sched,temp,1,0,[],{},"feasibility",strict=False)
            self.assertEqual(list(Path(temp).iterdir()),[])
            path=save_checkpoint(model,opt,sched,temp,1,0,["a","b"],{"test":"synthetic"},"full_pilot",strict=False)
            tensors=load_file(str(path/"adapter.safetensors"))
            named,_=audit_trainable(model,strict=False)
            self.assertEqual(set(tensors),set(named))
            for name,value in tensors.items(): torch.testing.assert_close(value,named[name])
            state=torch.load(path/"trainer_state.pt",weights_only=False)
            self.assertEqual(state["optimizer_step"],1)
            self.assertEqual(state["data_order"],["a","b"])
            self.assertIn("python_rng",state); self.assertIn("numpy_rng",state); self.assertIn("torch_rng",state)
            self.assertEqual(len(state["optimizer"]["state"]),2)
            with self.assertRaises(ValueError):
                save_checkpoint(model,opt,sched,temp,1,0,[],{},"full_pilot",strict=False)


@unittest.skipUnless((OUTPUT/"assessment.json").exists(),"Feasibility run not complete")
class MeasurementTests(unittest.TestCase):
    def test_bound_preservation_and_report_reproduction(self):
        from lora_training_v1.preflight import check
        from lora_training_v1.report import build_report
        check()
        text,summary=build_report()
        self.assertEqual((OUTPUT/"M2_FEASIBILITY_REPORT.md").read_text(),text)
        self.assertEqual(json.loads((OUTPUT/"assessment.json").read_text()),summary)
        self.assertLessEqual(summary["completed_optimizer_updates"],3)
        self.assertEqual(summary["trained_checkpoints_saved"],0)
        self.assertFalse(summary["full_pilot_started"])


if __name__=="__main__": unittest.main()
