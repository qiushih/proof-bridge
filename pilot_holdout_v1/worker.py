"""Exactly six inference attempts in one fresh model process; never train."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json

from baseline.dataset import digest
from baseline.protocol import extract_body
from baseline.setup_model import check_model
from constrained_v1.experiment import generate
from constrained_v1.tokens import Vocabulary
from lora_training_v1.core import audit_trainable, base_hashes, tensor_digest
from pilot_holdout_v1.protocol import OUTPUT, ROOT, MODELS, check_protocol, opened_rows, write_once
from prompt_v2.experiment import Inference, load_frozen
from prompt_v2.protocol import build_messages, format_compliance
from training_protocol_v1.adaptation import attach_for_future_training
from training_protocol_v1.scoring import step_adherence
from verifier import verify, sha256_file


def load_adapter(model, tensors, strict=True):
    import torch
    named,_=audit_trainable(model,strict=strict)
    if set(tensors)!=set(named): raise ValueError('Adapter tensor keys differ')
    for name,target in named.items():
        source=tensors[name]
        if source.shape!=target.shape or source.dtype!=target.dtype or not torch.isfinite(source).all():
            raise ValueError('Adapter shape/dtype/value differs')
    with torch.no_grad():
        for name,target in named.items(): target.copy_(tensors[name])
    hashes={name:tensor_digest(p) for name,p in named.items()}
    if hashes!={name:tensor_digest(p) for name,p in tensors.items()}: raise ValueError('Adapter load differs')
    return named,hashes


def run(model_key):
    frozen=check_protocol()
    if model_key not in MODELS: raise ValueError('Only the fixed two models may be evaluated')
    folder=OUTPUT/model_key
    folder.mkdir(exist_ok=False)
    rows=opened_rows()
    engine=Inference(frozen['generation_settings'])
    if engine.lock!=frozen['model'] or engine.generation.to_dict()!=frozen['effective_generation_config']:
        raise ValueError('Pinned model or generation configuration differs')
    before=base_hashes(engine.model)
    named,adapter_hashes={},{}
    if model_key=='step18':
        from safetensors.torch import load_file
        engine.torch.manual_seed(frozen['generation_settings']['seed'])
        attach_for_future_training(engine.model)
        tensors=load_file(str(ROOT/frozen['selected']['selected_adapter_path']))
        named,adapter_hashes=load_adapter(engine.model,tensors)
        del tensors
    engine.model.requires_grad_(False)
    engine.model.eval()
    if base_hashes(engine.model)!=before: raise ValueError('Adapter attachment changed base weights')
    vocabulary=Vocabulary(engine.tokenizer)
    _,prompt=load_frozen()
    write_once(folder/'manifest.json',{'model_key':model_key,'model':engine.lock,'runtime':engine.runtime,
        'effective_generation_config':engine.generation.to_dict(),'expected_attempts':6,
        'selected_adapter_sha256':frozen['selected']['selected_adapter_sha256'] if named else None,
        'base_parameter_sha256_before':before,'adapter_parameter_sha256_before':adapter_hashes,
        'trainable_parameters_during_inference':sum(p.numel() for p in engine.model.parameters() if p.requires_grad),
        'optimizer_updates':0,'fresh_vocabulary':True})
    with (folder/'raw_generations.jsonl').open('x') as output, (folder/'attempts.jsonl').open('x') as journal:
        for example in rows:
            key=f"holdout_v1:{model_key}:{example['id']}"
            journal.write(json.dumps({'key':key,'event':'started','attempt':1,'at_utc':datetime.now(timezone.utc).isoformat()})+'\n'); journal.flush()
            messages=build_messages(prompt,example,'theorem_and_informal')
            text,generated=generate(engine,vocabulary,example['formal_statement'],messages)
            body,extraction=extract_body(text)
            verification=asdict(verify(example['formal_statement'],body,timeout=10.0))
            row={'key':key,'model_key':model_key,'example_id':example['id'],'theorem_id':example['theorem_id'],
                'argument_id':example['argument_id'],'condition':'theorem_and_informal',
                'formal_statement':example['formal_statement'],'attempt':1,'repair_attempts':0,
                'generated_text':text,'proof_body':body,'proof_body_sha256':digest(body),'output_extraction':extraction,
                'verification':verification,'failure_category':None if verification['status']=='PASS' else verification['category'],
                'format_compliance':format_compliance(example['formal_statement'],text),
                'requested_proof_steps':step_adherence(example,body,'argument_'+example['argument_id'])} | generated
            output.write(json.dumps(row,ensure_ascii=False)+'\n'); output.flush()
            journal.write(json.dumps({'key':key,'event':'finished','attempt':1})+'\n'); journal.flush()
            # No scores or proof text are printed before the masked review.
            print('Completed planned inference attempt',flush=True)
            if generated['generation_error']: raise RuntimeError('Recorded runtime error; no retry')
    after=base_hashes(engine.model)
    adapter_after={name:tensor_digest(p) for name,p in named.items()}
    if after!=before or adapter_after!=adapter_hashes: raise ValueError('Inference changed model weights')
    check_protocol()
    if check_model()!=engine.lock: raise ValueError('Base model files changed')
    write_once(folder/'completion.json',{'status':'COMPLETE','attempts':6,'optimizer_updates':0,
        'base_parameter_sha256_after':after,'adapter_parameter_sha256_after':adapter_after,
        'base_and_adapter_weights_unchanged':True,
        'files_sha256':{p.name:sha256_file(p) for p in folder.iterdir() if p.is_file()}})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_key',choices=MODELS)
    run(parser.parse_args().model_key)
