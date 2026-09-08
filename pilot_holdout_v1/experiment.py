"""One holdout opening and twelve total attempts; later commands replay only."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import subprocess
import sys
import time
import traceback

from baseline.dataset import digest
from baseline.protocol import extract_body
from pilot_holdout_v1.protocol import (ROOT,MODULE,OUTPUT,FROZEN,MODELS,check_protocol,freeze_protocol,
    open_original_once,opened_rows,read_jsonl,write_once,check_inputs)
from prompt_v2.experiment import load_frozen
from prompt_v2.protocol import build_messages,format_compliance
from training_protocol_v1.scoring import step_adherence
from verifier import sha256_file,verify


def run():
    if OUTPUT.exists(): raise ValueError('Holdout run exists: no reopening, resume, replacement or retry')
    frozen=check_protocol()
    OUTPUT.mkdir(parents=True)
    started=time.perf_counter()
    try:
        rows=open_original_once()
        references=[]
        for row in rows:
            actual=asdict(verify(row['formal_statement'],row['proof_body'],timeout=10.0))
            if actual['status']!='PASS' or actual['source_sha256']!=row['verification']['source_sha256']:
                raise ValueError('Fixed holdout reference verification failed; stop without editing')
            references.append({'example_id':row['id'],'verification':actual})
        write_once(OUTPUT/'reference_verification.json',{'status':'PASS','references':references})
        for model in MODELS:
            with (OUTPUT/(model+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-u','-m','pilot_holdout_v1.worker',model],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode: raise RuntimeError('Inference worker stopped; retain evidence without retry')
            print('One of the two fixed model evaluations completed; scores withheld for review',flush=True)
        check_inputs()
        paths=[p for p in OUTPUT.rglob('*') if p.is_file()]
        write_once(OUTPUT/'completion.json',{'status':'COMPLETE','attempts':12,'optimizer_updates':0,
            'protocol_sha256':sha256_file(FROZEN),'finished_at_utc':datetime.now(timezone.utc).isoformat(),
            'wall_seconds':time.perf_counter()-started,'original_content_openings':1,
            'files_sha256':{str(p.relative_to(OUTPUT)):sha256_file(p) for p in sorted(paths)}})
        print('COMPLETE: exactly twelve one-attempt outputs; holdout consumed; zero weight updates',flush=True)
    except BaseException as error:
        write_once(OUTPUT/'interruption.json',{'status':'INTERRUPTED','error':str(error),'traceback':traceback.format_exc(),
            'no_retry':True,'wall_seconds':time.perf_counter()-started})
        raise


def check(replay_tokens=False,reverify=False):
    frozen=check_protocol()
    completion=json.loads((OUTPUT/'completion.json').read_text())
    if completion['status']!='COMPLETE' or completion['attempts']!=12 or completion['optimizer_updates']!=0 or (OUTPUT/'interruption.json').exists():
        raise ValueError('Incomplete or out-of-scope evaluation')
    if completion['protocol_sha256']!=sha256_file(FROZEN): raise ValueError('Protocol binding changed')
    for name,expected in completion['files_sha256'].items():
        if sha256_file(OUTPUT/name)!=expected: raise ValueError(f'Changed holdout evidence: {name}')
    examples={r['id']:r for r in opened_rows()}
    _,prompt=load_frozen()
    tokenizer=vocabulary=None
    if replay_tokens:
        from constrained_v1.experiment import tokenizer_only
        from constrained_v1.tokens import Vocabulary
        tokenizer=tokenizer_only();vocabulary=Vocabulary(tokenizer)
    all_rows=[]
    for model in MODELS:
        folder=OUTPUT/model
        rows=read_jsonl(folder/'raw_generations.jsonl')
        manifest=json.loads((folder/'manifest.json').read_text())
        done=json.loads((folder/'completion.json').read_text())
        if [r['example_id'] for r in rows]!=frozen['example_ids'] or done['attempts']!=6:
            raise ValueError('Six fixed rows required for each model')
        if manifest['model']!=frozen['model'] or manifest['effective_generation_config']!=frozen['effective_generation_config']:
            raise ValueError('Model or generation settings differ')
        if manifest['trainable_parameters_during_inference']!=0 or done['optimizer_updates']!=0:
            raise ValueError('Inference must not train')
        expected_adapter=frozen['selected']['selected_adapter_sha256'] if model=='step18' else None
        if manifest['selected_adapter_sha256']!=expected_adapter: raise ValueError('Different selected checkpoint')
        if manifest['base_parameter_sha256_before']!=done['base_parameter_sha256_after'] or manifest['adapter_parameter_sha256_before']!=done['adapter_parameter_sha256_after']:
            raise ValueError('Weights changed during inference')
        for name,expected in done['files_sha256'].items():
            if sha256_file(folder/name)!=expected: raise ValueError('Per-model evidence differs')
        events=read_jsonl(folder/'attempts.jsonl')
        if len(events)!=12 or any(e['attempt']!=1 for e in events): raise ValueError('Incorrect attempt count')
        for event in ('started','finished'):
            if Counter(e['key'] for e in events if e['event']==event)!=Counter({r['key']:1 for r in rows}): raise ValueError('Missing/repeated attempt')
        for row in rows:
            example=examples[row['example_id']]
            if (row['key']!=f"holdout_v1:{model}:{example['id']}" or row['model_key']!=model
                or row['condition']!='theorem_and_informal' or row['formal_statement']!=example['formal_statement']
                or row['theorem_id']!=example['theorem_id'] or row['argument_id']!=example['argument_id']
                or row['prompt']['messages']!=build_messages(prompt,example,'theorem_and_informal')
                or row['attempt']!=1 or row['repair_attempts']!=0
                or row['proof_body_sha256']!=digest(row['proof_body'])
                or (row['proof_body'],row['output_extraction'])!=extract_body(row['generated_text'])):
                raise ValueError('Candidate/input drift')
            if row['requested_proof_steps']!=step_adherence(example,row['proof_body'],'argument_'+example['argument_id']) or row['format_compliance']!=format_compliance(example['formal_statement'],row['generated_text']):
                raise ValueError('Step/format assessment drift')
            p,n=len(row['prompt']['input_token_ids']),len(row['completion_token_ids'])
            if n>256 or row['usage']!={'prompt_tokens':p,'completion_tokens':n,'total_tokens':p+n}: raise ValueError('Changed token budget')
            if tokenizer:
                rendered=tokenizer.apply_chat_template(row['prompt']['messages'],tokenize=False,add_generation_prompt=True)
                if row['prompt']['rendered_text']!=rendered or row['prompt']['sha256']!=digest(rendered) or row['prompt']['input_token_ids']!=tokenizer.encode(rendered,add_special_tokens=False):
                    raise ValueError('Prompt token replay differs')
                from constrained_v1.grammar import initial,feed
                state=initial(row['formal_statement'])
                for index,token in enumerate(row['completion_token_ids']):
                    if token not in vocabulary.allowed(state): raise ValueError('Disallowed constrained token')
                    if token==vocabulary.eos_id:
                        if index!=n-1: raise ValueError('Early EOS')
                    else: state=feed(state,vocabulary.pieces[token])
                if tokenizer.decode(row['completion_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=row['generated_text']:
                    raise ValueError('Completion token replay differs')
            if reverify:
                actual=asdict(verify(row['formal_statement'],row['proof_body'],timeout=10.0))
                for field in ('status','category','source_sha256','kernel_checked','assumptions_checked'):
                    if actual[field]!=row['verification'][field]: raise ValueError('Rocq replay differs')
        all_rows.extend(rows)
    if (MODULE/'release.json').exists():
        for name,expected in json.loads((MODULE/'release.json').read_text())['files_sha256'].items():
            if sha256_file(ROOT/name)!=expected: raise ValueError(f'Frozen holdout release changed: {name}')
    return frozen,completion,all_rows


def freeze_results():
    if (MODULE/'release.json').exists(): raise ValueError('Already frozen')
    from pilot_holdout_v1.assessment import build_report
    text,summary,assessed=build_report()
    if (OUTPUT/'HOLDOUT_REPORT.md').read_text()!=text or json.loads((OUTPUT/'summary.json').read_text())!=summary:
        raise ValueError('Report/summary differ')
    if read_jsonl(OUTPUT/'assessed_results.jsonl')!=assessed: raise ValueError('Stale assessed rows')
    validation=json.loads((OUTPUT/'validation.json').read_text())
    if validation['status']!='PASS' or validation['token_replays']!=12 or validation['rocq_replays']!=12:
        raise ValueError('Full validation required')
    paths=[p for d in (MODULE,OUTPUT) for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths.append(ROOT/'tests/test_pilot_holdout_v1.py')
    write_once(MODULE/'release.json',{'status':'FROZEN','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'holdout_status':'CONSUMED','training_updates':0,'generations':12,'checkpoint_selection_changed':False,
        'files_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(paths)}})
    print('FROZEN: twelve assessed outputs; holdout consumed; original checkpoint selection preserved')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['freeze-protocol','run','check','freeze-results'])
    parser.add_argument('--replay-tokens',action='store_true');parser.add_argument('--reverify',action='store_true')
    args=parser.parse_args()
    if args.action=='freeze-protocol': freeze_protocol()
    elif args.action=='run': run()
    elif args.action=='freeze-results': freeze_results()
    else:
        check(args.replay_tokens,args.reverify)
        print('PASS: twelve saved outputs and weight hashes checked; no scores disclosed or new inference')
