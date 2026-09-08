"""Run once, then check saved training/checkpoint/dev evidence without inference."""

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
import time
import traceback

from baseline.dataset import digest
from baseline.protocol import extract_body
from baseline.setup_model import check_model
from pilot_v1.protocol import ROOT
from training_protocol_v1.data import config, epoch_order
from training_protocol_v1.experiment import check_protocol, dev_groups, messages
from verifier import sha256_file, verify

MODULE = ROOT/'pilot_training_v1'
OUTPUT = ROOT/'results/lora-pilot-v1'


def check_inputs():
    protected = json.loads((MODULE/'protected_inputs.json').read_text())['files_sha256']
    for name, expected in protected.items():
        if sha256_file(ROOT/name) != expected:
            raise ValueError(f'Protected historical file changed: {name}')
    check_protocol()
    return len(protected)


def fingerprints():
    paths = [p for p in MODULE.iterdir() if p.is_file() and p.name != 'release.json']
    paths.append(ROOT/'tests/test_pilot_training_v1.py')
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(paths)}


def run():
    if OUTPUT.exists():
        raise ValueError('Pilot exists: refusing resume, overwrite, automatic retry or additional updates')
    count = check_inputs()
    model = check_model()
    from lora_training_v1.core import train_pilot
    from lora_training_v1.telemetry import process_memory
    from pilot_training_v1.evaluation import CheckpointEvaluator
    saved = json.loads((ROOT/'results/pretraining-dev-v1/manifest.json').read_text())
    if model != saved['model']:
        raise ValueError('Pinned base differs')
    OUTPUT.mkdir(parents=True)
    manifest = {'experiment_id':'proofbridge-lora-pilot-v1', 'started_at_utc':datetime.now(timezone.utc).isoformat(),
        'model':model, 'configuration':config(), 'generation_settings':saved['generation_settings'],
        'effective_generation_config':saved['effective_generation_config'], 'fingerprints':fingerprints(),
        'protected_historical_files':count, 'initialization':'Fresh pinned base and frozen seeded A / zero B; no adapter load',
        'feasibility_updates_reused':False, 'expected_optimizer_updates':60,'expected_dev_generations':60,
        'step_zero_source':'results/pretraining-dev-v1/assessed_results.jsonl',
        'step_zero_sha256':sha256_file(ROOT/'results/pretraining-dev-v1/assessed_results.jsonl'),
        'holdout_content_reads':0, 'selection_rule':'unchanged training_protocol_v1.scoring.selection_score',
        'review_policy':'All 60 saved candidates receive hash-bound assistant reviews before final selection; no review alters training budget.',
        'inference_rng_policy':'Save/restore Python, NumPy, Torch state around each six-row evaluation',
        'vocabulary_cache_policy':'Fresh once per pilot; shared across dev attempts; no reference prewarming'}
    (OUTPUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    started = time.perf_counter()
    evaluator = CheckpointEvaluator(OUTPUT)
    try:
        train_pilot(OUTPUT/'training', evaluator)
        if evaluator.completed_steps != list(range(6,61,6)):
            raise ValueError('Missing checkpoint evaluations')
        check_inputs()
        if check_model() != model:
            raise ValueError('Original model files changed')
        # Reviews may be added while training runs; bind raw evidence separately.
        paths = [OUTPUT/'manifest.json'] + list((OUTPUT/'training').rglob('*'))
        paths += [p for d in OUTPUT.glob('dev-step-*') for p in d.iterdir()
                  if p.name in ('raw_generations.jsonl','attempts.jsonl','completion.json')]
        completion = {'status':'COMPLETE', 'optimizer_updates':60,'dev_generations':60,
            'base_weights_unchanged':True,'base_model_files_unchanged':True,
            'wall_seconds':time.perf_counter()-started,'final_process_memory':process_memory(),
            'finished_at_utc':datetime.now(timezone.utc).isoformat(),
            'files_sha256':{str(p.relative_to(OUTPUT)):sha256_file(p) for p in sorted(paths) if p.is_file()}}
        (OUTPUT/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
        print('COMPLETE: 60 updates, 10 checkpoints, 60 dev outputs; selection awaits reviews; holdout sealed',flush=True)
    except BaseException as error:
        (OUTPUT/'interruption.json').write_text(json.dumps({'status':'INTERRUPTED','error':str(error),
            'traceback':traceback.format_exc(),'completed_checkpoint_steps':evaluator.completed_steps,
            'no_automatic_retry':True,'wall_seconds':time.perf_counter()-started},indent=2)+'\n')
        raise


def read_rows(folder):
    return [json.loads(line) for line in (folder/'raw_generations.jsonl').read_text().splitlines()]


def check(replay_tokens=False, reverify=False):
    check_inputs()
    manifest = json.loads((OUTPUT/'manifest.json').read_text())
    completion = json.loads((OUTPUT/'completion.json').read_text())
    if manifest['fingerprints'] != fingerprints() or manifest['configuration'] != config():
        raise ValueError('Implementation or frozen configuration drift')
    for name, expected in completion['files_sha256'].items():
        if sha256_file(OUTPUT/name) != expected:
            raise ValueError(f'Run evidence changed: {name}')
    if completion['status'] != 'COMPLETE' or (OUTPUT/'interruption.json').exists():
        raise ValueError('Interrupted run cannot be selected')
    events = [json.loads(line) for line in (OUTPUT/'training/training_metrics.jsonl').read_text().splitlines()]
    updates = [e for e in events if e['event']=='optimizer_finished']
    starts = [e for e in events if e['event']=='optimizer_started']
    if [e['update'] for e in updates] != list(range(1,61)) or [e['update'] for e in starts] != list(range(1,61)):
        raise ValueError('Training update budget/order differs')
    if sum(e['event']=='forward_finished' for e in events)!=240 or sum(e['event']=='backward_finished' for e in events)!=240:
        raise ValueError('Microbatch budget differs')
    for u in updates:
        epoch,index=divmod(u['update']-1,6)
        if u['example_ids'] != epoch_order(epoch)[4*index:4*index+4] or u['microbatches']!=4 or u['learning_rate']!=0.0002:
            raise ValueError('Frozen training order/accumulation/learning rate differs')
        if not all(math.isfinite(v) for v in u['losses']) or u['trainable_audit']['trainable_parameters']!=540672:
            raise ValueError('Invalid loss or trainable weights')
    from training_protocol_v1.scoring import step_adherence, mathematical_structure
    from prompt_v2.protocol import format_compliance
    groups={g['theorem_id']:g for g in dev_groups()}
    tokenizer=vocabulary=None
    if replay_tokens:
        from constrained_v1.experiment import tokenizer_only
        from constrained_v1.tokens import Vocabulary
        tokenizer=tokenizer_only(); vocabulary=Vocabulary(tokenizer)
    all_rows=[]
    for step in range(6,61,6):
        folder=OUTPUT/f'dev-step-{step:04d}'
        checkpoint=OUTPUT/f'training/step-{step:04d}'
        rows=read_rows(folder)
        evidence=json.loads((folder/'completion.json').read_text())
        if evidence['attempts']!=6 or sorted(r['dev_example_id'] for r in rows)!=config()['data']['dev_ids']:
            raise ValueError('Expected precisely six frozen dev rows per checkpoint')
        for name,expected in evidence['files_sha256'].items():
            if sha256_file(folder/name)!=expected: raise ValueError('Checkpoint dev evidence differs')
        journal=[json.loads(line) for line in (folder/'attempts.jsonl').read_text().splitlines()]
        if len(journal)!=12 or any(e['attempt']!=1 for e in journal): raise ValueError('Invalid attempt journal')
        for event in ('started','finished'):
            if Counter(e['key'] for e in journal if e['event']==event)!=Counter({r['key']:1 for r in rows}):
                raise ValueError('Attempt retried or missing')
        for r in rows:
            group=groups[r['theorem_id']]
            example=group['arguments'][r['condition'][-1]]
            if (r['key']!=f"pilot_training_v1:step-{step:04d}:{example['id']}" or r['optimizer_step']!=step
                or r['formal_statement']!=example['formal_statement'] or r['dev_example_id']!=example['id']
                or r['prompt']['messages']!=messages(group,r['condition']) or r['attempt']!=1 or r['repair_attempts']!=0
                or r['checkpoint_adapter_sha256']!=sha256_file(checkpoint/'adapter.safetensors')
                or r['proof_body_sha256']!=digest(r['proof_body'])
                or (r['proof_body'],r['output_extraction'])!=extract_body(r['generated_text'])):
                raise ValueError('Changed dev generation/input/checkpoint')
            if (r['requested_proof_steps']!=step_adherence(example,r['proof_body'],r['condition'])
                or r['mathematical_structure_review_aid']!=mathematical_structure(r['formal_statement'],r['proof_body'])
                or r['format_compliance']!=format_compliance(r['formal_statement'],r['generated_text'])):
                raise ValueError('Assessment drift')
            p,n=len(r['prompt']['input_token_ids']),len(r['completion_token_ids'])
            if n>256 or r['usage']!={'prompt_tokens':p,'completion_tokens':n,'total_tokens':p+n}:
                raise ValueError('Token budget differs')
            if tokenizer:
                rendered=tokenizer.apply_chat_template(r['prompt']['messages'],tokenize=False,add_generation_prompt=True)
                if r['prompt']['rendered_text']!=rendered or r['prompt']['sha256']!=digest(rendered) or r['prompt']['input_token_ids']!=tokenizer.encode(rendered,add_special_tokens=False):
                    raise ValueError('Prompt token replay differs')
                from constrained_v1.grammar import initial, feed
                state=initial(r['formal_statement'])
                for position,token in enumerate(r['completion_token_ids']):
                    if token not in vocabulary.allowed(state): raise ValueError('Disallowed constrained token')
                    if token==vocabulary.eos_id:
                        if position!=n-1: raise ValueError('Premature EOS')
                    else: state=feed(state,vocabulary.pieces[token])
                if tokenizer.decode(r['completion_token_ids'],skip_special_tokens=True,clean_up_tokenization_spaces=False)!=r['generated_text']:
                    raise ValueError('Output token replay differs')
            if reverify:
                actual=asdict(verify(r['formal_statement'],r['proof_body'],timeout=10.0))
                for field in ('status','category','source_sha256','kernel_checked','assumptions_checked'):
                    if actual[field]!=r['verification'][field]: raise ValueError('Rocq replay differs')
        all_rows.extend(rows)
    release=MODULE/'release.json'
    if release.exists():
        for name,expected in json.loads(release.read_text())['files_sha256'].items():
            if sha256_file(ROOT/name)!=expected: raise ValueError(f'Frozen pilot evidence changed: {name}')
    return manifest,completion,updates,all_rows


def freeze():
    if (MODULE/'release.json').exists(): raise ValueError('Already frozen')
    from pilot_training_v1.report import build_report
    text,summary,assessed=build_report()
    if (OUTPUT/'LORA_PILOT_REPORT.md').read_text()!=text or json.loads((OUTPUT/'summary.json').read_text())!=summary:
        raise ValueError('Stale final report')
    validation=json.loads((OUTPUT/'validation.json').read_text())
    if validation['status']!='PASS' or validation['dev_token_replays']!=60 or validation['dev_rocq_replays']!=60:
        raise ValueError('Final validation required')
    paths=[p for d in (MODULE,OUTPUT) for p in d.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    paths.append(ROOT/'tests/test_pilot_training_v1.py')
    (MODULE/'release.json').write_text(json.dumps({'status':'FROZEN','release_id':'proofbridge-lora-pilot-v1',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),'selected_step':summary['selected_step'],
        'files_sha256':{str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(paths)}},indent=2)+'\n')
    print('Frozen pilot, all checkpoints, dev reviews and selection; holdout sealed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','check','freeze'])
    parser.add_argument('--replay-tokens',action='store_true')
    parser.add_argument('--reverify',action='store_true')
    args=parser.parse_args()
    if args.action=='run': run()
    elif args.action=='freeze': freeze()
    else:
        check(args.replay_tokens,args.reverify)
        print('PASS: 60 updates, 60 dev outputs, saved checkpoint/input integrity; no new inference')
