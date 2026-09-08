"""Label-masked review, frozen judgments, then descriptive paired comparison."""

import argparse
from datetime import datetime, timezone
import json
import random
import statistics

from pilot_holdout_v1.protocol import OUTPUT,ROOT,MODELS,opened_rows,read_jsonl,write_once
from pilot_holdout_v1.experiment import check
from verifier import sha256_file


def masked_records(raw,examples,seed):
    shuffled=list(raw);random.Random(seed).shuffle(shuffled)
    packet=[];mapping={}
    for index,row in enumerate(shuffled,1):
        identity=f'R{index:02d}';example=examples[row['example_id']]
        packet.append({'review_id':identity,'example_id':example['id'],
            'formal_statement':example['formal_statement'],'informal_proof':example['informal_proof'],
            'reference_proof_body':example['proof_body'],'proof_body':row['proof_body'],
            'proof_body_sha256':row['proof_body_sha256'],'verification':row['verification']})
        mapping[identity]=row['key']
    return packet,mapping


def make_packet():
    if (OUTPUT/'review_packet.jsonl').exists(): raise ValueError('Review packet already exists')
    frozen,_,raw=check()
    examples={r['id']:r for r in opened_rows()}
    packet,mapping=masked_records(raw,examples,frozen['review_shuffle_seed'])
    with (OUTPUT/'review_packet.jsonl').open('x') as stream:
        for row in packet: stream.write(json.dumps(row,ensure_ascii=False)+'\n')
    write_once(OUTPUT/'review_mapping.json',mapping)
    write_once(OUTPUT/'packet_manifest.json',{'records':12,'review_shuffle_seed':frozen['review_shuffle_seed'],
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'packet_sha256':sha256_file(OUTPUT/'review_packet.jsonl'),
        'mapping_sha256':sha256_file(OUTPUT/'review_mapping.json'),
        'masked_fields':['model_key','checkpoint','latency','usage','constraint_trace','requested_proof_steps']})
    print('Created twelve label-masked review records; do not read the mapping before freezing judgments')


def validate_reviews(packet,record):
    decisions=record['reviews']
    by_id={r['review_id']:r for r in decisions}
    if len(by_id)!=len(decisions) or set(by_id)!={r['review_id'] for r in packet}:
        raise ValueError('Every unique masked record requires a judgment')
    if record['reviewer']!='assistant' or record['human_reviewed'] is not False:
        raise ValueError('Reviewer provenance differs')
    for row in packet:
        review=by_id[row['review_id']]
        if review['proof_body_sha256']!=row['proof_body_sha256'] or not review['reason'].strip():
            raise ValueError('Unbound/empty review')
        if review['status'] not in ('FAITHFUL','UNFAITHFUL','NOT_ESTABLISHED'): raise ValueError('Invalid judgment')
        if row['verification']['status']!='PASS' and review['status']!='NOT_ESTABLISHED':
            raise ValueError('Failed proofs cannot establish a complete argument')
    return by_id


def freeze_reviews():
    if (OUTPUT/'review_freeze.json').exists(): raise ValueError('Judgments already frozen')
    check()
    manifest=json.loads((OUTPUT/'packet_manifest.json').read_text())
    if manifest['packet_sha256']!=sha256_file(OUTPUT/'review_packet.jsonl') or manifest['mapping_sha256']!=sha256_file(OUTPUT/'review_mapping.json'):
        raise ValueError('Review material changed')
    packet=read_jsonl(OUTPUT/'review_packet.jsonl')
    record=json.loads((OUTPUT/'masked_reviews.json').read_text())
    if len(packet)!=12: raise ValueError('Twelve reviews required')
    validate_reviews(packet,record)
    write_once(OUTPUT/'review_freeze.json',{'status':'FROZEN_BEFORE_MODEL_LABEL_DISCLOSURE',
        'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'packet_sha256':manifest['packet_sha256'],'mapping_sha256':manifest['mapping_sha256'],
        'reviews_sha256':sha256_file(OUTPUT/'masked_reviews.json'),'reviews':12})
    print('Frozen all twelve mathematical judgments before model-label disclosure')


def attach_masked_reviews(raw,packet,mapping,decisions):
    by_id=validate_reviews(packet,decisions)
    if set(mapping)!={r['review_id'] for r in packet} or set(mapping.values())!={r['key'] for r in raw}:
        raise ValueError('Review mapping differs')
    by_key={mapping[identity]:review for identity,review in by_id.items()}
    result=[]
    for row in raw:
        review=by_key[row['key']]
        if review['proof_body_sha256']!=row['proof_body_sha256']: raise ValueError('Mapped proof hash differs')
        result.append(row|{'mathematical_argument_fidelity':review})
    return result


def metrics(rows):
    passed=lambda r:r['verification']['status']=='PASS'
    return {'rows':len(rows),'verified':sum(passed(r) for r in rows),
        'verified_mathematically_faithful':sum(passed(r) and r['mathematical_argument_fidelity']['status']=='FAITHFUL' for r in rows),
        'attempted_step_matches':sum(r['requested_proof_steps']['matched'] is True for r in rows),
        'verified_requested_steps':sum(passed(r) and r['requested_proof_steps']['matched'] is True for r in rows),
        'format_compliant':sum(r['format_compliance']['compliant'] for r in rows),
        'mean_latency_seconds':statistics.mean(r['generation_latency_seconds'] for r in rows),
        'completion_tokens':sum(r['usage']['completion_tokens'] for r in rows)}


def build_report():
    # The review freeze is required before this function accesses model labels.
    review_freeze=json.loads((OUTPUT/'review_freeze.json').read_text())
    if review_freeze['status']!='FROZEN_BEFORE_MODEL_LABEL_DISCLOSURE': raise ValueError('Review freeze required')
    for name,field in [('masked_reviews.json','reviews_sha256'),('review_packet.jsonl','packet_sha256'),('review_mapping.json','mapping_sha256')]:
        if sha256_file(OUTPUT/name)!=review_freeze[field]: raise ValueError('Frozen review changed')
    frozen,completion,raw=check()
    packet=read_jsonl(OUTPUT/'review_packet.jsonl')
    decisions=json.loads((OUTPUT/'masked_reviews.json').read_text())
    mapping=json.loads((OUTPUT/'review_mapping.json').read_text())
    examples={r['id']:r for r in opened_rows()}
    expected_packet,expected_mapping=masked_records(raw,examples,frozen['review_shuffle_seed'])
    if packet!=expected_packet or mapping!=expected_mapping: raise ValueError('Review packet not derived from saved outputs')
    assessed=attach_masked_reviews(raw,packet,mapping,decisions)
    by_model={model:metrics([r for r in assessed if r['model_key']==model]) for model in MODELS}
    delta={k:by_model['step18'][k]-by_model['base'][k] for k in ('verified','verified_mathematically_faithful','verified_requested_steps')}
    pairs=[]
    for theorem in sorted({r['theorem_id'] for r in assessed}):
        outcome={'theorem_id':theorem}
        for model in MODELS:
            group=[r for r in assessed if r['theorem_id']==theorem and r['model_key']==model]
            if len(group)!=2 or {r['argument_id'] for r in group}!={'A','B'}: raise ValueError('Expected correlated A/B pair')
            outcome[model]={'both_verified':all(r['verification']['status']=='PASS' for r in group),
                'both_verified_faithful':all(r['verification']['status']=='PASS' and r['mathematical_argument_fidelity']['status']=='FAITHFUL' for r in group),
                'both_verified_step_adherent':all(r['verification']['status']=='PASS' and r['requested_proof_steps']['matched'] is True for r in group),
                'bodies_differ':group[0]['proof_body']!=group[1]['proof_body']}
        pairs.append(outcome)
    improved=delta['verified_requested_steps']>0 and delta['verified']>=0 and delta['verified_mathematically_faithful']>=0
    summary={'experiment_id':'proofbridge-holdout-v1','checkpoint_step':18,
        'model_revision':frozen['model']['revision'],'selected_adapter_sha256':frozen['selected']['selected_adapter_sha256'],
        'by_model':by_model,'trained_minus_base_counts':delta,
        'proof_step_improvement_preserving_verification_and_fidelity':improved,'theorem_pairs':pairs,
        'generations':12,'optimizer_updates':0,'original_holdout_content_openings':1,'holdout_status':'CONSUMED',
        'checkpoint_selection_changed':False,'protocol_scope':'argument-conditioned only; 12-output user-approved scope',
        'review_method':'label-masked assistant judgments frozen before report label disclosure; not independent human review',
        'wall_seconds':completion['wall_seconds'],
        'limitations':['Six rows are three correlated theorem pairs in one historically exposed family.',
            'Fresh-instance transfer under this pipeline; no claim of wholly unseen mathematics or absence of pretraining contamination.',
            'No theorem-only control, so this comparison alone cannot establish causal informal-proof dependence.',
            'No further tuning or checkpoint selection may reuse this as an untouched holdout.']}
    conclusion=('Verified requested-step adherence improved while verification and mathematical fidelity were preserved.' if improved else
        'The holdout did not show improved verified requested-step adherence while preserving verification and mathematical fidelity.')
    lines=['# Holdout comparison v1','',conclusion,'',
        'The pinned base and preselected step-18 adapter each received the same six formal-theorem-plus-informal-proof inputs. Exactly 12 one-attempt generations used unchanged Prompt v2, constrained-v1 and the verifier, with a 256-token budget and no repair.','',
        '| Metric | Base | Step 18 | Difference |','| --- | ---: | ---: | ---: |']
    for label,key in [('Rocq verified','verified'),('Verified + mathematically faithful','verified_mathematically_faithful'),('Verified + requested steps','verified_requested_steps')]:
        a,b=by_model['base'][key],by_model['step18'][key]
        lines.append(f'| {label} | {a}/6 ({100*a/6:.1f}%) | {b}/6 ({100*b/6:.1f}%) | {b-a:+d} rows |')
    lines+=['','## All outputs','','| Model | Row | Rocq | Mathematical fidelity | Requested steps |','| --- | --- | --- | --- | --- |']
    for r in assessed:
        lines.append(f"| {r['model_key']} | {r['example_id']} | {r['verification']['status']} / {r['verification']['category']} | {r['mathematical_argument_fidelity']['status']} | {r['requested_proof_steps']['status']} |")
    lines+=['','## Theorem-pair outcomes','','| Theorem | Base: both variants verified and step-adherent | Step 18: both variants verified and step-adherent |','| --- | --- | --- |']
    for pair in pairs: lines.append(f"| {pair['theorem_id']} | {pair['base']['both_verified_step_adherent']} | {pair['step18']['both_verified_step_adherent']} |")
    lines+=['','The six rows are three correlated A/B pairs. Bodies changing between A and B is not sufficient: both proofs must verify and match their respective step contracts.','',
        '## Mathematical review','',
        'All twelve proofs were reviewed in shuffled records without model labels, latency, token traces or step scores. The judgments were hash-frozen before the mapping was disclosed for this report. These are assistant reviews, not independent human or guaranteed blind judgments; proof style can suggest model identity. Equivalent successful tactic implementations remain mathematically faithful; failed proofs are NOT_ESTABLISHED.','']
    for r in assessed: lines.append(f"- **{r['model_key']} {r['example_id']}**: {r['mathematical_argument_fidelity']['reason']}")
    lines+=['','## Custody and reproduction','',
        'The protocol and exact adapter hash were frozen before opening the reserved JSONL. Its contents were parsed once into a separate immutable snapshot. All six gold references were reverified before inference. Base and adapter parameters were hash-compared before/after inference; no weights, prompt, decoder, dataset, verifier or selected checkpoint changed. The original reserved source and historical QA details remain unchanged.','',
        'The holdout is now **consumed**. No result here changes the dev-selected step 18 or authorizes further tuning on these rows. The later user-approved 12-generation scope uses argument-conditioned inputs only; the older split-policy proposal also mentioned theorem-only controls, which were not run.','',
        f"Mean generation latency: base **{by_model['base']['mean_latency_seconds']:.2f}s**, step 18 **{by_model['step18']['mean_latency_seconds']:.2f}s**. Completion tokens: base **{by_model['base']['completion_tokens']}**, step 18 **{by_model['step18']['completion_tokens']}**. Total run wall time: **{completion['wall_seconds']:.2f}s**. Sequential worker order and host/cache state limit timing comparisons.",'',
        'From the repository root, replay without training, new generations or reopening the original reserved contents:','', '```sh',
        '.venv/bin/python -m pilot_holdout_v1.experiment check --replay-tokens --reverify',
        '.venv/bin/python -m pilot_holdout_v1.assessment report',
        '.venv/bin/python -m unittest discover -s tests -v','```','',
        'The one-time command was `.venv/bin/python -m pilot_holdout_v1.experiment run`; it refuses an existing run. `PROTOCOL.md` contains the full precommitted workflow. Machine-readable metrics, all assessed outputs, original raw prompts/token traces, masked reviews, reference evidence and consumed-holdout custody records are saved alongside this report.','', '## Limits','']
    lines+=['- '+s for s in summary['limitations']];lines.append('')
    return '\n'.join(lines),summary,assessed


def report():
    text,summary,rows=build_report()
    outputs={OUTPUT/'HOLDOUT_REPORT.md':text,OUTPUT/'summary.json':json.dumps(summary,indent=2)+'\n',
        OUTPUT/'assessed_results.jsonl':''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)}
    for path,body in outputs.items():
        if path.exists() and path.read_text()!=body: raise ValueError('Refusing to replace a different assessment')
        path.write_text(body)
    print(json.dumps(summary['by_model']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['packet','freeze-reviews','report'])
    action=parser.parse_args().action
    if action=='packet': make_packet()
    elif action=='freeze-reviews': freeze_reviews()
    else: report()
