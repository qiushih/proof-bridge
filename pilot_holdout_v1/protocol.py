"""Frozen inputs and the single authorized content-opening boundary."""

from datetime import datetime, timezone
import json

from baseline.setup_model import check_model
from pilot_v1.protocol import ROOT, DATA, MANIFEST, audit_rows
from verifier import sha256_file

MODULE=ROOT/'pilot_holdout_v1'
FROZEN=MODULE/'frozen.json'
OUTPUT=ROOT/'results/holdout-v1'
RESERVED=DATA/'reserved/holdout.jsonl'
SELECTED=ROOT/'results/lora-pilot-v1/selected_checkpoint.json'
MODELS=('base','step18')


def write_once(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False); stream.write('\n')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def expected_ids():
    return json.loads(MANIFEST.read_text())['partitions']['holdout']['example_ids']


def check_inputs():
    protected=json.loads((MODULE/'protected_inputs.json').read_text())['files_sha256']
    for name,expected in protected.items():
        if sha256_file(ROOT/name)!=expected: raise ValueError(f'Historical artifact changed: {name}')
    selected=json.loads(SELECTED.read_text())
    if selected['selected_step']!=18 or selected['selected_adapter_sha256']!='004eac24b10840832b2babc290e535f91a5a887d68db5977f58e812c8e57e6f0':
        raise ValueError('Preselected checkpoint differs')
    if sha256_file(ROOT/selected['selected_adapter_path'])!=selected['selected_adapter_sha256']:
        raise ValueError('Selected adapter changed')
    release=json.loads((ROOT/'pilot_training_v1/release.json').read_text())
    if release['status']!='FROZEN' or release['selected_step']!=18: raise ValueError('Pilot selection must be frozen first')
    return len(protected)


def fingerprints():
    paths=[p for p in MODULE.iterdir() if p.is_file() and p.name not in ('frozen.json','release.json')]
    paths.append(ROOT/'tests/test_pilot_holdout_v1.py')
    return {str(p.relative_to(ROOT)):sha256_file(p) for p in sorted(paths)}


def freeze_protocol():
    if FROZEN.exists() or OUTPUT.exists(): raise ValueError('Protocol/run exists; no overwrite')
    check_inputs()
    preflight=json.loads((MODULE/'preflight.json').read_text())
    if preflight['status']!='PASS' or preflight['holdout_content_reads']!=0:
        raise ValueError('Passing unopened preflight required')
    model=check_model()
    write_once(FROZEN,{'status':'FROZEN_BEFORE_HOLDOUT_OPENING','frozen_at_utc':datetime.now(timezone.utc).isoformat(),
        'fingerprints':fingerprints(),'model':model,'selected':json.loads(SELECTED.read_text()),
        'selected_checkpoint_record_sha256':sha256_file(SELECTED),
        'holdout_sha256':sha256_file(RESERVED),'example_ids':expected_ids(),
        'models':list(MODELS),'conditions':['theorem_and_informal'],'expected_generations':12,
        'generation_settings':json.loads((ROOT/'baseline/config.json').read_text())['generation'],
        'effective_generation_config':json.loads((ROOT/'results/pretraining-dev-v1/manifest.json').read_text())['effective_generation_config'],
        'review_shuffle_seed':53017,'holdout_content_reads':0,'training_or_weight_updates':0,
        'scope_authorization':'User approved the later 12 argument-conditioned comparison; older theorem-only proposal is not executed.'})
    print('FROZEN: protocol and step-18 adapter fixed; holdout contents remain unopened')


def check_protocol():
    check_inputs()
    record=json.loads(FROZEN.read_text())
    if record['fingerprints']!=fingerprints() or record['status']!='FROZEN_BEFORE_HOLDOUT_OPENING':
        raise ValueError('Holdout protocol changed after freezing')
    if sha256_file(RESERVED)!=record['holdout_sha256'] or sha256_file(SELECTED)!=record['selected_checkpoint_record_sha256']:
        raise ValueError('Held input or selected checkpoint changed')
    return record


def open_original_once():
    frozen=check_protocol()
    marker=OUTPUT/'custody.json'
    # Exclusive custody record precedes the sole original content read.
    write_once(marker,{'status':'UNSEALED_CONSUMED','opened_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':sha256_file(FROZEN),'protocol_frozen_at_utc':frozen['frozen_at_utc'],
        'original_sha256':frozen['holdout_sha256'],'authorized_original_content_openings':1,
        'future_tuning_requires_new_holdout':True})
    text=RESERVED.read_text()
    with (OUTPUT/'opened_holdout.jsonl').open('x') as stream: stream.write(text)
    return opened_rows()


def opened_rows():
    frozen=check_protocol()
    custody=json.loads((OUTPUT/'custody.json').read_text())
    if custody['status']!='UNSEALED_CONSUMED' or custody['opened_at_utc']<=frozen['frozen_at_utc']:
        raise ValueError('Opening must follow protocol freeze')
    path=OUTPUT/'opened_holdout.jsonl'
    if sha256_file(path)!=frozen['holdout_sha256']: raise ValueError('Opened snapshot differs from sealed bytes')
    rows=read_jsonl(path)
    if [r['id'] for r in rows]!=frozen['example_ids'] or len(rows)!=6 or any(r['split']!='holdout' for r in rows):
        raise ValueError('Holdout membership differs')
    audit_rows(rows)
    return rows
