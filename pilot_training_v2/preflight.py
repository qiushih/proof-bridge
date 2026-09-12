"""Token-only preparation and runner freeze. Never loads Qwen weights."""
from baseline.setup_model import check_model
from baseline_dev_v2.experiment import check_prompt
from constrained_v1.experiment import tokenizer_only
from training_protocol_v1.data import encode_example
from training_protocol_v2.preparation import baseline_records
from pilot_training_v2.common import *


def prepare(output):
    require(not output.exists(), 'Preflight exists; refusing overwrite')
    before, code = inputs(), code_hashes()
    observed = hardware()
    rocq = verifier_smoke()
    model_lock = check_model()  # Byte hashes only, no model deserialization.
    c = config()
    require(model_lock['model_id'] == c['model_id'] and model_lock['revision'] == c['model_revision'], 'Wrong base model')
    tokenizer = tokenizer_only()
    prompt = read(ROOT / 'prompt_v2/selected.json')
    encoded = {}
    for row in public_rows('intervention') + dev_rows():
        r = encode_example(tokenizer, prompt, row)
        p = r['prompt_length']
        require(r['labels'][:p] == [-100] * p and r['labels'][p:] == r['input_ids'][p:]
                and r['labels'][-1] == tokenizer.eos_token_id and r['target_length'] <= 256, 'Invalid loss mask/token budget')
        encoded[row['id']] = {'encoding_sha256': digest(__import__('json').dumps(r, sort_keys=True)),
                             **{k: r[k] for k in ('prompt_length', 'target_length', 'sequence_length')}}
    zero = baseline_records()
    examples = {r['id']: r for r in dev_rows()}
    for row in zero:
        check_prompt(row, examples[row['dev_example_id']], tokenizer)
    streams = schedules()
    for arm in ARMS:
        require(sum(encoded[i]['target_length'] for i in streams[arm]['microbatches']) == streams[arm]['supervised_tokens']
                and sum(encoded[i]['sequence_length'] for i in streams[arm]['microbatches']) == streams[arm]['total_sequence_tokens'], 'Schedule token totals differ')
    require(before == inputs() and code == code_hashes(), 'Inputs/code changed during preflight')
    output.mkdir(parents=True)
    write(output / 'preflight.json', {'status': 'PASS', 'prepared_at_utc': now(), 'inputs_sha256': before,
          'code_sha256': code, 'configuration': c, 'hardware': observed, 'model': model_lock,
          'public_encodings': encoded, 'verifier_smoke_check': rocq, 'schedules_sha256': sha256_file(ROOT / 'data/pilot-v2-training/schedule.json'),
          'step_zero': {'source': 'results/v2-dev-baseline/assessed_results.jsonl',
                        'source_keys': [r['key'] for r in zero], 'prompt_and_token_identity_checked': True,
                        'new_generations': 0},
          'expected_updates_per_arm': 60, 'expected_microbatches_per_arm': 240,
          'expected_new_dev_generations': 160, 'qwen_weights_loaded': False,
          'qwen_optimizer_updates': 0, 'model_generations': 0, 'holdout_content_parsed': False})
    return {'status': 'PASS', 'public_encodings_checked': len(encoded), 'qwen_weights_loaded': False, 'model_generations': 0}


def freeze(output):
    require(not (HERE / 'release.json').exists(), 'Runner already frozen')
    p, validation = read(output / 'preflight.json'), read(output / 'validation.json')
    require(p['inputs_sha256'] == inputs() and p['code_sha256'] == code_hashes(), 'Stale preflight')
    require(validation['status'] == 'PASS' and validation['tests_run'] >= 20 and validation['tests_failed'] == 0
            and validation['code_sha256'] == code_hashes() and validation['qwen_weights_loaded'] is False,
            'Bound synthetic/regression validation required')
    write(HERE / 'release.json', {'status': 'FROZEN_RUNNER_NOT_TRAINED', 'frozen_at_utc': now(),
          'inputs_sha256': inputs(), 'code_sha256': code_hashes(),
          'preflight_sha256': sealed_files(output), 'qwen_optimizer_updates': 0, 'model_generations': 0})
    return {'status': 'FROZEN_RUNNER_NOT_TRAINED', 'tests_run': validation['tests_run']}
