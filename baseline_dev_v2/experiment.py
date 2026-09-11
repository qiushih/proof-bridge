"""Freeze, run exactly four new attempts, assess, and replay the v2 dev baseline."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys

from baseline.protocol import extract_body
from baseline.setup_model import check_model
from constrained_v1.experiment import generate, tokenizer_only
from constrained_v1.grammar import initial, feed
from constrained_v1.tokens import Vocabulary
from pilot_dev_v2 import release as dev
from pilot_v1.protocol import ROOT, digest, read_rows
from prompt_v2.protocol import build_messages, format_compliance
from training_protocol_v1.scoring import step_adherence
from verifier import sha256_file, verify

OUT = ROOT / 'results/v2-dev-baseline'
HERE = Path(__file__).resolve().parent
MODELS = ('base', 'step18')
OLD_IDS = [f'pd0{n}_{a}' for n in (1, 2, 3) for a in ('A', 'B')]
NEW_IDS = ['pd04_A', 'pd04_B']
RAW = {'base': 'results/pretraining-dev-v1/raw_generations.jsonl',
       'step18': 'results/lora-pilot-v1/dev-step-0018/raw_generations.jsonl'}
ASSESSED = {'base': 'results/pretraining-dev-v1/assessed_results.jsonl',
            'step18': 'results/lora-pilot-v1/assessed_results.jsonl'}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    with path.open('x') as f:
        f.write(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def write_rows(path, rows):
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def examples():
    dev.check()
    rows = read_rows(dev.DATA / 'dev.jsonl')
    require([r['id'] for r in rows] == OLD_IDS + NEW_IDS, 'Unexpected dev membership')
    return {r['id']: r for r in rows}


def sources():
    files = dev.fingerprints()
    names = list(RAW.values()) + list(ASSESSED.values()) + [
        'data/pilot-v2/dev.jsonl', 'data/pilot-v2/dev_manifest.json', 'data/pilot-v2/release.json',
        'baseline/config.json', 'baseline/requirements.lock', 'baseline/run.py',
        'baseline/protocol.py', 'baseline/setup_model.py', 'lora_training_v1/core.py',
        'training_protocol_v1/adaptation.py', 'pilot_training_v1/evaluation.py',
        'results/pretraining-dev-v1/manifest.json', 'results/pretraining-dev-v1/completion.json',
        'results/lora-pilot-v1/manifest.json', 'results/lora-pilot-v1/completion.json',
        'results/lora-pilot-v1/selected_checkpoint.json',
        'results/lora-pilot-v1/dev-step-0018/completion.json',
        'results/lora-pilot-v1/training/step-0018/checkpoint.json',
        'results/lora-pilot-v1/training/step-0018/adapter.safetensors']
    files.update({name: sha256_file(ROOT / name) for name in names})
    return files


def prompt_for(example):
    return build_messages(read(ROOT / 'prompt_v2/selected.json'), example, 'theorem_and_informal')


def check_prompt(row, example, tokenizer):
    messages = prompt_for(example)
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    expected = {'messages': messages, 'rendered_text': rendered, 'sha256': digest(rendered),
                'input_token_ids': tokenizer.encode(rendered, add_special_tokens=False)}
    require(row['formal_statement'] == example['formal_statement'] and row['prompt'] == expected,
            'Theorem, prompt or token identity differs')


def selected():
    s = read(ROOT / 'results/lora-pilot-v1/selected_checkpoint.json')
    require(s['selected_step'] == 18 and s['selected_adapter_sha256'] ==
            '004eac24b10840832b2babc290e535f91a5a887d68db5977f58e812c8e57e6f0', 'Wrong selected checkpoint')
    require(s['selected_adapter_path'] == 'results/lora-pilot-v1/training/step-0018/adapter.safetensors', 'Unexpected adapter path')
    require(sha256_file(ROOT / s['selected_adapter_path']) == s['selected_adapter_sha256'], 'Adapter changed')
    return s


def reuse_records(exs, tokenizer):
    baseline = read(ROOT / 'results/pretraining-dev-v1/manifest.json')
    pilot = read(ROOT / 'results/lora-pilot-v1/manifest.json')
    for field in ('model', 'generation_settings', 'effective_generation_config'):
        require(baseline[field] == pilot[field], 'Historical inference settings differ')
    require(baseline['model'] == read(ROOT / 'baseline/model.lock.json'), 'Historical model differs')
    require(baseline['generation_settings'] == read(ROOT / 'baseline/config.json')['generation'], 'Settings changed')
    # Verify original completion records for just the reused raw runs/manifests.
    for model, folder in (('base', 'results/pretraining-dev-v1'), ('step18', 'results/lora-pilot-v1/dev-step-0018')):
        completion = read(ROOT / folder / 'completion.json')
        for name, expected in completion['files_sha256'].items():
            require(sha256_file(ROOT / folder / name) == expected, 'Historical completion hash mismatch')
    complete = read(ROOT / 'results/lora-pilot-v1/completion.json')
    for name in ('manifest.json', 'training/step-0018/adapter.safetensors'):
        require(sha256_file(ROOT / 'results/lora-pilot-v1' / name) == complete['files_sha256'][name], 'Pilot custody mismatch')
    rows, evidence = [], []
    adapter = selected()
    for model in MODELS:
        raw = [r for r in read_rows(ROOT / RAW[model]) if r.get('dev_example_id') in OLD_IDS]
        judged = {r['key']: r for r in read_rows(ROOT / ASSESSED[model])
                  if r.get('dev_example_id') in OLD_IDS and (model == 'base' or r.get('optimizer_step') == 18)}
        require(Counter(r['dev_example_id'] for r in raw) == Counter(OLD_IDS), 'Reuse requires six unique argument rows')
        for row in raw:
            example = exs[row['dev_example_id']]
            check_prompt(row, example, tokenizer)
            require(row['attempt'] == 1 and row['repair_attempts'] == 0 and not row['generation_error'], 'Historical attempt invalid')
            require(row['condition'] == 'argument_' + example['argument_id'], 'Historical condition differs')
            if model == 'step18':
                require(row['optimizer_step'] == 18 and row['checkpoint_adapter_sha256'] == adapter['selected_adapter_sha256'], 'Historical adapter differs')
            assessed = judged[row['key']]
            require(all(assessed[k] == v for k, v in row.items()), 'Assessed record differs from raw')
            review = assessed['mathematical_argument_fidelity']
            require(review['proof_body_sha256'] == digest(row['proof_body']), 'Historical review not hash-bound')
            evidence.append({'model_key': model, 'example_id': example['id'], 'source_key': row['key'],
                             'prompt_messages_rendering_tokens_identical': True, 'model_and_settings_identical': True,
                             'original_review_reused': True})
            rows.append(assessed | {'model_key': model, 'example_id': example['id'], 'subset': 'original_six',
                                   'origin': {'kind': 'reused', 'raw_path': RAW[model], 'assessed_path': ASSESSED[model],
                                              'source_key': row['key'], 'raw_file_sha256': sha256_file(ROOT / RAW[model])}})
    return rows, evidence


def freeze(output):
    require(not output.exists(), 'Output exists; refusing protocol overwrite')
    exs = examples()
    model = check_model()
    before = sources()
    tokenizer = tokenizer_only()
    reused, reuse_audit = reuse_records(exs, tokenizer)
    old = read(ROOT / 'results/pretraining-dev-v1/manifest.json')
    require(old['model'] == model, 'Base model differs')
    output.mkdir(parents=True)
    (output / 'PROTOCOL.md').write_bytes((HERE / 'PROTOCOL.md').read_bytes())
    write_rows(output / 'reused_records.jsonl', reused)
    write(output / 'reuse_audit.json', {'status': 'PASS', 'reused_records': 12, 'records': reuse_audit})
    protocol = {'status': 'FROZEN_BEFORE_NEW_GENERATIONS', 'frozen_at_utc': now(),
                'experiment_id': 'proofbridge-v2-dev-baseline', 'expected_new_attempts': 4,
                'expected_reused_records': 12, 'new_ids': NEW_IDS, 'old_ids': OLD_IDS,
                'model_order': list(MODELS), 'model': model, 'selected': selected(),
                'generation_settings': old['generation_settings'],
                'effective_generation_config': old['effective_generation_config'],
                'python_version': old['runtime']['python'],
                'required_packages': {k: old['runtime']['packages'][k] for k in ('torch', 'transformers', 'tokenizers', 'safetensors', 'numpy')},
                'sources_sha256': before,
                'runner_sha256': {p.name: sha256_file(p) for p in sorted(HERE.iterdir()) if p.suffix in {'.py', '.md'}},
                'preparation_sha256': {p: sha256_file(output / p) for p in ('PROTOCOL.md', 'reused_records.jsonl', 'reuse_audit.json')},
                'training_updates': 0, 'repair_attempts': 0, 'holdout_content_reads': 0,
                'checkpoint_selection': 'unchanged; step18 is fixed, no selection in this baseline'}
    require(before == sources(), 'Inputs changed during preparation')
    write(output / 'protocol.json', protocol)
    return {'status': 'FROZEN', 'reused': 12, 'new_attempts_authorized': 4}


def check_protocol(output):
    p = read(output / 'protocol.json')
    require(p['status'] == 'FROZEN_BEFORE_NEW_GENERATIONS' and p['expected_new_attempts'] == 4, 'Invalid protocol')
    require(p['sources_sha256'] == sources(), 'Protected inputs changed')
    require(p['runner_sha256'] == {f.name: sha256_file(f) for f in sorted(HERE.iterdir()) if f.suffix in {'.py', '.md'}}, 'Frozen runner changed')
    for name, expected in p['preparation_sha256'].items():
        require(sha256_file(output / name) == expected, 'Prepared evidence changed')
    examples()
    return p


def load_adapter(model, path):
    import torch
    from safetensors.torch import load_file
    from training_protocol_v1.adaptation import attach_for_future_training
    from lora_training_v1.core import audit_trainable, tensor_digest
    attach_for_future_training(model)
    named, _ = audit_trainable(model)
    tensors = load_file(str(path))
    require(set(named) == set(tensors), 'Wrong adapter tensor names')
    with torch.no_grad():
        for name, target in named.items():
            source = tensors[name]
            require(target.shape == source.shape and target.dtype == source.dtype and torch.isfinite(source).all(), 'Invalid adapter tensor')
            target.copy_(source)
    hashes = {n: tensor_digest(p) for n, p in named.items()}
    require(hashes == {n: tensor_digest(p) for n, p in tensors.items()}, 'Adapter load hash differs')
    return named, hashes


def worker(output, model_key):
    from prompt_v2.experiment import Inference
    from lora_training_v1.core import base_hashes, tensor_digest
    p = check_protocol(output)
    require((output / 'run_started.json').exists(), 'Run not started')
    require(model_key in MODELS, 'Unexpected model')
    folder = output / model_key
    folder.mkdir(exist_ok=False)
    engine = Inference(p['generation_settings'])
    require(engine.lock == p['model'] and engine.generation.to_dict() == p['effective_generation_config'], 'Model/settings differ')
    require(engine.runtime['python'] == p['python_version'], 'Python version differs')
    require(all(engine.runtime['packages'][k] == v for k, v in p['required_packages'].items()), 'Model library version differs')
    before = base_hashes(engine.model)
    named, adapter_before = {}, {}
    if model_key == 'step18':
        named, adapter_before = load_adapter(engine.model, ROOT / p['selected']['selected_adapter_path'])
    engine.model.requires_grad_(False)
    engine.model.eval()
    require(base_hashes(engine.model) == before, 'Adapter attachment changed base')
    vocabulary = Vocabulary(engine.tokenizer)
    write(folder / 'manifest.json', {'model_key': model_key, 'runtime': engine.runtime, 'model': engine.lock,
          'effective_generation_config': engine.generation.to_dict(), 'generation_settings': p['generation_settings'],
          'protocol_sha256': sha256_file(output / 'protocol.json'), 'expected_attempts': 2,
          'selected_adapter_sha256': p['selected']['selected_adapter_sha256'] if named else None,
          'base_hashes_before': before, 'adapter_hashes_before': adapter_before,
          'trainable_parameters': sum(t.numel() for t in engine.model.parameters() if t.requires_grad)})
    exs = examples()
    with (folder / 'raw_generations.jsonl').open('x') as raw, (folder / 'attempts.jsonl').open('x') as journal:
        for example_id in NEW_IDS:
            e = exs[example_id]
            key = f'v2_dev_baseline:{model_key}:{example_id}'
            journal.write(json.dumps({'key': key, 'event': 'started', 'attempt': 1, 'at_utc': now()}) + '\n'); journal.flush()
            print(f'Generating {model_key} {example_id}', flush=True)
            text, generated = generate(engine, vocabulary, e['formal_statement'], prompt_for(e))
            body, extraction = extract_body(text)
            verification = asdict(verify(e['formal_statement'], body, timeout=10.0))
            record = {'key': key, 'model_key': model_key, 'example_id': example_id, 'dev_example_id': example_id,
                      'theorem_id': e['theorem_id'], 'condition': 'argument_' + e['argument_id'],
                      'formal_statement': e['formal_statement'], 'subset': 'new_two',
                      'attempt': 1, 'repair_attempts': 0, 'origin': {'kind': 'new', 'run': 'v2-dev-baseline'},
                      'generated_text': text, 'proof_body': body, 'proof_body_sha256': digest(body),
                      'output_extraction': extraction, 'verification': verification,
                      'failure_category': None if verification['status'] == 'PASS' else verification['category'],
                      'format_compliance': format_compliance(e['formal_statement'], text),
                      'requested_proof_steps': step_adherence(e, body, 'argument_' + e['argument_id'])} | generated
            raw.write(json.dumps(record, ensure_ascii=False) + '\n'); raw.flush()
            journal.write(json.dumps({'key': key, 'event': 'finished', 'attempt': 1}) + '\n'); journal.flush()
            print('Attempt recorded', flush=True)
            require(not generated['generation_error'], 'Generation error recorded; no retry')
    after = base_hashes(engine.model)
    adapter_after = {n: tensor_digest(t) for n, t in named.items()}
    require(before == after and adapter_before == adapter_after, 'Inference altered weights')
    require(check_model() == p['model'], 'On-disk base changed')
    check_protocol(output)
    write(folder / 'completion.json', {'status': 'COMPLETE', 'attempts': 2, 'optimizer_updates': 0,
          'base_hashes_after': after, 'adapter_hashes_after': adapter_after, 'weights_unchanged': True,
          'files_sha256': {n: sha256_file(folder / n) for n in ('manifest.json', 'raw_generations.jsonl', 'attempts.jsonl')}})


def run(output):
    p = check_protocol(output)
    write(output / 'run_started.json', {'started_at_utc': now(), 'protocol_sha256': sha256_file(output / 'protocol.json'), 'expected_attempts': 4})
    for model in MODELS:
        subprocess.run([sys.executable, '-B', '-m', 'baseline_dev_v2.experiment', 'worker', '--model', model, '--output', str(output)], cwd=ROOT, check=True)
    write(output / 'generation_complete.json', {'status': 'COMPLETE', 'new_attempts': 4, 'reused_records': 12,
          'finished_at_utc': now(), 'optimizer_updates': 0,
          'protocol_sha256': sha256_file(output / 'protocol.json'),
          'files_sha256': {str(f.relative_to(output)): sha256_file(f) for model in MODELS for f in sorted((output / model).iterdir()) if f.is_file()}})
    return {'status': 'COMPLETE', 'new_attempts': 4, 'training_updates': 0}


def all_raw(output):
    return read_rows(output / 'reused_records.jsonl') + [r for model in MODELS for r in read_rows(output / model / 'raw_generations.jsonl')]


def attach_review(row, review):
    require(review['key'] == row['key'] and review['proof_body_sha256'] == digest(row['proof_body']), 'Review not bound to output')
    require(review['reviewer'] == 'assistant' and review['human_reviewed'] is False and review['reason'].strip(), 'Invalid review provenance')
    allowed = {'FAITHFUL', 'UNFAITHFUL'} if row['verification']['status'] == 'PASS' else {'NOT_ESTABLISHED'}
    require(review['status'] in allowed, 'Failed proof cannot establish mathematical fidelity')
    return row | {'mathematical_argument_fidelity': review}


def metrics(rows):
    passed = lambda r: r['verification']['status'] == 'PASS'
    faithful = lambda r: passed(r) and r['mathematical_argument_fidelity']['status'] == 'FAITHFUL'
    adherent = lambda r: faithful(r) and r['requested_proof_steps']['matched'] is True
    groups = defaultdict(list)
    for r in rows:
        groups[r['theorem_id']].append(r)
    pairs = {t: {'both_verified': len(g) == 2 and all(passed(r) for r in g),
                 'both_verified_faithful': len(g) == 2 and all(faithful(r) for r in g),
                 'both_verified_faithful_step_adherent': len(g) == 2 and all(adherent(r) for r in g)} for t, g in groups.items()}
    values = {'verified': sum(passed(r) for r in rows), 'verified_faithful': sum(faithful(r) for r in rows),
              'requested_step_matches': sum(r['requested_proof_steps']['matched'] is True for r in rows),
              'verified_requested_steps': sum(passed(r) and r['requested_proof_steps']['matched'] is True for r in rows),
              'verified_faithful_requested_steps': sum(adherent(r) for r in rows)}
    return {'rows': len(rows), **values, 'percentages': {k: round(100 * v / len(rows), 2) for k, v in values.items()},
            'failure_categories': dict(Counter(r['failure_category'] for r in rows if not passed(r))),
            'theorem_pairs': pairs, 'successful_pairs': sum(x['both_verified_faithful_step_adherent'] for x in pairs.values()),
            'pair_count': len(pairs), 'pair_success_percent': round(100 * sum(x['both_verified_faithful_step_adherent'] for x in pairs.values()) / len(pairs), 2),
            'mean_generation_seconds': statistics.mean(r['generation_latency_seconds'] for r in rows),
            'completion_tokens': sum(r['usage']['completion_tokens'] for r in rows),
            'prompt_tokens': sum(r['usage']['prompt_tokens'] for r in rows)}


def replay_record(row, example, tokenizer=None, vocabulary=None, reverify=False):
    require(row['proof_body_sha256'] == digest(row['proof_body']), 'Body hash differs')
    require((row['proof_body'], row['output_extraction']) == extract_body(row['generated_text']), 'Extraction differs')
    require(row['formal_statement'] == example['formal_statement'], 'Theorem differs')
    require(row['requested_proof_steps'] == step_adherence(example, row['proof_body'], row['condition']), 'Step assessment differs')
    require(row['format_compliance'] == format_compliance(example['formal_statement'], row['generated_text']), 'Format assessment differs')
    require(row['attempt'] == 1 and row['repair_attempts'] == 0 and row['generation_error'] is None, 'Attempt differs')
    p, n = len(row['prompt']['input_token_ids']), len(row['completion_token_ids'])
    require(n <= 256 and row['usage'] == {'prompt_tokens': p, 'completion_tokens': n, 'total_tokens': p + n}, 'Token accounting differs')
    if tokenizer:
        check_prompt(row, example, tokenizer)
        state = initial(row['formal_statement'])
        for i, token in enumerate(row['completion_token_ids']):
            require(token in vocabulary.allowed(state), 'Completion violates decoder')
            if token == vocabulary.eos_id:
                require(i == n - 1, 'Premature EOS')
            else:
                state = feed(state, vocabulary.pieces[token])
        require(tokenizer.decode(row['completion_token_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=False) == row['generated_text'], 'Completion token replay differs')
    if reverify:
        actual = asdict(verify(row['formal_statement'], row['proof_body'], timeout=10.0))
        for field in ('status', 'category', 'source_sha256', 'kernel_checked', 'assumptions_checked'):
            require(actual[field] == row['verification'][field], 'Rocq replay differs')
        return {'key': row['key'], 'verification': actual}


def check(output, replay_tokens=False, reverify=False):
    p = check_protocol(output)
    complete = read(output / 'generation_complete.json')
    require(complete['new_attempts'] == 4 and complete['protocol_sha256'] == sha256_file(output / 'protocol.json'), 'Generation completion differs')
    started = read(output / 'run_started.json')
    require(started['started_at_utc'] > p['frozen_at_utc'] and started['protocol_sha256'] == sha256_file(output / 'protocol.json'), 'Protocol did not precede inference')
    for name, expected in complete['files_sha256'].items():
        require(sha256_file(output / name) == expected, 'Raw evidence changed')
    for model in MODELS:
        folder = output / model
        records = read_rows(folder / 'raw_generations.jsonl')
        require([r['example_id'] for r in records] == NEW_IDS and all(r['model_key'] == model for r in records), 'New attempt identity differs')
        events = read_rows(folder / 'attempts.jsonl')
        require([(e['key'], e['event'], e['attempt']) for e in events] == [(r['key'], event, 1) for r in records for event in ('started', 'finished')], 'Attempt count/order differs')
        manifest, done = read(folder / 'manifest.json'), read(folder / 'completion.json')
        require(manifest['base_hashes_before'] == done['base_hashes_after'] and manifest['adapter_hashes_before'] == done['adapter_hashes_after'], 'Weight preservation differs')
        require(manifest['trainable_parameters'] == 0 and done['optimizer_updates'] == 0, 'Weights were trainable')
    rows = all_raw(output)
    require(len(rows) == 16 and len({(r['model_key'], r['example_id']) for r in rows}) == 16, 'Expected sixteen unique records')
    exs = examples()
    tokenizer = tokenizer_only() if replay_tokens else None
    vocabulary = Vocabulary(tokenizer) if tokenizer else None
    replay = [replay_record(r, exs[r['example_id']], tokenizer, vocabulary, reverify) for r in rows]
    if (output / 'release.json').exists():
        seal = read(output / 'release.json')
        for name, expected in seal['files_sha256'].items():
            require(sha256_file(output / name) == expected, 'Final report/assessment changed')
        assessed = read_rows(output / 'assessed_results.jsonl')
        require(len(assessed) == 16, 'Missing assessments')
        for raw, judged in zip(rows, assessed):
            require(all(judged[k] == v for k, v in raw.items()), 'Assessment changed raw evidence')
            attach_review(raw, judged['mathematical_argument_fidelity'])
    return {'status': 'PASS', 'new_attempts': 4, 'reused_records': 12, 'total_records': 16,
            'token_replay': replay_tokens, 'rocq_replays': 16 if reverify else 0, 'replay_records': replay if reverify else []}


def assess(output):
    require(not (output / 'release.json').exists(), 'Assessment already frozen')
    evidence = check(output, reverify=True)
    reviews = read(output / 'new_reviews.json')
    rows = all_raw(output)
    new = [r for r in rows if r['origin']['kind'] == 'new']
    require(len(reviews) == 4 and {r['key'] for r in reviews} == {r['key'] for r in new}, 'Exactly four new reviews required')
    by_key = {r['key']: r for r in reviews}
    assessed = [attach_review(r, by_key[r['key']]) if r['origin']['kind'] == 'new' else attach_review(r, r['mathematical_argument_fidelity']) for r in rows]
    summary = {model: {subset: metrics([r for r in assessed if r['model_key'] == model and (subset == 'all_eight' or r['subset'] == subset)])
                       for subset in ('original_six', 'new_two', 'all_eight')} for model in MODELS}
    write_rows(output / 'assessed_results.jsonl', assessed)
    write(output / 'summary.json', summary)
    write(output / 'verification_replay.json', evidence)
    (output / 'V2_DEV_BASELINE.md').write_text(report(summary, assessed))
    names = ['protocol.json', 'run_started.json', 'generation_complete.json', 'new_reviews.json', 'assessed_results.jsonl', 'summary.json', 'verification_replay.json', 'V2_DEV_BASELINE.md']
    write(output / 'release.json', {'status': 'FROZEN', 'frozen_at_utc': now(), 'files_sha256': {n: sha256_file(output / n) for n in names}, 'training_updates': 0})
    return {'status': 'FROZEN', 'summary': summary}


def report(summary, rows):
    text = '# V2 development baseline\n\nExactly **four new generations** plus **twelve reused records** compare the pinned base model and the fixed step-18 adapter on eight development rows. No training or checkpoint selection occurred.\n\n'
    text += '| Subset | Model | Rocq verified | Verified + faithful | Verified + requested steps | Both-argument successes |\n| --- | --- | ---: | ---: | ---: | ---: |\n'
    for subset in ('original_six', 'new_two', 'all_eight'):
        for model in MODELS:
            m = summary[model][subset]
            text += f"| {subset} | {model} | {m['verified']}/{m['rows']} | {m['verified_faithful']}/{m['rows']} | {m['verified_requested_steps']}/{m['rows']} | {m['successful_pairs']}/{m['pair_count']} |\n"
    text += '\nBoth-argument success requires both proofs to verify, remain mathematically faithful and follow their respective requested steps. Failed proofs count as NOT_ESTABLISHED, never faithful.\n\n## New outputs and assistant reviews\n'
    for r in rows:
        if r['subset'] != 'new_two':
            continue
        text += f"\n### {r['model_key']} / {r['example_id']}\n\n```coq\n{r['proof_body']}\n```\n\nRocq: {r['verification']['status']} / {r['verification']['category']}. Mathematical fidelity: {r['mathematical_argument_fidelity']['status']}. Requested steps: {r['requested_proof_steps']['status']}.\n\n{r['mathematical_argument_fidelity']['reason']}\n"
    text += '\n## Latency and usage\n\n| Source subset | Model | Mean generation seconds | Completion tokens |\n| --- | --- | ---: | ---: |\n'
    for subset in ('original_six', 'new_two'):
        for model in MODELS:
            m = summary[model][subset]
            text += f"| {subset} | {model} | {m['mean_generation_seconds']:.2f} | {m['completion_tokens']} |\n"
    text += '''
Historical and new runs have different cache/order/host conditions. Timings are descriptive, not a hardware speed comparison. Full prompt/completion token usage and runtime versions are retained in machine-readable records/manifests.

## Integrity and interpretation

The reuse audit confirms identical prompts, rendered bytes, token IDs, model identities and generation settings for the original six rows. Their original raw records and hash-bound reviews are retained unchanged. The base's historical B proofs use IH rewriting; those verified proofs remain mathematically faithful despite missing the requested congruence steps.

All sixteen saved outputs were replayed through the unchanged Rocq verifier with matching status/category/source hashes. New inference used frozen Prompt v2, constrained-v1, CPU float32, eager attention, seed 1729, greedy one-beam decoding, one attempt, no repair, and 256 tokens. In-memory base/adapter hashes match before/after. No optimizer was constructed and no model weights were updated.

The original six rows were used to select step 18. The two new rows are one theorem pair in the same historically exposed dev family. This is development evidence, not independent generalization or statistical proof of improvement. No new theorem-only controls were run, so no causal claim of informal-proof dependence is supported. Mathematical reviews are assistant reviews, not independent human assessments. Holdout and diagnostic example files were not opened. The v1 checkpoint-selection rule remains unchanged.

## Reproduction

From the repository root:

```sh
.venv/bin/python -B -m baseline_dev_v2.experiment check --replay-tokens --reverify
.venv/bin/python -B -m unittest discover -s tests -p 'test_v2_dev_baseline.py' -v
```

The one-time workflow was `freeze`, `run`, followed by four hash-bound entries in `new_reviews.json` and `assess`. Each command accepts `--output` for its explicit result directory. `run` refuses an existing start marker; do not delete it to retry. `check` never generates samples. `protocol.json` and PROTOCOL.md were frozen before the new attempts; `release.json` seals the final assessment and report. The raw new generations and attempt journals are in base/ and step18/; reused_records.jsonl identifies historical provenance; assessed_results.jsonl contains all sixteen results.
'''
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'worker', 'assess', 'check'))
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--replay-tokens', action='store_true')
    parser.add_argument('--reverify', action='store_true')
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.output, args.model); return
    if args.command == 'check':
        result = check(args.output, args.replay_tokens, args.reverify)
        result.pop('replay_records')
    else:
        result = {'freeze': freeze, 'run': run, 'assess': assess}[args.command](args.output)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
