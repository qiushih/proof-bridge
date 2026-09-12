"""Replay evidence, attach manual reviews, and report without reselection."""
from dataclasses import asdict
import json

from baseline_dev_v2.experiment import replay_record, metrics
from constrained_v1.experiment import tokenizer_only
from constrained_v1.tokens import Vocabulary
from pilot_training_v2.report import reviewed, persist_once
from verifier import verify
from pilot_holdout_v2.protocol import *
from pilot_holdout_v2.experiment import prompt_for


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def all_raw(output=OUTPUT):
    return [r for model in MODELS for r in read_rows(output / model / 'raw_generations.jsonl')]


def check_identity(rows, model, adapter_sha):
    require([r['example_id'] for r in rows] == IDS, 'Six ordered unique model outputs required')
    for r in rows:
        require(r['model_key'] == model and r['key'] == f'holdout_v2:{model}:{r["example_id"]}'
                and r['theorem_id'] == r['example_id'].rsplit('_', 1)[0]
                and r['condition'] == 'argument_' + r['example_id'][-1]
                and r['checkpoint_adapter_sha256'] == adapter_sha, 'Output identity or selected adapter differs')


def check_journal(rows, journal):
    require([(e['key'], e['event'], e['attempt']) for e in journal] ==
            [(r['key'], event, 1) for r in rows for event in ('started', 'finished')],
            'Attempt journal is missing, duplicated or out of order')


def check(output=OUTPUT, replay_tokens=False, reverify=False):
    p = check_protocol()
    require((output / 'generation_complete.json').exists() and not (output / 'interruption.json').exists(), 'No complete holdout run')
    done, start = read(output / 'generation_complete.json'), read(output / 'run_started.json')
    require(done['status'] == 'COMPLETE_AWAITING_REVIEWS' and done['attempts'] == 18
            and done['optimizer_updates'] == 0 and done['model_order'] == list(MODELS), 'Wrong run budget')
    require(done['protocol_sha256'] == start['protocol_sha256'] == sha256_file(FROZEN), 'Unbound run')
    check_files(output, done['files_sha256'])
    examples = {r['id']: r for r in opened_rows(output, p)}
    tokenizer = tokenizer_only() if replay_tokens else None
    vocabulary = Vocabulary(tokenizer) if tokenizer else None
    replay, all_rows, base_hashes = [], [], []
    for model in MODELS:
        folder = output / model
        require(not (folder / 'interruption.json').exists(), 'Model run interrupted')
        m, d = read(folder / 'manifest.json'), read(folder / 'completion.json')
        expected_adapter = None if model == 'base' else p['selected'][model]['selected_adapter_sha256']
        require(m['protocol_sha256'] == sha256_file(FROZEN) and m['model_key'] == model
                and m['model'] == p['model'] and m['generation_settings'] == p['generation_settings']
                and m['effective_generation_config'] == p['effective_generation_config']
                and m['checkpoint_adapter_sha256'] == expected_adapter, 'Worker manifest differs')
        require(m['trainable_parameters'] == 0 and d['optimizer_updates'] == 0
                and d['status'] == 'COMPLETE' and d['attempts'] == m['expected_attempts'] == 6
                and d['weights_unchanged'] and m['base_hashes_before'] == d['base_hashes_after']
                and m['adapter_hashes_before'] == d['adapter_hashes_after'], 'Weight preservation or budget differs')
        if model == 'base':
            require(m['adapter_hashes_before'] == {}, 'Base model must have no adapter')
        else:
            cp = TRAINING / p['selected'][model]['selected_adapter_path_relative_to_run']
            require(m['adapter_hashes_before'] == read(cp.parent / 'checkpoint.json')['adapter_tensor_hashes'], 'Adapter tensors differ')
        base_hashes.append(m['base_hashes_before'])
        check_files(folder, d['files_sha256'])
        rows = read_rows(folder / 'raw_generations.jsonl')
        check_identity(rows, model, expected_adapter)
        check_journal(rows, read_rows(folder / 'attempts.jsonl'))
        for r in rows:
            e = examples[r['example_id']]
            require(r['prompt']['messages'] == prompt_for(e), 'Prompt content changed')
            require(r['failure_category'] == (None if r['verification']['status'] == 'PASS' else r['verification']['category']), 'Failure category differs')
            replay.append(replay_record(r, e, tokenizer, vocabulary, reverify))
        all_rows.extend(rows)
    require(base_hashes[0] == base_hashes[1] == base_hashes[2], 'Models did not share the same pinned base')
    require(len(all_rows) == len({r['key'] for r in all_rows}) == 18, 'Expected exactly 18 attempts')
    if (output / 'release.json').exists():
        check_files(output, read(output / 'release.json')['files_sha256'])
    return {'status': 'PASS', 'new_generations': 18, 'optimizer_updates': 0,
            'token_replays': 18 if replay_tokens else 0, 'rocq_replays': 18 if reverify else 0,
            'original_content_openings': 1, 'diagnostic_and_old_holdout_opened': False,
            'records': replay if reverify else []}


def assessed(output=OUTPUT):
    raw = all_raw(output)
    hashes = {f'{model}/raw_generations.jsonl': sha256_file(output / model / 'raw_generations.jsonl') for model in MODELS}
    return reviewed(raw, read(output / 'manual_reviews.json'), hashes)


def build_report(summary, rows):
    lines = ['# ProofBridge v2 final holdout comparison', '',
             'Exactly 18 new one-attempt generations compare the pinned base and the already selected control/intervention step-36 adapters. No checkpoint was reselected and no weights were updated.', '',
             '| Model | Verified | Verified + mathematically faithful | Verified + requested steps | Verified + faithful + requested steps | Both-argument successes |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in MODELS:
        m = summary['models'][model]
        lines.append(f"| {model} | {m['verified']}/6 | {m['verified_faithful']}/6 | {m['verified_requested_steps']}/6 | {m['verified_faithful_requested_steps']}/6 | {m['successful_pairs']}/3 |")
    lines += ['', 'Pair success requires both A and B to verify, be mathematically faithful, and match their respective requested steps. All failed proofs remain in the denominator. Raw requested-step matches and percentages are recorded separately in summary.json.', '',
              '## Per-theorem results', '', '| Theorem | Model | A: verified / faithful / steps | B: verified / faithful / steps |', '| --- | --- | --- | --- |']
    for theorem in sorted({r['theorem_id'] for r in rows}):
        for model in MODELS:
            pair = [next(r for r in rows if r['theorem_id'] == theorem and r['model_key'] == model and r['condition'] == 'argument_' + a) for a in ('A', 'B')]
            cells = [' / '.join(('yes' if r['verification']['status'] == 'PASS' else 'no',
                                'yes' if r['mathematical_argument_fidelity']['status'] == 'FAITHFUL' else 'no',
                                'yes' if r['requested_proof_steps']['matched'] else 'no')) for r in pair]
            lines.append(f'| {theorem} | {model} | {cells[0]} | {cells[1]} |')
    lines += ['', '## Interpretation and limits', '',
              'The frozen development intervention-success decision remains false: both selected arms scored 8/8 on dev. Holdout results do not revise that decision, the checkpoint selections, or any frozen input.', '',
              'These six rows are three correlated A/B pairs from one family. They are fresh held-out instances under the frozen grouping, not independent unseen mathematics. References were assistant-curated and checked before sealing. Fidelity judgments here are assistant reviews, not independent human review. Equivalent IH tactic implementations may be mathematically faithful while missing the requested proof step. There is no theorem-only control in this final evaluation, so these results do not establish causal informal-proof dependence. Further tuning requires a new holdout.', '',
              '## Integrity and timing', '',
              'The original holdout was opened once after the training results, selected hashes, runner, public preflight and protocol were committed. Later assessment/replay reads the hash-identical snapshot. Historical holdout and diagnostic contents remained unopened. All model processes used frozen Prompt v2, constrained-v1, CPU float32, greedy one-beam decoding, seed 1729, 256 tokens, one attempt, no repair, and the unchanged Rocq verifier. Base and adapter hashes match before and after inference.', '',
              '| Model | Mean generation seconds | Prompt tokens | Completion tokens |', '| --- | ---: | ---: | ---: |']
    for model in MODELS:
        m = summary['models'][model]
        lines.append(f"| {model} | {m['mean_generation_seconds']:.2f} | {m['prompt_tokens']} | {m['completion_tokens']} |")
    lines += ['', 'Timing is descriptive: fresh process/vocabulary per model, shared cache within each six-row run, no reference prewarming.', '',
              '## Reproduction', '', 'From the repository root:', '', '```sh',
              '.venv/bin/python -B -m pilot_holdout_v2.experiment check --replay-tokens --reverify',
              '.venv/bin/python -B -m pilot_holdout_v2.experiment report',
              '.venv/bin/python -B -m unittest discover -s tests -p test_pilot_holdout_v2.py -v', '```', '',
              'These commands replay saved evidence or test public fixtures; they do not generate new samples. Never delete the custody/start markers to repeat the holdout run.', '', '## Saved outputs and manual fidelity reviews', '']
    for r in rows:
        review = r['mathematical_argument_fidelity']
        lines += [f"### {r['model_key']} / {r['example_id']}", '', '```coq', r['proof_body'], '```', '',
                  f"Rocq: {r['verification']['status']} / {r['verification']['category']}; mathematical fidelity: {review['status']}; requested steps: {r['requested_proof_steps']['status']}.", '', review['reason'], '']
    return '\n'.join(lines)


def report(output=OUTPUT):
    check(output)
    rows = assessed(output)
    summary = {'experiment_id': 'proofbridge-holdout-v2', 'new_generations': 18,
               'models': {model: metrics([r for r in rows if r['model_key'] == model]) for model in MODELS},
               'selected_steps': {'base': 0, 'control': 36, 'intervention': 36},
               'development_success_unchanged': False, 'reselection': False,
               'optimizer_updates': 0, 'human_reviewed': False,
               'future_tuning_requires_new_holdout': True}
    persist_once(output / 'assessed_results.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    persist_once(output / 'summary.json', json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    persist_once(output / 'HOLDOUT_V2_REPORT.md', build_report(summary, rows))
    return {'status': 'ASSESSED', 'models': summary['models'], 'optimizer_updates': 0, 'reselection': False}


def freeze_results(output=OUTPUT):
    require(not (output / 'release.json').exists(), 'Results already frozen')
    result = report(output)
    validation = check(output, replay_tokens=True, reverify=True)
    reference_checks = []
    for e in opened_rows(output):
        v = asdict(verify(e['formal_statement'], e['proof_body'], timeout=10.0))
        require(v['status'] == 'PASS', 'Frozen holdout reference failed re-verification')
        reference_checks.append({'example_id': e['id'], 'verification': v})
    write(output / 'validation.json', validation)
    write(output / 'reference_verification.json', {'status': 'PASS', 'references': reference_checks})
    write(output / 'release.json', {'status': 'FROZEN_FINAL_HOLDOUT_RESULTS', 'frozen_at_utc': now(),
          'files_sha256': sealed_files(output), 'optimizer_updates': 0, 'reselection': False,
          'future_tuning_requires_new_holdout': True})
    return result | {'status': 'FROZEN_FINAL_HOLDOUT_RESULTS', 'rocq_replays': 18, 'reference_replays': 6}
