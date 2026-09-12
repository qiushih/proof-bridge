"""Require all 160 hash-bound reviews, select both arms and compare frozen gates."""
from collections import defaultdict
from training_protocol_v1.scoring import attach_reviews
from training_protocol_v2.preparation import baseline_records
from training_protocol_v2.scoring import metrics, selection_score, success_criteria
from pilot_training_v2.common import *
from pilot_training_v2.audit import check, raw_records


def select(checkpoints):
    require(set(checkpoints) == set(STEPS), 'All ten checkpoints and step zero required')
    scores = {step: selection_score(rows, step) for step, rows in checkpoints.items()}
    winner = max(scores, key=scores.get)
    return winner, scores


def reviewed(raw, review_file, expected_hashes):
    require(review_file['raw_sha256_by_checkpoint'] == expected_hashes, 'Reviews not bound to raw checkpoint files')
    require(review_file['reviewer'] == 'assistant' and review_file['human_reviewed'] is False,
            'This review workflow records assistant judgments, not independent human review')
    rows = attach_reviews(raw, review_file['reviews'])
    for r in rows:
        allowed = ('FAITHFUL', 'UNFAITHFUL', 'NOT_ESTABLISHED') if r['verification']['status'] == 'PASS' else ('NOT_ESTABLISHED',)
        require(r['mathematical_argument_fidelity']['status'] in allowed, 'Fidelity status conflicts with verification')
    return rows


def build(output):
    check(output)
    raw = raw_records(output)
    hashes = {f'{arm}/dev-step-{step:04d}/raw_generations.jsonl': sha256_file(output / arm / f'dev-step-{step:04d}/raw_generations.jsonl')
              for arm in ARMS for step in STEPS[1:]}
    assessed = reviewed(raw, read(output / 'manual_reviews.json'), hashes)
    zero = baseline_records()
    groups = defaultdict(dict)
    selected_rows, arms = {}, {}
    for arm in ARMS:
        checkpoints = {0: zero, **{step: [r for r in assessed if r['arm'] == arm and r['optimizer_step'] == step] for step in STEPS[1:]}}
        step, scores = select(checkpoints)
        selected_rows[arm] = checkpoints[step]
        path = f'{arm}/training/step-{step:04d}/adapter.safetensors' if step else None
        arms[arm] = {'selected_step': step, 'selection_score': scores[step],
                     'selected_adapter_path_relative_to_run': path,
                     'selected_adapter_sha256': sha256_file(output / path) if path else None,
                     'selected_metrics': metrics(checkpoints[step]),
                     'checkpoints': [{'step': s, 'score': scores[s], 'metrics': metrics(checkpoints[s])} for s in STEPS]}
    summary = {'experiment_id': 'proofbridge-lora-pilot-v2', 'arms': arms, 'baseline': metrics(zero),
               'development_success': success_criteria(selected_rows['control'], selected_rows['intervention']),
               'optimizer_updates': 120, 'microbatches': 480, 'new_dev_generations': 160,
               'shared_step_zero_outputs_reused': 8, 'holdout_evaluated': False, 'human_reviewed': False,
               'limitations': ['One seed; two additional rows in one existing training family.',
                              'Selection and success assessed on the same eight correlated dev rows.',
                              'Equivalent tactics may preserve mathematical fidelity while failing requested-step adherence.',
                              'Update-matched arms have different row exposures and token totals.',
                              'No independent holdout claim or causal informal-proof dependence claim from these dev results.']}
    lines = ['# ProofBridge v2 controlled training comparison', '',
             f"Predeclared development success: **{summary['development_success']['success']}**. Holdout remains sealed.", '',
             '| Subset | Model | Verified | Verified + faithful | Verified + faithful + requested steps | Both-argument successes |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for subset in ('original_six', 'new_two', 'all_eight'):
        for name, values in [('base', summary['baseline'])] + [(arm, arms[arm]['selected_metrics']) for arm in ARMS]:
            m = values[subset]
            lines.append(f"| {subset} | {name} | {m['verified']}/{m['rows']} | {m['verified_faithful']}/{m['rows']} | {m['verified_faithful_requested_steps']}/{m['rows']} | {m['both_argument_successes']}/{m['theorem_pairs']} |")
    lines += ['', 'All ten checkpoints and the shared step-zero baseline enter each selection. Ties choose the earliest step.', '']
    for arm in ARMS:
        lines.append(f"Selected {arm}: step {arms[arm]['selected_step']}, score {arms[arm]['selection_score']}.")
    lines += ['', '## Success gates', '']
    lines.extend(f"- {name}: {value}" for name, value in summary['development_success']['gates'].items())
    lines += ['', 'Full checkpoint scores, separate step-match counts and percentages are in summary.json. Raw prompts, token usage, latency, verifier diagnostics and decoder traces remain in each dev-step directory. Training logs record loss, update time and memory; timing is descriptive.', '',
              '## Limits', '']
    lines.extend('- ' + text for text in summary['limitations'])
    lines += ['', '## Reproduction', '', '```sh',
              '.venv/bin/python -B -m pilot_training_v2.experiment check --replay-tokens --reverify',
              '.venv/bin/python -B -m pilot_training_v2.experiment report', '```', '',
              'These commands replay saved evidence and reviews; they never train or generate new samples.', '']
    return '\n'.join(lines), summary, assessed


def persist_once(path, content):
    if path.exists():
        require(path.read_text() == content, 'Saved report differs; refusing overwrite')
    else:
        with path.open('x') as f:
            f.write(content)


def write_report(output=OUTPUT):
    text, summary, assessed = build(output)
    persist_once(output / 'assessed_results.jsonl', ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in assessed))
    persist_once(output / 'summary.json', json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    persist_once(output / 'selected_checkpoints.json', json.dumps({arm: {k: v for k, v in summary['arms'][arm].items() if k != 'checkpoints'} for arm in ARMS}, indent=2) + '\n')
    persist_once(output / 'V2_TRAINING_REPORT.md', text)
    return {'status': 'ASSESSED', 'selected_steps': {a: summary['arms'][a]['selected_step'] for a in ARMS},
            'development_success': summary['development_success'], 'holdout_evaluated': False}


def freeze_results(output=OUTPUT):
    require(not (output / 'release.json').exists(), 'Results already frozen')
    result = write_report(output)
    validation = check(output, replay_tokens=True, reverify=True)
    write(output / 'validation.json', validation)
    write(output / 'release.json', {'status': 'FROZEN_TRAINED_DEV_RESULTS', 'frozen_at_utc': now(),
          'selected_steps': result['selected_steps'], 'holdout_evaluated': False,
          'files_sha256': sealed_files(output)})
    return {**result, 'status': 'FROZEN_TRAINED_DEV_RESULTS'}
