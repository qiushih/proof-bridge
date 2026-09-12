"""Replay saved run evidence without training or generating additional samples."""
import math
from baseline_dev_v2.experiment import replay_record
from constrained_v1.experiment import tokenizer_only
from constrained_v1.tokens import Vocabulary
from training_protocol_v1.scoring import mathematical_structure
from prompt_v2.protocol import build_messages
from pilot_training_v2.common import *


def check_events(events, stream):
    expected = [('training_started', None, None)]
    for step in range(1, 61):
        for micro in range(1, 5):
            expected.extend((event, step, micro) for event in ('forward_started', 'forward_finished', 'backward_finished'))
        expected.extend((event, step, None) for event in ('optimizer_started', 'optimizer_finished'))
        if step % 6 == 0:
            expected.append(('checkpoint_evaluated', step, None))
    expected.append(('training_completed', None, None))
    actual = [(e['event'], e.get('update', e.get('optimizer_step')), e.get('microbatch')) for e in events]
    require(actual == expected, 'Training event order, retry or budget differs')
    updates = [e for e in events if e['event'] == 'optimizer_finished']
    starts = [e for e in events if e['event'] == 'forward_started']
    require([e['example_id'] for e in starts] == stream, 'Microbatch stream differs')
    for u in updates:
        index = (u['update'] - 1) * 4
        require(u['example_ids'] == stream[index:index+4] and u['microbatches'] == 4
                and u['pad_to'] is None and u['learning_rate'] == 0.0002, 'Update settings differ')
        require(len(u['losses']) == 4 and all(math.isfinite(x) for x in u['losses'])
                and math.isfinite(u['gradient_norm_before_clip']) and u['adapter_tensors_changed'] > 0,
                'Non-finite or ineffective update')
        require(u['trainable_audit']['trainable_parameters'] == 540672
                and u['trainable_audit']['base_trainable_parameters'] == 0
                and u['trainable_audit']['base_gradient_tensors'] == 0, 'Unexpected trainable parameters')
    return updates


def raw_records(output):
    return [r for arm in ARMS for step in STEPS[1:]
            for r in read_rows(output / arm / f'dev-step-{step:04d}/raw_generations.jsonl')]


def check(output=OUTPUT, replay_tokens=False, reverify=False):
    require((output / 'completion.json').exists() and not (output / 'interruption.json').exists(), 'No complete two-arm run; partial runs cannot be selected')
    check_ready()
    manifest, complete = read(output / 'manifest.json'), read(output / 'completion.json')
    require(manifest['configuration'] == config() and manifest['code_sha256'] == code_hashes()
            and manifest['inputs_sha256'] == inputs() and manifest['arm_order'] == list(ARMS)
            and manifest['runner_release_sha256'] == sha256_file(HERE / 'release.json'), 'Run inputs/configuration drifted')
    generation = read(ROOT / 'results/pretraining-dev-v1/manifest.json')
    require(manifest['model'] == read(ROOT / 'baseline/model.lock.json')
            and manifest['generation_settings'] == generation['generation_settings']
            and manifest['effective_generation_config'] == generation['effective_generation_config'], 'Model/generation settings differ')
    require(complete['status'] == 'COMPLETE_AWAITING_REVIEWS' and complete['optimizer_updates'] == 120
            and complete['microbatches'] == 480 and complete['new_dev_generations'] == 160, 'Wrong total budget')
    check_files(output, complete['files_sha256'])
    from pilot_training_v2.core import matching_initial
    matching_initial(read(output / 'control/manifest.json')['runtime'], read(output / 'intervention/manifest.json')['runtime'])
    tokenizer = tokenizer_only() if replay_tokens else None
    vocabulary = Vocabulary(tokenizer) if tokenizer else None
    examples = {r['id']: r for r in dev_rows()}
    frozen_streams, all_rows = schedules(), []
    for arm in ARMS:
        folder = output / arm
        m, done = read(folder / 'manifest.json'), read(folder / 'completion.json')
        require(not (folder / 'interruption.json').exists() and done['status'] == 'COMPLETE'
                and done['optimizer_updates'] == 60 and done['microbatches'] == 240
                and done['dev_generations'] == 80, 'Arm incomplete')
        require(m['parent_manifest_sha256'] == sha256_file(output / 'manifest.json') and
                m['runtime']['fresh_adapter_initialization'] is True and
                m['runtime']['feasibility_or_v1_adapter_loaded'] is False and
                m['runtime']['base_hashes_before'] == done['base_hashes_after'], 'Initialization/base preservation differs')
        check_files(folder, done['files_sha256'])
        stream = frozen_streams[arm]['microbatches']
        check_events(read_rows(folder / 'training/training_metrics.jsonl'), stream)
        require(sorted(p.name for p in (folder / 'training').glob('step-*')) == [f'step-{s:04d}' for s in STEPS[1:]], 'Checkpoint set differs')
        require(sorted(p.name for p in folder.glob('dev-step-*')) == [f'dev-step-{s:04d}' for s in STEPS[1:]], 'Evaluation checkpoint set differs')
        for step in STEPS[1:]:
            checkpoint, evaluation = folder / f'training/step-{step:04d}', folder / f'dev-step-{step:04d}'
            cp, ev = read(checkpoint / 'checkpoint.json'), read(evaluation / 'completion.json')
            require(cp['status'] == 'COMPLETE' and cp['optimizer_step'] == step and cp['arm'] == arm
                    and cp['completed_microbatches'] == cp['next_microbatch_index'] == 4 * step
                    and cp['completed_cycles'] == 4*step // len(set(stream))
                    and cp['next_cycle_offset'] == 4*step % len(set(stream))
                    and cp['adapter_parameters_only'] and not cp['base_weights_saved'], 'Checkpoint cursor/content differs')
            require(cp['bindings']['parent_manifest_sha256'] == sha256_file(output / 'manifest.json')
                    and cp['bindings']['arm_manifest_sha256'] == sha256_file(folder / 'manifest.json')
                    and cp['bindings']['schedule_sha256'] == sha256_file(ROOT / 'data/pilot-v2-training/schedule.json')
                    and cp['bindings']['runner_release_sha256'] == sha256_file(HERE / 'release.json')
                    and cp['bindings']['protocol_manifest_sha256'] == sha256_file(ROOT / 'data/pilot-v2-training/split_manifest.json'), 'Checkpoint not bound to protocol')
            check_files(checkpoint, cp['files_sha256'])
            require(set(cp['adapter_tensor_hashes']) == set(m['runtime']['initial_adapter_hashes'])
                    and ev['status'] == 'COMPLETE' and ev['arm'] == arm and ev['optimizer_step'] == step
                    and ev['attempts'] == 8 and ev['adapter_hashes_before'] == ev['adapter_hashes_after'] == cp['adapter_tensor_hashes'], 'Evaluation adapter state differs')
            check_files(evaluation, ev['files_sha256'])
            rows, journal = read_rows(evaluation / 'raw_generations.jsonl'), read_rows(evaluation / 'attempts.jsonl')
            require([r['dev_example_id'] for r in rows] == DEV_IDS, 'Dev membership/order differs')
            require([(e['key'], e['event'], e['attempt']) for e in journal] ==
                    [(r['key'], event, 1) for r in rows for event in ('started', 'finished')], 'Attempt journal differs')
            for r in rows:
                example = examples[r['dev_example_id']]
                require(r['key'] == f"pilot_training_v2:{arm}:step-{step:04d}:{example['id']}"
                        and r['arm'] == arm and r['optimizer_step'] == step and r['theorem_id'] == example['theorem_id']
                        and r['condition'] == 'argument_' + example['argument_id']
                        and r['checkpoint_adapter_sha256'] == sha256_file(checkpoint / 'adapter.safetensors'), 'Raw output identity differs')
                require(r['prompt']['messages'] == build_messages(read(ROOT / 'prompt_v2/selected.json'), example, 'theorem_and_informal'), 'Prompt messages differ')
                require(r['failure_category'] == (None if r['verification']['status'] == 'PASS' else r['verification']['category'])
                        and r['mathematical_structure_review_aid'] == mathematical_structure(r['formal_statement'], r['proof_body'])
                        and r['requested_base_action_review_aid'] == example['argument_contract']['base'], 'Review aid/category differs')
                replay_record(r, example, tokenizer, vocabulary, reverify)
            all_rows.extend(rows)
    require(len(all_rows) == len({r['key'] for r in all_rows}) == 160, 'Expected exactly 160 unique new dev outputs')
    if (output / 'release.json').exists():
        check_files(output, read(output / 'release.json')['files_sha256'])
    return {'status': 'PASS', 'optimizer_updates': 120, 'microbatches': 480,
            'new_dev_generations': 160, 'dev_token_replays': 160 if replay_tokens else 0,
            'dev_rocq_replays': 160 if reverify else 0, 'holdout_content_parsed': False}
