"""Explicit future run; read-only checks and report commands never start training."""
import argparse
import os
import signal
import subprocess
import sys
import time
import traceback

from baseline.setup_model import check_model
from pilot_training_v2.common import *


def launch_arm(output, arm):
    child = subprocess.Popen([sys.executable, '-B', '-m', 'pilot_training_v2.experiment',
                              'worker', '--output', str(output), '--arm', arm],
                             cwd=ROOT, start_new_session=True)
    try:
        code = child.wait()
        require(code == 0, f'{arm} worker failed; no retries or later arm')
    except BaseException:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=10)
        raise


def run(output=OUTPUT):
    require(not output.exists(), 'Run already exists; refusing resume, overwrite or additional attempts')
    release = check_ready()
    observed, model = hardware(), check_model()
    rocq = verifier_smoke()
    before = inputs()
    generation = read(ROOT / 'results/pretraining-dev-v1/manifest.json')
    output.mkdir(parents=True)
    write(output / 'manifest.json', {'experiment_id': 'proofbridge-lora-pilot-v2', 'started_at_utc': now(),
          'configuration': config(), 'model': model, 'hardware': observed, 'verifier_smoke_check': rocq, 'arm_order': list(ARMS),
          'inputs_sha256': before, 'runner_release_sha256': sha256_file(HERE / 'release.json'),
          'code_sha256': release['code_sha256'], 'expected_optimizer_updates': 120,
          'expected_dev_generations': 160, 'generation_settings': generation['generation_settings'],
          'effective_generation_config': generation['effective_generation_config'],
          'holdout_content_reads': 0, 'step_zero_source': 'results/v2-dev-baseline/assessed_results.jsonl'})
    started = time.perf_counter()
    try:
        for arm in ARMS:
            launch_arm(output, arm)
        from pilot_training_v2.core import matching_initial
        matching_initial(read(output / 'control/manifest.json')['runtime'], read(output / 'intervention/manifest.json')['runtime'])
        require(inputs() == before and check_model() == model, 'Inputs/model changed')
        paths = [output / 'manifest.json'] + [p for arm in ARMS for p in (output / arm).rglob('*') if p.is_file()]
        write(output / 'completion.json', {'status': 'COMPLETE_AWAITING_REVIEWS', 'optimizer_updates': 120,
              'microbatches': 480, 'new_dev_generations': 160, 'initial_adapters_identical': True,
              'wall_seconds': time.perf_counter() - started, 'finished_at_utc': now(),
              'files_sha256': {str(p.relative_to(output)): sha256_file(p) for p in sorted(paths)}})
    except BaseException as error:
        write(output / 'interruption.json', {'status': 'INTERRUPTED', 'error': str(error),
              'traceback': traceback.format_exc(), 'at_utc': now(), 'automatic_retry': False})
        raise
    return {'status': 'COMPLETE_AWAITING_REVIEWS', 'updates': 120, 'new_dev_generations': 160}


def worker(output, arm):
    require(arm in ARMS, 'Worker arm must be control or intervention')
    require(not (output / arm).exists(), 'Arm already started; no retry')
    check_ready()
    root = read(output / 'manifest.json')
    require(root['code_sha256'] == code_hashes() and root['inputs_sha256'] == inputs()
            and root['configuration'] == config() and not (output / 'interruption.json').exists(), 'Parent run manifest invalid')
    control_runtime = None
    if arm == 'intervention':
        control = read(output / 'control/completion.json')
        require(control['status'] == 'COMPLETE' and control['optimizer_updates'] == 60 and control['dev_generations'] == 80, 'Control must finish first')
        check_files(output / 'control', control['files_sha256'])
        control_runtime = read(output / 'control/manifest.json')['runtime']
    else:
        require(not (output / 'intervention').exists(), 'Wrong arm order')
    observed = hardware()
    require(observed == root['hardware'], 'Hardware changed between arms')
    folder = output / arm
    folder.mkdir()
    from pilot_training_v2.core import setup, matching_initial, make_optimizer, run_updates
    from pilot_training_v2.evaluation import CheckpointEvaluator
    try:
        model, tokenizer, encoded, before, runtime = setup(arm)
        require(runtime['model'] == root['model'], 'Loaded model differs')
        expected_encodings = read(PREFLIGHT / 'preflight.json')['public_encodings']
        require(all(digest(json.dumps(value, sort_keys=True)) == expected_encodings[key]['encoding_sha256']
                    for key, value in encoded.items()), 'Live serialization differs from frozen preflight')
        if control_runtime is not None:
            matching_initial(control_runtime, runtime)
        write(folder / 'manifest.json', {'arm': arm, 'started_at_utc': now(), 'runtime': runtime,
              'parent_manifest_sha256': sha256_file(output / 'manifest.json'),
              'identical_initial_adapters_checked_before_updates': arm == 'intervention',
              'expected_updates': 60, 'expected_microbatches': 240, 'expected_dev_generations': 80})
        optimizer, scheduler = make_optimizer(model)
        evaluator = CheckpointEvaluator(folder, arm, tokenizer)
        bindings = {'parent_manifest_sha256': sha256_file(output / 'manifest.json'),
                    'arm_manifest_sha256': sha256_file(folder / 'manifest.json'),
                    'schedule_sha256': sha256_file(ROOT / 'data/pilot-v2-training/schedule.json'),
                    'runner_release_sha256': sha256_file(HERE / 'release.json'),
                    'protocol_manifest_sha256': sha256_file(ROOT / 'data/pilot-v2-training/split_manifest.json')}
        result = run_updates(model, optimizer, scheduler, encoded, schedules()[arm], folder / 'training',
                             arm, before, bindings, evaluator)
        require(evaluator.completed_steps == list(range(6, 61, 6)), 'Missing dev checkpoint evaluations')
        require(inputs() == root['inputs_sha256'] and check_model() == root['model'], 'End-of-arm input/model drift')
        write(folder / 'completion.json', {'status': 'COMPLETE', 'arm': arm, **result,
              'dev_generations': 80, 'base_weights_unchanged': before == result['base_hashes_after'],
              'finished_at_utc': now(), 'files_sha256': sealed_files(folder)})
    except BaseException as error:
        write(folder / 'interruption.json', {'status': 'INTERRUPTED', 'error': str(error),
              'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'freeze-runner', 'check-ready', 'run', 'worker', 'check', 'report', 'freeze-results'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--replay-tokens', action='store_true')
    parser.add_argument('--reverify', action='store_true')
    args = parser.parse_args()
    with sealed_io():
        if args.action in ('prepare', 'freeze-runner'):
            from pilot_training_v2.preflight import prepare, freeze
            result = (prepare if args.action == 'prepare' else freeze)(args.output or PREFLIGHT)
        elif args.action == 'check-ready':
            check_ready(args.output or PREFLIGHT)
            result = {'status': 'READY', 'qwen_training_started_by_check': False}
        elif args.action == 'run':
            result = run(args.output or OUTPUT)
        elif args.action == 'worker':
            worker(args.output or OUTPUT, args.arm); return
        elif args.action == 'check':
            from pilot_training_v2.audit import check
            result = check(args.output or OUTPUT, args.replay_tokens, args.reverify)
        else:
            from pilot_training_v2.report import write_report, freeze_results
            result = (write_report if args.action == 'report' else freeze_results)(args.output or OUTPUT)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
